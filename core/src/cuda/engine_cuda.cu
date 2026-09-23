// CUDA engine.
//
// Design notes
// * All per-pixel arithmetic comes from common/pixel_ops.hpp (shared with the CPU
//   engine), so the kernels below only decide how pixels map to threads.
// * Device buffers are cached across calls and only grow, so repeated flow() /
//   synthesize() calls on same-sized frames never touch cudaMalloc.
// * One LK iteration is three launches: a fused "warp + residual + normal
//   equation terms" kernel producing 5 planes, a box filter over those 5 planes,
//   and the per-pixel 2x2 solve.
// * Box filter variants (selected at construction; the benchmark compares them):
//     optimized: separable. Horizontal pass stages a row segment plus halo in
//                shared memory; vertical pass is a coalesced running sum where
//                each thread walks a strip of rows.
//     naive:     one thread per output pixel reading its whole (2r+1)^2 window
//                straight from global memory.
// * Splatting uses float atomicAdd into accumulation buffers.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/pixel_ops.hpp"
#include "flipster/engine.hpp"

namespace flipster {
namespace {

#define FL_CUDA_CHECK(expr)                                                                          \
  do {                                                                                               \
    const cudaError_t err_ = (expr);                                                                 \
    if (err_ != cudaSuccess)                                                                         \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) + " (" #expr \
                               ") at " __FILE__ ":" + std::to_string(__LINE__));                     \
  } while (0)

#define FL_LAUNCH_CHECK() FL_CUDA_CHECK(cudaGetLastError())

// ------------------------------------------------------------------ memory

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), cap_(o.cap_) { o.ptr_ = nullptr, o.cap_ = 0; }
  DeviceBuffer& operator=(DeviceBuffer&&) = delete;
  ~DeviceBuffer() {
    if (ptr_) cudaFree(ptr_);
  }

  float* get(size_t n) {  // grow-only
    if (n > cap_) {
      if (ptr_) FL_CUDA_CHECK(cudaFree(ptr_));
      ptr_ = nullptr;
      FL_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(float)));
      cap_ = n;
    }
    return ptr_;
  }
  float* ptr() const { return ptr_; }

 private:
  float* ptr_ = nullptr;
  size_t cap_ = 0;
};

void upload(float* dst, const float* src, size_t n) {
  FL_CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyHostToDevice));
}
void download(float* dst, const float* src, size_t n) {
  FL_CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost));
}

// ------------------------------------------------------------------ launch helpers

constexpr int kBX = 32, kBY = 8;
inline dim3 grid2d(int w, int h, int planes = 1) { return dim3((w + kBX - 1) / kBX, (h + kBY - 1) / kBY, planes); }
inline dim3 block2d() { return dim3(kBX, kBY); }
inline int blocks1d(size_t n, int b = 256) { return static_cast<int>((n + b - 1) / b); }

#define FL_XY                                        \
  const int x = blockIdx.x * blockDim.x + threadIdx.x; \
  const int y = blockIdx.y * blockDim.y + threadIdx.y; \
  if (x >= w || y >= h) return;                      \
  const int k = y * w + x;

// ------------------------------------------------------------------ pyramid kernels

__global__ void k_gauss_h(const float* __restrict__ in, float* __restrict__ out, int w, int h) {
  FL_XY
  const float* row = in + y * w;
  float s = 0.f;
#pragma unroll
  for (int d = -2; d <= 2; ++d) s += px::g5(d) * row[px::clampi(x + d, 0, w - 1)];
  out[k] = s;
}

__global__ void k_gauss_v(const float* __restrict__ in, float* __restrict__ out, int w, int h) {
  FL_XY
  float s = 0.f;
#pragma unroll
  for (int d = -2; d <= 2; ++d) s += px::g5(d) * in[px::clampi(y + d, 0, h - 1) * w + x];
  out[k] = s;
}

__global__ void k_decimate2(const float* __restrict__ in, int iw, float* __restrict__ out, int w, int h) {
  FL_XY
  out[k] = in[(2 * y) * iw + 2 * x];
}

__global__ void k_gradient(const float* __restrict__ img, float* __restrict__ ix, float* __restrict__ iy, int w,
                           int h) {
  FL_XY
  px::gradient(img, w, h, x, y, &ix[k], &iy[k]);
}

// ------------------------------------------------------------------ LK kernels

// Fused: warp I1 by the current flow, residual, and the 5 normal-equation terms.
__global__ void k_lk_terms(const float* __restrict__ i0, const float* __restrict__ i1, const float* __restrict__ ix,
                           const float* __restrict__ iy, const float* __restrict__ u, const float* __restrict__ v,
                           float* __restrict__ terms, int w, int h) {
  FL_XY
  const size_t n = static_cast<size_t>(w) * h;
  const float wx = static_cast<float>(x) + u[k];
  const float wy = static_cast<float>(y) + v[k];
  const bool inside = px::inside_image(wx, wy, w, h);
  const float i1w = px::sample_bilinear(i1, w, h, wx, wy);
  const px::LkTerms t = px::lk_terms(i0[k], i1w, ix[k], iy[k], u[k], v[k], inside);
  terms[k] = t.gxx;
  terms[n + k] = t.gxy;
  terms[2 * n + k] = t.gyy;
  terms[3 * n + k] = t.bx;
  terms[4 * n + k] = t.by;
}

__global__ void k_lk_update(const float* __restrict__ boxed, float* __restrict__ u, float* __restrict__ v, int n,
                            float damping, float damping_floor, float zero_pull, float max_step) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  float uk = u[k], vk = v[k];
  px::lk_update(boxed[k], boxed[n + k], boxed[2 * n + k], boxed[3 * n + k], boxed[4 * n + k], damping, damping_floor,
                zero_pull, max_step, &uk, &vk);
  u[k] = uk;
  v[k] = vk;
}

__global__ void k_median3x3(const float* __restrict__ in, float* __restrict__ out, int w, int h) {
  FL_XY
  out[k] = px::median3x3_at(in, w, h, x, y);
}

__global__ void k_upsample_flow(const float* __restrict__ uc, const float* __restrict__ vc, int cw, int ch,
                                float* __restrict__ u, float* __restrict__ v, int w, int h) {
  FL_XY
  const float cx = px::upsample_coord(x), cy = px::upsample_coord(y);
  u[k] = 2.f * px::sample_bilinear(uc, cw, ch, cx, cy);
  v[k] = 2.f * px::sample_bilinear(vc, cw, ch, cx, cy);
}

// ------------------------------------------------------------------ box filters

constexpr int kRowTile = 256;
constexpr int kColStrip = 32;

// Horizontal window sums; one block = kRowTile pixels of one row of one plane.
__global__ void k_box_h_shared(const float* __restrict__ in, float* __restrict__ out, int w, int h, int r) {
  extern __shared__ float tile[];
  const int y = blockIdx.y;
  const size_t plane = static_cast<size_t>(blockIdx.z) * w * h;
  const float* row = in + plane + static_cast<size_t>(y) * w;
  const int x0 = blockIdx.x * kRowTile;
  for (int i = threadIdx.x; i < kRowTile + 2 * r; i += blockDim.x)
    tile[i] = row[px::clampi(x0 - r + i, 0, w - 1)];
  __syncthreads();
  const int x = x0 + threadIdx.x;
  if (x >= w) return;
  float s = 0.f;
  for (int d = 0; d <= 2 * r; ++d) s += tile[threadIdx.x + d];
  out[plane + static_cast<size_t>(y) * w + x] = s;
}

// Vertical running sums; each thread owns one column and walks kColStrip rows.
// Adjacent threads read adjacent columns, so every row access is coalesced.
__global__ void k_box_v_running(const float* __restrict__ in, float* __restrict__ out, int w, int h, int r,
                                float norm) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= w) return;
  const size_t plane = static_cast<size_t>(blockIdx.z) * w * h;
  const float* src = in + plane;
  float* dst = out + plane;
  const int y0 = blockIdx.y * kColStrip;
  const int y1 = min(h, y0 + kColStrip);
  float s = 0.f;
  for (int d = -r; d <= r; ++d) s += src[px::clampi(y0 + d, 0, h - 1) * w + x];
  for (int y = y0; y < y1; ++y) {
    dst[y * w + x] = s * norm;
    s += src[px::clampi(y + r + 1, 0, h - 1) * w + x] - src[px::clampi(y - r, 0, h - 1) * w + x];
  }
}

// Baseline: every output pixel reads its whole window from global memory.
__global__ void k_box_naive(const float* __restrict__ in, float* __restrict__ out, int w, int h, int r, float norm) {
  FL_XY
  const size_t plane = static_cast<size_t>(blockIdx.z) * w * h;
  float s = 0.f;
  for (int dy = -r; dy <= r; ++dy) {
    const float* row = in + plane + static_cast<size_t>(px::clampi(y + dy, 0, h - 1)) * w;
    for (int dx = -r; dx <= r; ++dx) s += row[px::clampi(x + dx, 0, w - 1)];
  }
  out[plane + k] = s * norm;
}

// ------------------------------------------------------------------ synthesis kernels

__global__ void k_warp(const float* __restrict__ img, const float* __restrict__ u, const float* __restrict__ v,
                       float* __restrict__ out, int w, int h, int channels) {
  FL_XY
  const size_t n = static_cast<size_t>(w) * h;
  const float sx = static_cast<float>(x) + u[k], sy = static_cast<float>(y) + v[k];
  for (int c = 0; c < channels; ++c) out[c * n + k] = px::sample_bilinear(img + c * n, w, h, sx, sy);
}

__global__ void k_weights(const float* __restrict__ imp, float beta, float* __restrict__ out, int n) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  out[k] = imp ? expf(beta * imp[k]) : 1.f;
}

__global__ void k_fb_reliability(const float* __restrict__ fu, const float* __restrict__ fv,
                                 const float* __restrict__ bu, const float* __restrict__ bv, float* __restrict__ out,
                                 int w, int h, float alpha, float beta) {
  FL_XY
  const float u = fu[k], v = fv[k];
  const float sx = static_cast<float>(x) + u, sy = static_cast<float>(y) + v;
  out[k] = px::fb_reliability(u, v, px::sample_bilinear(bu, w, h, sx, sy), px::sample_bilinear(bv, w, h, sx, sy),
                              alpha, beta);
}

// acc layout: [C color planes][weight][reliability]
__global__ void k_splat(const float* __restrict__ color, const float* __restrict__ weight,
                        const float* __restrict__ rel, const float* __restrict__ u, const float* __restrict__ v,
                        float t, float* __restrict__ acc, int w, int h, int channels) {
  FL_XY
  const size_t n = static_cast<size_t>(w) * h;
  const float qx = static_cast<float>(x) + t * u[k];
  const float qy = static_cast<float>(y) + t * v[k];
  const int x0 = static_cast<int>(floorf(qx)), y0 = static_cast<int>(floorf(qy));
  const float fx = qx - static_cast<float>(x0), fy = qy - static_cast<float>(y0);
  const float bws[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
  const float wk = weight[k], rk = rel[k];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int tx = x0 + (j & 1), ty = y0 + (j >> 1);
    if (tx < 0 || tx >= w || ty < 0 || ty >= h) continue;
    const size_t q = static_cast<size_t>(ty) * w + tx;
    const float ww = wk * bws[j];
    atomicAdd(&acc[channels * n + q], ww);
    atomicAdd(&acc[(channels + 1) * n + q], ww * rk);
    for (int c = 0; c < channels; ++c) atomicAdd(&acc[c * n + q], ww * color[c * n + k]);
  }
}

__global__ void k_compose(const float* __restrict__ c0, const float* __restrict__ c1, const float* __restrict__ a0,
                          const float* __restrict__ a1, float t, float min_rel, float* __restrict__ out, int n,
                          int channels) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  const float aw0 = a0[channels * n + k], aw1 = a1[channels * n + k];
  const bool cov0 = aw0 > kCoverageEps, cov1 = aw1 > kCoverageEps;
  const float w0 = cov0 ? aw0 : 1.f, w1 = cov1 ? aw1 : 1.f;
  const float k0 = cov0 ? (1.f - t) * (min_rel + (1.f - min_rel) * a0[(channels + 1) * n + k] / w0) : 0.f;
  const float k1 = cov1 ? t * (min_rel + (1.f - min_rel) * a1[(channels + 1) * n + k] / w1) : 0.f;
  const float tot = k0 + k1;
  for (int c = 0; c < channels; ++c) {
    out[c * n + k] = tot > 0.f ? (k0 * a0[c * n + k] / w0 + k1 * a1[c * n + k] / w1) / tot
                               : (1.f - t) * c0[c * n + k] + t * c1[c * n + k];
  }
}

// ------------------------------------------------------------------ timing

class GpuTimer {
 public:
  GpuTimer() {
    FL_CUDA_CHECK(cudaEventCreate(&a_));
    FL_CUDA_CHECK(cudaEventCreate(&b_));
  }
  ~GpuTimer() {
    cudaEventDestroy(a_);
    cudaEventDestroy(b_);
  }
  void start() { FL_CUDA_CHECK(cudaEventRecord(a_)); }
  double stop_ms() {
    FL_CUDA_CHECK(cudaEventRecord(b_));
    FL_CUDA_CHECK(cudaEventSynchronize(b_));
    float ms = 0.f;
    FL_CUDA_CHECK(cudaEventElapsedTime(&ms, a_, b_));
    return ms;
  }

 private:
  cudaEvent_t a_{}, b_{};
};

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ------------------------------------------------------------------ engine

class CudaEngine final : public Engine {
 public:
  CudaEngine(bool naive_box, int device) : naive_(naive_box), device_(device) {
    FL_CUDA_CHECK(cudaSetDevice(device_));
    FL_CUDA_CHECK(cudaFree(nullptr));  // force context creation now, not inside a timed call
  }

  std::string name() const override { return naive_ ? "cuda-naive" : "cuda"; }

  Flow flow(const Plane& a, const Plane& b, const FlowParams& p) override {
    require_same_size(a.width, a.height, b.width, b.height, "flow");
    FL_CUDA_CHECK(cudaSetDevice(device_));
    timings_.clear();
    const auto wall0 = Clock::now();
    const int levels = p.resolve_levels(a.height, a.width);

    // Level sizes.
    std::vector<int> ws{a.width}, hs{a.height};
    for (int l = 1; l < levels; ++l) {
      ws.push_back((ws.back() + 1) / 2);
      hs.push_back((hs.back() + 1) / 2);
    }
    if (static_cast<int>(pyr0_.size()) < levels) {
      pyr0_.resize(levels);
      pyr1_.resize(levels);
    }
    const size_t n0 = static_cast<size_t>(a.width) * a.height;

    GpuTimer timer;
    timer.start();
    float* raw0 = scratch_a_.get(n0);
    float* raw1 = scratch_b_.get(n0);
    upload(raw0, a.ptr(), n0);
    upload(raw1, b.ptr(), n0);
    timings_["upload_ms"] = timer.stop_ms();

    timer.start();
    float* tmp = tmp_.get(n0);
    for (int l = 0; l < levels; ++l) {
      const size_t n = static_cast<size_t>(ws[l]) * hs[l];
      float* d0 = pyr0_[l].get(n);
      float* d1 = pyr1_[l].get(n);
      if (l == 0) {
        blur(raw0, tmp, d0, ws[0], hs[0]);
        blur(raw1, tmp, d1, ws[0], hs[0]);
      } else {
        // blur level l-1 into scratch, then keep every second pixel.
        float* s0 = scratch_a_.get(static_cast<size_t>(ws[l - 1]) * hs[l - 1]);
        float* s1 = scratch_b_.get(static_cast<size_t>(ws[l - 1]) * hs[l - 1]);
        blur(pyr0_[l - 1].ptr(), tmp, s0, ws[l - 1], hs[l - 1]);
        blur(pyr1_[l - 1].ptr(), tmp, s1, ws[l - 1], hs[l - 1]);
        k_decimate2<<<grid2d(ws[l], hs[l]), block2d()>>>(s0, ws[l - 1], d0, ws[l], hs[l]);
        k_decimate2<<<grid2d(ws[l], hs[l]), block2d()>>>(s1, ws[l - 1], d1, ws[l], hs[l]);
        FL_LAUNCH_CHECK();
      }
    }
    const double pyramid_ms = timer.stop_ms();

    timer.start();
    float* u = u_[0].get(n0);
    float* v = v_[0].get(n0);
    float* u2 = u_[1].get(n0);
    float* v2 = v_[1].get(n0);
    float* ix = ix_.get(n0);
    float* iy = iy_.get(n0);
    float* terms = terms_.get(5 * n0);
    float* boxed = boxed_.get(5 * n0);
    float* boxtmp = boxtmp_.get(5 * n0);
    const int top = levels - 1;
    FL_CUDA_CHECK(cudaMemset(u, 0, static_cast<size_t>(ws[top]) * hs[top] * sizeof(float)));
    FL_CUDA_CHECK(cudaMemset(v, 0, static_cast<size_t>(ws[top]) * hs[top] * sizeof(float)));

    for (int l = top; l >= 0; --l) {
      const int w = ws[l], h = hs[l];
      const int n = w * h;
      if (l != top) {
        k_upsample_flow<<<grid2d(w, h), block2d()>>>(u, v, ws[l + 1], hs[l + 1], u2, v2, w, h);
        FL_LAUNCH_CHECK();
        std::swap(u, u2);
        std::swap(v, v2);
      }
      k_gradient<<<grid2d(w, h), block2d()>>>(pyr0_[l].ptr(), ix, iy, w, h);
      FL_LAUNCH_CHECK();
      const int iters = p.iterations * (l == top ? kCoarseIterMult : 1);
      for (int it = 0; it < iters; ++it) {
        k_lk_terms<<<grid2d(w, h), block2d()>>>(pyr0_[l].ptr(), pyr1_[l].ptr(), ix, iy, u, v, terms, w, h);
        FL_LAUNCH_CHECK();
        box_mean(terms, boxtmp, boxed, w, h, 5, p.window_radius);
        k_lk_update<<<blocks1d(n), 256>>>(boxed, u, v, n, p.damping, p.damping_floor, p.zero_pull, p.max_step);
        FL_LAUNCH_CHECK();
      }
      if (p.median) {
        k_median3x3<<<grid2d(w, h), block2d()>>>(u, u2, w, h);
        k_median3x3<<<grid2d(w, h), block2d()>>>(v, v2, w, h);
        FL_LAUNCH_CHECK();
        std::swap(u, u2);
        std::swap(v, v2);
      }
    }
    const double lk_ms = timer.stop_ms();

    timer.start();
    Flow f(a.width, a.height);
    download(f.u.ptr(), u, n0);
    download(f.v.ptr(), v, n0);
    timings_["download_ms"] = timer.stop_ms();
    timings_["pyramid_ms"] = pyramid_ms;
    timings_["lk_ms"] = lk_ms;
    timings_["kernel_ms"] = pyramid_ms + lk_ms;
    timings_["flow_ms"] = ms_since(wall0);
    return f;
  }

  Image warp(const Image& img, const Flow& f) override {
    require_same_size(img.width, img.height, f.width(), f.height(), "warp");
    FL_CUDA_CHECK(cudaSetDevice(device_));
    const int w = img.width, h = img.height, C = img.channels;
    const size_t n = img.plane_size();
    float* dimg = scratch_a_.get(n * C);
    float* dout = scratch_b_.get(n * C);
    float* du = u_[0].get(n);
    float* dv = v_[0].get(n);
    upload(dimg, img.data.data(), n * C);
    upload(du, f.u.ptr(), n);
    upload(dv, f.v.ptr(), n);
    k_warp<<<grid2d(w, h), block2d()>>>(dimg, du, dv, dout, w, h, C);
    FL_LAUNCH_CHECK();
    Image out(w, h, C);
    download(out.data.data(), dout, n * C);
    return out;
  }

  void set_pair(const Image& c0, const Image& c1, const Plane& imp0, const Plane& imp1, const Flow& f01,
                const Flow& f10, const SplatParams& sp) override {
    require_same_size(c0.width, c0.height, c1.width, c1.height, "set_pair");
    require_same_size(c0.width, c0.height, f01.width(), f01.height(), "set_pair");
    require_same_size(c0.width, c0.height, f10.width(), f10.height(), "set_pair");
    if (c0.channels != c1.channels) throw std::invalid_argument("set_pair: channel mismatch");
    FL_CUDA_CHECK(cudaSetDevice(device_));
    timings_.clear();
    const auto wall0 = Clock::now();
    w_ = c0.width;
    h_ = c0.height;
    ch_ = c0.channels;
    sp_ = sp;
    const size_t n = c0.plane_size();
    upload(c0_.get(n * ch_), c0.data.data(), n * ch_);
    upload(c1_.get(n * ch_), c1.data.data(), n * ch_);
    upload(f01u_.get(n), f01.u.ptr(), n);
    upload(f01v_.get(n), f01.v.ptr(), n);
    upload(f10u_.get(n), f10.u.ptr(), n);
    upload(f10v_.get(n), f10.v.ptr(), n);
    load_weights(imp0, w0_, n, sp.softmax_beta);
    load_weights(imp1, w1_, n, sp.softmax_beta);
    k_fb_reliability<<<grid2d(w_, h_), block2d()>>>(f01u_.ptr(), f01v_.ptr(), f10u_.ptr(), f10v_.ptr(), r0_.get(n),
                                                     w_, h_, sp.fb_alpha, sp.fb_beta);
    k_fb_reliability<<<grid2d(w_, h_), block2d()>>>(f10u_.ptr(), f10v_.ptr(), f01u_.ptr(), f01v_.ptr(), r1_.get(n),
                                                     w_, h_, sp.fb_alpha, sp.fb_beta);
    FL_LAUNCH_CHECK();
    FL_CUDA_CHECK(cudaDeviceSynchronize());
    loaded_ = true;
    timings_["set_pair_ms"] = ms_since(wall0);
  }

  Image synthesize(float t) override {
    if (!loaded_) throw std::runtime_error("call set_pair() first");
    FL_CUDA_CHECK(cudaSetDevice(device_));
    timings_.clear();
    const auto wall0 = Clock::now();
    const size_t n = static_cast<size_t>(w_) * h_;
    const size_t acc_n = n * (ch_ + 2);
    float* a0 = acc0_.get(acc_n);
    float* a1 = acc1_.get(acc_n);
    float* out = out_.get(n * ch_);
    GpuTimer timer;
    timer.start();
    FL_CUDA_CHECK(cudaMemset(a0, 0, acc_n * sizeof(float)));
    FL_CUDA_CHECK(cudaMemset(a1, 0, acc_n * sizeof(float)));
    k_splat<<<grid2d(w_, h_), block2d()>>>(c0_.ptr(), w0_.ptr(), r0_.ptr(), f01u_.ptr(), f01v_.ptr(), t, a0, w_, h_,
                                            ch_);
    k_splat<<<grid2d(w_, h_), block2d()>>>(c1_.ptr(), w1_.ptr(), r1_.ptr(), f10u_.ptr(), f10v_.ptr(), 1.f - t, a1, w_,
                                            h_, ch_);
    k_compose<<<blocks1d(n), 256>>>(c0_.ptr(), c1_.ptr(), a0, a1, t, sp_.min_reliability, out, static_cast<int>(n),
                                    ch_);
    FL_LAUNCH_CHECK();
    timings_["kernel_ms"] = timer.stop_ms();
    Image img(w_, h_, ch_);
    download(img.data.data(), out, n * ch_);
    timings_["synth_ms"] = ms_since(wall0);
    return img;
  }

  std::pair<Plane, Plane> reliability() override {
    if (!loaded_) throw std::runtime_error("call set_pair() first");
    Plane a(w_, h_), b(w_, h_);
    download(a.ptr(), r0_.ptr(), a.size());
    download(b.ptr(), r1_.ptr(), b.size());
    return {a, b};
  }

 private:
  void blur(const float* in, float* tmp, float* out, int w, int h) {
    k_gauss_h<<<grid2d(w, h), block2d()>>>(in, tmp, w, h);
    k_gauss_v<<<grid2d(w, h), block2d()>>>(tmp, out, w, h);
    FL_LAUNCH_CHECK();
  }

  // Normalized box mean over `planes` consecutive w*h planes.
  void box_mean(const float* in, float* tmp, float* out, int w, int h, int planes, int r) {
    const float norm = 1.f / static_cast<float>((2 * r + 1) * (2 * r + 1));
    if (naive_) {
      k_box_naive<<<grid2d(w, h, planes), block2d()>>>(in, out, w, h, r, norm);
    } else {
      const dim3 gh((w + kRowTile - 1) / kRowTile, h, planes);
      k_box_h_shared<<<gh, kRowTile, (kRowTile + 2 * r) * sizeof(float)>>>(in, tmp, w, h, r);
      const dim3 gv((w + 127) / 128, (h + kColStrip - 1) / kColStrip, planes);
      k_box_v_running<<<gv, 128>>>(tmp, out, w, h, r, norm);
    }
    FL_LAUNCH_CHECK();
  }

  void load_weights(const Plane& imp, DeviceBuffer& dst, size_t n, float beta) {
    float* d = dst.get(n);
    float* src = nullptr;
    if (!imp.empty()) {
      require_same_size(imp.width, imp.height, w_, h_, "importance");
      src = imp_tmp_.get(n);
      upload(src, imp.ptr(), n);
    }
    k_weights<<<blocks1d(n), 256>>>(src, beta, d, static_cast<int>(n));
    FL_LAUNCH_CHECK();
  }

  bool naive_;
  int device_;
  // flow
  std::vector<DeviceBuffer> pyr0_, pyr1_;
  DeviceBuffer scratch_a_, scratch_b_, tmp_, ix_, iy_, terms_, boxed_, boxtmp_;
  DeviceBuffer u_[2], v_[2];
  // synthesis
  int w_ = 0, h_ = 0, ch_ = 0;
  SplatParams sp_;
  bool loaded_ = false;
  DeviceBuffer c0_, c1_, f01u_, f01v_, f10u_, f10v_, w0_, w1_, r0_, r1_, acc0_, acc1_, out_, imp_tmp_;
};

}  // namespace

bool cuda_compiled() { return true; }

bool cuda_available() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();  // clear the sticky error from a missing driver
    return false;
  }
  return count > 0;
}

std::string cuda_device_name() {
  if (!cuda_available()) return "";
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return "";
  return std::string(prop.name) + " (sm_" + std::to_string(prop.major) + std::to_string(prop.minor) + ")";
}

std::unique_ptr<Engine> make_cuda_engine(const std::string& variant, int device) {
  if (!cuda_available()) throw std::runtime_error("no CUDA device available");
  if (variant != "optimized" && variant != "naive")
    throw std::invalid_argument("unknown CUDA variant '" + variant + "' (expected 'optimized' or 'naive')");
  return std::make_unique<CudaEngine>(variant == "naive", device);
}

}  // namespace flipster
