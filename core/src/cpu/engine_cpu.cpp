// CPU engine: C++17 + OpenMP. Same arithmetic as the NumPy reference and the
// CUDA kernels (shared through common/pixel_ops.hpp).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "common/pixel_ops.hpp"
#include "flipster/engine.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace flipster {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ------------------------------------------------------------------ filters

// Separable [1 4 6 4 1]/16 blur with replicate borders.
Plane gauss5(const Plane& in) {
  const int w = in.width, h = in.height;
  Plane tmp(w, h), out(w, h);
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    const float* row = in.ptr() + static_cast<size_t>(y) * w;
    float* dst = tmp.ptr() + static_cast<size_t>(y) * w;
    for (int x = 0; x < w; ++x) {
      float s = 0.f;
      for (int k = -2; k <= 2; ++k) s += px::g5(k) * row[px::clampi(x + k, 0, w - 1)];
      dst[x] = s;
    }
  }
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    float* dst = out.ptr() + static_cast<size_t>(y) * w;
    for (int x = 0; x < w; ++x) {
      float s = 0.f;
      for (int k = -2; k <= 2; ++k) s += px::g5(k) * tmp.at(x, px::clampi(y + k, 0, h - 1));
      dst[x] = s;
    }
  }
  return out;
}

Plane downsample2(const Plane& in) {
  Plane blurred = gauss5(in);
  const int w = (in.width + 1) / 2, h = (in.height + 1) / 2;
  Plane out(w, h);
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) out.at(x, y) = blurred.at(2 * x, 2 * y);
  return out;
}

// Normalized (2r+1)^2 box mean with replicate borders, separable running sums
// accumulated in double (matches cv2.boxFilter on float32).
void box_mean(const float* in, float* out, int w, int h, int r, std::vector<double>& scratch) {
  scratch.resize(static_cast<size_t>(w) * h);
  double* rows = scratch.data();
  const double norm = 1.0 / (static_cast<double>(2 * r + 1) * (2 * r + 1));
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    const float* src = in + static_cast<size_t>(y) * w;
    double* dst = rows + static_cast<size_t>(y) * w;
    double s = 0.0;
    for (int k = -r; k <= r; ++k) s += src[px::clampi(k, 0, w - 1)];
    for (int x = 0; x < w; ++x) {
      dst[x] = s;
      s += src[px::clampi(x + r + 1, 0, w - 1)] - src[px::clampi(x - r, 0, w - 1)];
    }
  }
  // Vertical pass over column strips so each thread streams whole rows.
  const int strip = 64;
  const int nstrips = (w + strip - 1) / strip;
#pragma omp parallel for schedule(static)
  for (int si = 0; si < nstrips; ++si) {
    const int x0 = si * strip, x1 = std::min(w, x0 + strip);
    double acc[64];
    for (int x = x0; x < x1; ++x) {
      double s = 0.0;
      for (int k = -r; k <= r; ++k) s += rows[static_cast<size_t>(px::clampi(k, 0, h - 1)) * w + x];
      acc[x - x0] = s;
    }
    for (int y = 0; y < h; ++y) {
      const double* add = rows + static_cast<size_t>(px::clampi(y + r + 1, 0, h - 1)) * w;
      const double* sub = rows + static_cast<size_t>(px::clampi(y - r, 0, h - 1)) * w;
      float* dst = out + static_cast<size_t>(y) * w;
      for (int x = x0; x < x1; ++x) {
        dst[x] = static_cast<float>(acc[x - x0] * norm);
        acc[x - x0] += add[x] - sub[x];
      }
    }
  }
}

Plane median3x3(const Plane& in) {
  Plane out(in.width, in.height);
#pragma omp parallel for schedule(static)
  for (int y = 0; y < in.height; ++y)
    for (int x = 0; x < in.width; ++x) out.at(x, y) = px::median3x3_at(in.ptr(), in.width, in.height, x, y);
  return out;
}

void upsample_flow(Flow& f, int w, int h) {
  Flow out(w, h);
  const int cw = f.width(), ch = f.height();
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    const float cy = px::upsample_coord(y);
    for (int x = 0; x < w; ++x) {
      const float cx = px::upsample_coord(x);
      out.u.at(x, y) = 2.f * px::sample_bilinear(f.u.ptr(), cw, ch, cx, cy);
      out.v.at(x, y) = 2.f * px::sample_bilinear(f.v.ptr(), cw, ch, cx, cy);
    }
  }
  f = std::move(out);
}

// ------------------------------------------------------------------ LK

void lk_level(const Plane& i0, const Plane& i1, Flow& f, const FlowParams& p, int iterations) {
  const int w = i0.width, h = i0.height;
  const size_t n = static_cast<size_t>(w) * h;
  Plane ix(w, h), iy(w, h);
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) px::gradient(i0.ptr(), w, h, x, y, &ix.at(x, y), &iy.at(x, y));

  std::vector<float> terms(5 * n), boxed(5 * n);
  std::vector<double> scratch;
  float* u = f.u.ptr();
  float* v = f.v.ptr();

  for (int it = 0; it < iterations; ++it) {
#pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const size_t k = static_cast<size_t>(y) * w + x;
        const float wx = static_cast<float>(x) + u[k];
        const float wy = static_cast<float>(y) + v[k];
        const bool inside = px::inside_image(wx, wy, w, h);
        const float i1w = px::sample_bilinear(i1.ptr(), w, h, wx, wy);
        const px::LkTerms t = px::lk_terms(i0.ptr()[k], i1w, ix.ptr()[k], iy.ptr()[k], u[k], v[k], inside);
        terms[k] = t.gxx;
        terms[n + k] = t.gxy;
        terms[2 * n + k] = t.gyy;
        terms[3 * n + k] = t.bx;
        terms[4 * n + k] = t.by;
      }
    }
    for (int c = 0; c < 5; ++c) box_mean(terms.data() + c * n, boxed.data() + c * n, w, h, p.window_radius, scratch);
    const long long nn = static_cast<long long>(n);
#pragma omp parallel for schedule(static)
    for (long long k = 0; k < nn; ++k) {
      px::lk_update(boxed[k], boxed[n + k], boxed[2 * n + k], boxed[3 * n + k], boxed[4 * n + k], p.damping,
                    p.damping_floor, p.zero_pull, p.max_step, &u[k], &v[k]);
    }
  }
  if (p.median) {
    f.u = median3x3(f.u);
    f.v = median3x3(f.v);
  }
}

// ------------------------------------------------------------------ synthesis

Plane fb_reliability(const Flow& fwd, const Flow& bwd, const SplatParams& sp) {
  const int w = fwd.width(), h = fwd.height();
  Plane out(w, h);
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float u = fwd.u.at(x, y), v = fwd.v.at(x, y);
      const float sx = static_cast<float>(x) + u, sy = static_cast<float>(y) + v;
      const float bu = px::sample_bilinear(bwd.u.ptr(), w, h, sx, sy);
      const float bv = px::sample_bilinear(bwd.v.ptr(), w, h, sx, sy);
      out.at(x, y) = px::fb_reliability(u, v, bu, bv, sp.fb_alpha, sp.fb_beta);
    }
  }
  return out;
}

struct Accum {
  int c = 0;
  std::vector<double> color, weight, rel;
  void reset(size_t n, int channels) {
    c = channels;
    color.assign(n * channels, 0.0);
    weight.assign(n, 0.0);
    rel.assign(n, 0.0);
  }
};

void splat(const Image& color, const Plane& weight, const Plane& rel, const Flow& f, float t, Accum& acc) {
  const int w = color.width, h = color.height, C = color.channels;
  const size_t n = color.plane_size();
  acc.reset(n, C);
  double* aw = acc.weight.data();
  double* ar = acc.rel.data();
  double* ac = acc.color.data();
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const size_t k = static_cast<size_t>(y) * w + x;
      const float qx = static_cast<float>(x) + t * f.u.ptr()[k];
      const float qy = static_cast<float>(y) + t * f.v.ptr()[k];
      const int x0 = static_cast<int>(std::floor(qx));
      const int y0 = static_cast<int>(std::floor(qy));
      const float fx = qx - static_cast<float>(x0);
      const float fy = qy - static_cast<float>(y0);
      const float bws[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
      const int dxs[4] = {0, 1, 0, 1};
      const int dys[4] = {0, 0, 1, 1};
      for (int j = 0; j < 4; ++j) {
        const int tx = x0 + dxs[j], ty = y0 + dys[j];
        if (tx < 0 || tx >= w || ty < 0 || ty >= h) continue;
        const size_t q = static_cast<size_t>(ty) * w + tx;
        const float ww = weight.ptr()[k] * bws[j];
#pragma omp atomic
        aw[q] += static_cast<double>(ww);
        const float wr = ww * rel.ptr()[k];
#pragma omp atomic
        ar[q] += static_cast<double>(wr);
        for (int c = 0; c < C; ++c) {
          const float wc = ww * color.plane(c)[k];
#pragma omp atomic
          ac[c * n + q] += static_cast<double>(wc);
        }
      }
    }
  }
}

Image compose(const Image& c0, const Image& c1, const Accum& s0, const Accum& s1, float t, const SplatParams& sp) {
  const int C = c0.channels;
  const size_t n = c0.plane_size();
  Image out(c0.width, c0.height, C);
  const double m = sp.min_reliability;
  const double tt = t;
  const long long nn = static_cast<long long>(n);
#pragma omp parallel for schedule(static)
  for (long long kk = 0; kk < nn; ++kk) {
    const size_t k = static_cast<size_t>(kk);
    const bool cov0 = s0.weight[k] > kCoverageEps, cov1 = s1.weight[k] > kCoverageEps;
    const double w0 = cov0 ? s0.weight[k] : 1.0, w1 = cov1 ? s1.weight[k] : 1.0;
    const double k0 = cov0 ? (1 - tt) * (m + (1 - m) * s0.rel[k] / w0) : 0.0;
    const double k1 = cov1 ? tt * (m + (1 - m) * s1.rel[k] / w1) : 0.0;
    const double tot = k0 + k1;
    for (int c = 0; c < C; ++c) {
      double val;
      if (tot > 0) {
        val = (k0 * s0.color[c * n + k] / w0 + k1 * s1.color[c * n + k] / w1) / tot;
      } else {
        val = (1 - tt) * c0.plane(c)[k] + tt * c1.plane(c)[k];
      }
      out.plane(c)[k] = static_cast<float>(val);
    }
  }
  return out;
}

// ------------------------------------------------------------------ engine

class CpuEngine final : public Engine {
 public:
  std::string name() const override { return "cpu"; }

  Flow flow(const Plane& a, const Plane& b, const FlowParams& p) override {
    require_same_size(a.width, a.height, b.width, b.height, "flow");
    timings_.clear();
    const auto t0 = Clock::now();
    const int levels = p.resolve_levels(a.height, a.width);
    std::vector<Plane> pyr0{gauss5(a)}, pyr1{gauss5(b)};
    for (int l = 1; l < levels; ++l) {
      pyr0.push_back(downsample2(pyr0.back()));
      pyr1.push_back(downsample2(pyr1.back()));
    }
    timings_["pyramid_ms"] = ms_since(t0);
    const auto t1 = Clock::now();
    Flow f(pyr0.back().width, pyr0.back().height);
    for (int l = levels - 1; l >= 0; --l) {
      if (f.width() != pyr0[l].width || f.height() != pyr0[l].height) upsample_flow(f, pyr0[l].width, pyr0[l].height);
      const int iters = p.iterations * (l == levels - 1 ? kCoarseIterMult : 1);
      lk_level(pyr0[l], pyr1[l], f, p, iters);
    }
    timings_["lk_ms"] = ms_since(t1);
    timings_["flow_ms"] = ms_since(t0);
    return f;
  }

  Image warp(const Image& img, const Flow& f) override {
    require_same_size(img.width, img.height, f.width(), f.height(), "warp");
    const int w = img.width, h = img.height;
    Image out(w, h, img.channels);
#pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const float sx = static_cast<float>(x) + f.u.at(x, y), sy = static_cast<float>(y) + f.v.at(x, y);
        for (int c = 0; c < img.channels; ++c)
          out.plane(c)[static_cast<size_t>(y) * w + x] = px::sample_bilinear(img.plane(c), w, h, sx, sy);
      }
    return out;
  }

  void set_pair(const Image& c0, const Image& c1, const Plane& imp0, const Plane& imp1, const Flow& f01,
                const Flow& f10, const SplatParams& sp) override {
    require_same_size(c0.width, c0.height, c1.width, c1.height, "set_pair");
    require_same_size(c0.width, c0.height, f01.width(), f01.height(), "set_pair");
    require_same_size(c0.width, c0.height, f10.width(), f10.height(), "set_pair");
    timings_.clear();
    const auto t0 = Clock::now();
    c0_ = c0;
    c1_ = c1;
    f01_ = f01;
    f10_ = f10;
    sp_ = sp;
    w0_ = weights(imp0, c0.width, c0.height, sp.softmax_beta);
    w1_ = weights(imp1, c0.width, c0.height, sp.softmax_beta);
    r0_ = fb_reliability(f01, f10, sp);
    r1_ = fb_reliability(f10, f01, sp);
    loaded_ = true;
    timings_["set_pair_ms"] = ms_since(t0);
  }

  Image synthesize(float t) override {
    if (!loaded_) throw std::runtime_error("call set_pair() first");
    timings_.clear();
    const auto t0 = Clock::now();
    splat(c0_, w0_, r0_, f01_, t, a0_);
    splat(c1_, w1_, r1_, f10_, 1.f - t, a1_);
    Image out = compose(c0_, c1_, a0_, a1_, t, sp_);
    timings_["synth_ms"] = ms_since(t0);
    return out;
  }

  std::pair<Plane, Plane> reliability() override {
    if (!loaded_) throw std::runtime_error("call set_pair() first");
    return {r0_, r1_};
  }

 private:
  static Plane weights(const Plane& imp, int w, int h, float beta) {
    Plane out(w, h, 1.f);
    if (imp.empty()) return out;
    require_same_size(imp.width, imp.height, w, h, "importance");
    for (size_t k = 0; k < out.size(); ++k) out.data[k] = std::exp(beta * imp.data[k]);
    return out;
  }

  Image c0_, c1_;
  Flow f01_, f10_;
  Plane w0_, w1_, r0_, r1_;
  SplatParams sp_;
  Accum a0_, a1_;
  bool loaded_ = false;
};

}  // namespace

std::unique_ptr<Engine> make_cpu_engine() { return std::make_unique<CpuEngine>(); }

}  // namespace flipster
