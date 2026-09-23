// Per-pixel math shared verbatim by the CPU and CUDA engines.
//
// Everything that decides *what* a pixel's value is lives here as a
// __host__ __device__ inline function; the engines only decide *how* pixels are
// scheduled (OpenMP loops vs. CUDA thread blocks). That way the CPU unit tests
// exercise the exact arithmetic the GPU kernels run.
#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define FL_HD __host__ __device__ __forceinline__
#else
#define FL_HD inline
#endif

namespace flipster {
namespace px {

FL_HD int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
FL_HD float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Bilinear sample with clamp-to-edge. Matches reference.sample_bilinear.
FL_HD float sample_bilinear(const float* img, int w, int h, float x, float y) {
  x = clampf(x, 0.f, static_cast<float>(w - 1));
  y = clampf(y, 0.f, static_cast<float>(h - 1));
  const int x0 = static_cast<int>(floorf(x));
  const int y0 = static_cast<int>(floorf(y));
  const int x1 = x0 + 1 < w ? x0 + 1 : w - 1;
  const int y1 = y0 + 1 < h ? y0 + 1 : h - 1;
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
  const float top = img[y0 * w + x0] * (1.f - fx) + img[y0 * w + x1] * fx;
  const float bot = img[y1 * w + x0] * (1.f - fx) + img[y1 * w + x1] * fx;
  return top * (1.f - fy) + bot * fy;
}

// Central-difference gradient (correctly scaled by 1/2), replicate border.
FL_HD void gradient(const float* img, int w, int h, int x, int y, float* gx, float* gy) {
  const int xm = x > 0 ? x - 1 : 0, xp = x < w - 1 ? x + 1 : w - 1;
  const int ym = y > 0 ? y - 1 : 0, yp = y < h - 1 ? y + 1 : h - 1;
  *gx = (img[y * w + xp] - img[y * w + xm]) * 0.5f;
  *gy = (img[yp * w + x] - img[ym * w + x]) * 0.5f;
}

// Per-pixel terms of the masked, per-pixel-linearized LK normal equations.
// Outputs m*Ix^2, m*IxIy, m*Iy^2, Ix*res, Iy*res where
// res = m * (I1(x + u) - I0(x) - Ix u - Iy v) and m = 1 if the sample is inside I1.
struct LkTerms {
  float gxx, gxy, gyy, bx, by;
};

FL_HD LkTerms lk_terms(float i0, float i1_warped, float ix, float iy, float u, float v, bool inside) {
  LkTerms t;
  const float m = inside ? 1.f : 0.f;
  const float res = m * ((i1_warped - i0) - (ix * u + iy * v));
  t.gxx = m * ix * ix;
  t.gxy = m * ix * iy;
  t.gyy = m * iy * iy;
  t.bx = ix * res;
  t.by = iy * res;
  return t;
}

FL_HD bool inside_image(float x, float y, int w, int h) {
  return x >= 0.f && x <= static_cast<float>(w - 1) && y >= 0.f && y <= static_cast<float>(h - 1);
}

// Solve the damped 2x2 system for the new flow and apply the step clamp.
// Inputs are window means (box-filtered terms).
FL_HD void lk_update(float gxx, float gxy, float gyy, float bx, float by, float damping, float damping_floor,
                     float zero_pull, float max_step, float* u, float* v) {
  const float lam = damping * 0.5f * (gxx + gyy) + damping_floor;
  const float a = gxx + lam + zero_pull;
  const float c = gyy + lam + zero_pull;
  const float det = a * c - gxy * gxy;
  const float rx = lam * (*u) - bx;
  const float ry = lam * (*v) - by;
  const float nu = (c * rx - gxy * ry) / det;
  const float nv = (a * ry - gxy * rx) / det;
  *u += clampf(nu - *u, -max_step, max_step);
  *v += clampf(nv - *v, -max_step, max_step);
}

// Median of 9 values (exact; a small sorting network).
FL_HD void sort2(float& a, float& b) {
  const float lo = a < b ? a : b;
  const float hi = a < b ? b : a;
  a = lo;
  b = hi;
}

FL_HD float median9(float p0, float p1, float p2, float p3, float p4, float p5, float p6, float p7, float p8) {
  // Paeth / Devillard median-of-9 network (19 compare-exchanges).
  sort2(p1, p2); sort2(p4, p5); sort2(p7, p8);
  sort2(p0, p1); sort2(p3, p4); sort2(p6, p7);
  sort2(p1, p2); sort2(p4, p5); sort2(p7, p8);
  sort2(p0, p3); sort2(p5, p8); sort2(p4, p7);
  sort2(p3, p6); sort2(p1, p4); sort2(p2, p5);
  sort2(p4, p7); sort2(p4, p2); sort2(p6, p4);
  sort2(p4, p2);
  return p4;
}

FL_HD float median3x3_at(const float* img, int w, int h, int x, int y) {
  const int xm = x > 0 ? x - 1 : 0, xp = x < w - 1 ? x + 1 : w - 1;
  const int ym = y > 0 ? y - 1 : 0, yp = y < h - 1 ? y + 1 : h - 1;
  return median9(img[ym * w + xm], img[ym * w + x], img[ym * w + xp],
                 img[y * w + xm], img[y * w + x], img[y * w + xp],
                 img[yp * w + xm], img[yp * w + x], img[yp * w + xp]);
}

// Pixel-centre-aligned x2 upsampling source coordinate.
FL_HD float upsample_coord(int x) { return (static_cast<float>(x) + 0.5f) * 0.5f - 0.5f; }

// Soft forward-backward consistency: exp(-|F01 + F10(x + F01)|^2 / thr).
FL_HD float fb_reliability(float u, float v, float bu, float bv, float alpha, float beta) {
  const float eu = u + bu, ev = v + bv;
  const float err = eu * eu + ev * ev;
  const float thr = alpha * (u * u + v * v + bu * bu + bv * bv) + beta;
  return expf(-err / thr);
}

// 5-tap binomial kernel [1 4 6 4 1] / 16.
FL_HD float g5(int i) {
  return i == 0 ? 0.375f : ((i == 1 || i == -1) ? 0.25f : 0.0625f);
}

}  // namespace px
}  // namespace flipster
