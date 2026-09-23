// Synthetic inputs with analytically known motion.
#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "flipster/image.hpp"

namespace flipster::testing {

// Sum of random plane waves over several octaves, so every pyramid level has
// texture. Evaluated analytically, so a shifted copy is exact (no resampling).
struct WaveTexture {
  struct Wave { float fx, fy, phase, amp; };
  std::vector<Wave> waves;
  float lo = 0.f, hi = 1.f;

  explicit WaveTexture(unsigned seed = 1) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    const float kPi = 3.14159265358979f;
    for (float period : {6.f, 11.f, 23.f, 47.f, 95.f}) {
      for (int k = 0; k < 4; ++k) {
        const float ang = 2 * kPi * uni(rng);
        const float f = 2 * kPi / (period * (0.8f + 0.4f * uni(rng)));
        waves.push_back({f * std::cos(ang), f * std::sin(ang), 2 * kPi * uni(rng), period / 95.f});
      }
    }
    float s = 0.f;
    for (const auto& w : waves) s += w.amp;
    lo = -s;
    hi = s;
  }

  float operator()(float x, float y) const {
    float v = 0.f;
    for (const auto& w : waves) v += w.amp * std::sin(w.fx * x + w.fy * y + w.phase);
    return (v - lo) / (hi - lo);
  }

  // Content moved by (+dx, +dy): I1(x) = I0(x - d).
  Plane render(int width, int height, float dx = 0.f, float dy = 0.f) const {
    Plane p(width, height);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) p.at(x, y) = (*this)(x - dx, y - dy);
    return p;
  }
};

// Anti-aliased ring (1 = ink) centred at (cx, cy).
inline Plane ring(int w, int h, float cx, float cy, float radius, float thickness = 2.f) {
  Plane p(w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const float d = std::fabs(std::hypot(x - cx, y - cy) - radius);
      const float v = 1.f - (d - 0.5f * thickness);
      p.at(x, y) = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
    }
  return p;
}

// exp(-distance / sigma) field around ink (brute force; tests only).
inline Plane soft_field(const Plane& ink, float sigma) {
  Plane out(ink.width, ink.height);
  std::vector<std::pair<int, int>> pts;
  for (int y = 0; y < ink.height; ++y)
    for (int x = 0; x < ink.width; ++x)
      if (ink.at(x, y) > 0.35f) pts.emplace_back(x, y);
  for (int y = 0; y < ink.height; ++y)
    for (int x = 0; x < ink.width; ++x) {
      float best = 1e9f;
      for (const auto& q : pts) best = std::min(best, std::hypot(float(x - q.first), float(y - q.second)));
      out.at(x, y) = std::max(std::exp(-best / sigma), ink.at(x, y));
    }
  return out;
}

inline Image as_image(const Plane& p) {
  Image img(p.width, p.height, 1);
  std::copy(p.data.begin(), p.data.end(), img.data.begin());
  return img;
}

inline float centroid_x(const float* img, int w, int h) {
  double s = 0, sx = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      s += img[y * w + x];
      sx += img[y * w + x] * x;
    }
  return static_cast<float>(sx / s);
}

}  // namespace flipster::testing
