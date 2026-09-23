// Minimal planar float image types shared by the CPU and CUDA engines.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace flipster {

// A single-channel float image, row-major.
struct Plane {
  int width = 0;
  int height = 0;
  std::vector<float> data;

  Plane() = default;
  Plane(int w, int h, float fill = 0.f) : width(w), height(h), data(static_cast<size_t>(w) * h, fill) {}

  size_t size() const { return data.size(); }
  bool empty() const { return data.empty(); }
  float& at(int x, int y) { return data[static_cast<size_t>(y) * width + x]; }
  float at(int x, int y) const { return data[static_cast<size_t>(y) * width + x]; }
  float* ptr() { return data.data(); }
  const float* ptr() const { return data.data(); }
};

// A C-channel float image stored planar (channel-major): data[c][y][x].
struct Image {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<float> data;

  Image() = default;
  Image(int w, int h, int c, float fill = 0.f)
      : width(w), height(h), channels(c), data(static_cast<size_t>(w) * h * c, fill) {}

  size_t plane_size() const { return static_cast<size_t>(width) * height; }
  float* plane(int c) { return data.data() + c * plane_size(); }
  const float* plane(int c) const { return data.data() + c * plane_size(); }
};

// Dense optical flow: per-pixel displacement (u, v) in pixels.
struct Flow {
  Plane u, v;
  Flow() = default;
  Flow(int w, int h) : u(w, h), v(w, h) {}
  int width() const { return u.width; }
  int height() const { return u.height; }
};

inline void require_same_size(int w0, int h0, int w1, int h1, const char* what) {
  if (w0 != w1 || h0 != h1) throw std::invalid_argument(std::string(what) + ": size mismatch");
}

}  // namespace flipster
