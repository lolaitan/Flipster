// Algorithm parameters. Field names and defaults mirror python/flipster/params.py.
#pragma once

#include <algorithm>

namespace flipster {

struct FlowParams {
  int levels = 0;            // 0 = automatic
  int max_levels = 6;
  int min_size = 16;
  int iterations = 5;        // per level (the coarsest level runs 3x as many)
  int window_radius = 7;     // (2r+1)^2 window
  float damping = 0.05f;     // LM damping relative to (Gxx + Gyy) / 2
  float damping_floor = 1e-6f;
  float zero_pull = 0.f;     // weak prior toward zero motion
  float max_step = 4.f;      // per-iteration update clamp (px)
  bool median = true;        // 3x3 median on the flow after each level

  int resolve_levels(int height, int width) const {
    if (levels > 0) return levels;
    int n = 1, h = height, w = width;
    while (n < max_levels && std::min((h + 1) / 2, (w + 1) / 2) >= min_size) {
      h = (h + 1) / 2;
      w = (w + 1) / 2;
      ++n;
    }
    return n;
  }
};

constexpr int kCoarseIterMult = 3;

struct SplatParams {
  float softmax_beta = 10.f;    // importance weight = exp(beta * importance)
  float fb_alpha = 0.05f;       // forward-backward consistency (Sundaram et al. 2010)
  float fb_beta = 1.f;
  float min_reliability = 0.05f;
};

constexpr float kCoverageEps = 0.05f;

}  // namespace flipster
