// Backend-agnostic engine interface.
//
// The CPU (OpenMP) and CUDA engines implement the same algorithm; the Python
// NumPy reference in python/flipster/reference.py is the executable spec and the
// parity tests hold both native backends to it.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "flipster/image.hpp"
#include "flipster/params.hpp"

namespace flipster {

using Timings = std::map<std::string, double>;  // stage -> milliseconds

class Engine {
 public:
  virtual ~Engine() = default;
  virtual std::string name() const = 0;

  // Dense pyramidal Lucas-Kanade flow from `a` to `b` (single-channel, same size).
  virtual Flow flow(const Plane& a, const Plane& b, const FlowParams& p) = 0;

  // Backward warp: out(x) = img(x + flow(x)), bilinear, clamp-to-edge.
  virtual Image warp(const Image& img, const Flow& f) = 0;

  // Load a frame pair for in-between synthesis. Keeps everything resident
  // (on the GPU for CUDA) so synthesize() can be called for many t cheaply.
  // `imp0`/`imp1` may be empty (uniform importance).
  virtual void set_pair(const Image& c0, const Image& c1, const Plane& imp0, const Plane& imp1,
                        const Flow& f01, const Flow& f10, const SplatParams& sp) = 0;

  // Occlusion-aware forward-splatted frame at time t in (0, 1).
  virtual Image synthesize(float t) = 0;

  // Forward-backward reliability of the loaded pair (frame 0, frame 1).
  virtual std::pair<Plane, Plane> reliability() = 0;

  const Timings& timings() const { return timings_; }

 protected:
  Timings timings_;
};

// variant: "optimized" (default) or "naive" (CUDA only: 2-D global-memory box
// filter instead of the separable shared-memory one; used for benchmarks).
std::unique_ptr<Engine> make_cpu_engine();
std::unique_ptr<Engine> make_cuda_engine(const std::string& variant = "optimized", int device = 0);
std::unique_ptr<Engine> make_engine(const std::string& backend, const std::string& variant = "optimized");

bool cuda_compiled();
bool cuda_available();
std::string cuda_device_name();

}  // namespace flipster
