#include <stdexcept>

#include "flipster/engine.hpp"

namespace flipster {

std::unique_ptr<Engine> make_engine(const std::string& backend, const std::string& variant) {
  if (backend == "cpu") return make_cpu_engine();
  if (backend == "cuda") return make_cuda_engine(variant);
  throw std::invalid_argument("unknown backend '" + backend + "' (expected 'cpu' or 'cuda')");
}

#ifndef FLIPSTER_WITH_CUDA
bool cuda_compiled() { return false; }
bool cuda_available() { return false; }
std::string cuda_device_name() { return ""; }
std::unique_ptr<Engine> make_cuda_engine(const std::string&, int) {
  throw std::runtime_error("flipster was built without CUDA (configure with -DFLIPSTER_ENABLE_CUDA=ON)");
}
#endif

}  // namespace flipster
