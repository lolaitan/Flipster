// CPU <-> CUDA parity. Skipped automatically on machines without a GPU.
#include <gtest/gtest.h>

#include <cmath>

#include "flipster/engine.hpp"
#include "test_util.hpp"

using namespace flipster;
namespace ft = flipster::testing;

namespace {

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.f;
  for (size_t k = 0; k < a.size(); ++k) m = std::max(m, std::fabs(a[k] - b[k]));
  return m;
}

float mean_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double s = 0;
  for (size_t k = 0; k < a.size(); ++k) s += std::fabs(a[k] - b[k]);
  return static_cast<float>(s / a.size());
}

class CudaParity : public ::testing::TestWithParam<std::string> {
 protected:
  void SetUp() override {
    if (!cuda_available()) GTEST_SKIP() << "no CUDA device";
  }
};

}  // namespace

TEST_P(CudaParity, FlowMatchesCpu) {
  ft::WaveTexture tex(5);
  const Plane a = tex.render(333, 211), b = tex.render(333, 211, 9.3f, -4.1f);
  const Flow fc = make_cpu_engine()->flow(a, b, FlowParams());
  auto g = make_cuda_engine(GetParam());
  const Flow fg = g->flow(a, b, FlowParams());
  EXPECT_LT(mean_abs_diff(fc.u.data, fg.u.data), 1e-3f);
  EXPECT_LT(mean_abs_diff(fc.v.data, fg.v.data), 1e-3f);
  EXPECT_GT(g->timings().at("flow_ms"), 0.0);
}

TEST_P(CudaParity, SynthesisMatchesCpu) {
  const Plane a = ft::ring(150, 90, 50.f, 45.f, 12.f), b = ft::ring(150, 90, 72.f, 45.f, 12.f);
  const Plane fa = ft::soft_field(a, 4.f), fb = ft::soft_field(b, 4.f);
  auto c = make_cpu_engine();
  auto g = make_cuda_engine(GetParam());
  const Flow f01 = c->flow(fa, fb, FlowParams()), f10 = c->flow(fb, fa, FlowParams());
  c->set_pair(ft::as_image(a), ft::as_image(b), a, b, f01, f10, SplatParams());
  g->set_pair(ft::as_image(a), ft::as_image(b), a, b, f01, f10, SplatParams());
  for (float t : {0.25f, 0.5f, 0.8f}) {
    const Image ic = c->synthesize(t), ig = g->synthesize(t);
    EXPECT_LT(max_abs_diff(ic.data, ig.data), 2e-3f) << "t=" << t;
  }
  EXPECT_LT(max_abs_diff(c->reliability().first.data, g->reliability().first.data), 1e-4f);
}

TEST_P(CudaParity, WarpMatchesCpu) {
  ft::WaveTexture tex(2);
  const Plane a = tex.render(97, 61);
  Flow f(97, 61);
  for (size_t k = 0; k < f.u.size(); ++k) {
    f.u.data[k] = 3.3f * std::sin(0.1f * k);
    f.v.data[k] = -2.1f;
  }
  const Image wc = make_cpu_engine()->warp(ft::as_image(a), f);
  const Image wg = make_cuda_engine(GetParam())->warp(ft::as_image(a), f);
  EXPECT_LT(max_abs_diff(wc.data, wg.data), 1e-5f);
}

INSTANTIATE_TEST_SUITE_P(Variants, CudaParity, ::testing::Values("optimized", "naive"));

TEST(Cuda, AvailabilityIsConsistent) {
  if (!cuda_compiled()) {
    EXPECT_FALSE(cuda_available());
    EXPECT_THROW(make_cuda_engine(), std::runtime_error);
  } else if (cuda_available()) {
    EXPECT_FALSE(cuda_device_name().empty());
  }
}
