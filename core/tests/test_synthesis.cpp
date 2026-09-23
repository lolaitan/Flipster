#include <gtest/gtest.h>

#include "flipster/engine.hpp"
#include "test_util.hpp"

using namespace flipster;
namespace ft = flipster::testing;

namespace {

struct Pair {
  Plane a, b, fa, fb;
};

Pair moving_ring(float shift) {
  Pair p;
  p.a = ft::ring(160, 100, 60.f, 50.f, 14.f);
  p.b = ft::ring(160, 100, 60.f + shift, 50.f, 14.f);
  p.fa = ft::soft_field(p.a, 4.f);
  p.fb = ft::soft_field(p.b, 4.f);
  return p;
}

}  // namespace

TEST(Synthesis, MidpointIsSharpAndCentred) {
  const Pair p = moving_ring(20.f);
  auto e = make_cpu_engine();
  const Flow f01 = e->flow(p.fa, p.fb, FlowParams());
  const Flow f10 = e->flow(p.fb, p.fa, FlowParams());
  e->set_pair(ft::as_image(p.a), ft::as_image(p.b), p.a, p.b, f01, f10, SplatParams());
  const Image mid = e->synthesize(0.5f);
  EXPECT_NEAR(ft::centroid_x(mid.plane(0), mid.width, mid.height), 70.f, 0.75f);
  float peak = 0.f;
  for (float v : mid.data) peak = std::max(peak, v);
  EXPECT_GT(peak, 0.9f);  // a cross-dissolve would be ~0.5
}

TEST(Synthesis, EndpointsReproduceKeyframes) {
  const Pair p = moving_ring(12.f);
  auto e = make_cpu_engine();
  e->set_pair(ft::as_image(p.a), ft::as_image(p.b), p.a, p.b, e->flow(p.fa, p.fb, FlowParams()),
              e->flow(p.fb, p.fa, FlowParams()), SplatParams());
  const Image s0 = e->synthesize(1e-4f);
  double err = 0;
  for (size_t k = 0; k < p.a.size(); ++k) err += std::fabs(s0.data[k] - p.a.data[k]);
  EXPECT_LT(err / p.a.size(), 0.01);
}

TEST(Synthesis, WarpMovesContent) {
  Image img(30, 20, 1);
  img.plane(0)[10 * 30 + 15] = 1.f;
  Flow f(30, 20);
  for (float& u : f.u.data) u = 2.f;
  const Image out = make_cpu_engine()->warp(img, f);
  EXPECT_FLOAT_EQ(out.plane(0)[10 * 30 + 13], 1.f);
}

TEST(Synthesis, SynthesizeBeforeSetPairThrows) {
  EXPECT_THROW(make_cpu_engine()->synthesize(0.5f), std::runtime_error);
}
