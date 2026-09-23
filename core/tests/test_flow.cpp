#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "flipster/engine.hpp"
#include "test_util.hpp"

using namespace flipster;
using flipster::testing::WaveTexture;

namespace {

struct Epe {
  float median, mean;
};

Epe interior_epe(const Flow& f, float dx, float dy, int margin) {
  std::vector<float> e;
  double sum = 0;
  for (int y = margin; y < f.height() - margin; ++y)
    for (int x = margin; x < f.width() - margin; ++x) {
      const float d = std::hypot(f.u.at(x, y) - dx, f.v.at(x, y) - dy);
      e.push_back(d);
      sum += d;
    }
  std::nth_element(e.begin(), e.begin() + e.size() / 2, e.end());
  return {e[e.size() / 2], static_cast<float>(sum / e.size())};
}

class FlowTest : public ::testing::TestWithParam<std::tuple<float, float>> {};

}  // namespace

TEST_P(FlowTest, RecoversTranslation) {
  const auto [dx, dy] = GetParam();
  WaveTexture tex(3);
  const Plane a = tex.render(320, 240), b = tex.render(320, 240, dx, dy);
  auto engine = make_cpu_engine();
  const Flow f = engine->flow(a, b, FlowParams());
  const Epe e = interior_epe(f, dx, dy, 40);
  EXPECT_LT(e.median, 0.05f) << "dx=" << dx << " dy=" << dy;
  EXPECT_LT(e.mean, 0.25f);
  EXPECT_GT(engine->timings().at("flow_ms"), 0.0);
}

INSTANTIATE_TEST_SUITE_P(Motions, FlowTest,
                         ::testing::Values(std::make_tuple(0.4f, -0.25f), std::make_tuple(6.5f, 3.f),
                                           std::make_tuple(28.f, -17.f)));

TEST(Flow, PyramidIsWhatHandlesLargeMotion) {
  WaveTexture tex(3);
  const Plane a = tex.render(320, 240), b = tex.render(320, 240, 28.f, -17.f);
  FlowParams single;
  single.levels = 1;
  const Flow f = make_cpu_engine()->flow(a, b, single);
  EXPECT_GT(interior_epe(f, 28.f, -17.f, 40).median, 5.f);
}

TEST(Flow, FlatImageIsStable) {
  const Plane flat(64, 48, 0.5f);
  const Flow f = make_cpu_engine()->flow(flat, flat, FlowParams());
  for (size_t k = 0; k < f.u.size(); ++k) {
    ASSERT_TRUE(std::isfinite(f.u.data[k]) && std::isfinite(f.v.data[k]));
    ASSERT_EQ(f.u.data[k], 0.f);
  }
}

TEST(Flow, RejectsMismatchedSizes) {
  EXPECT_THROW(make_cpu_engine()->flow(Plane(10, 10), Plane(11, 10), FlowParams()), std::invalid_argument);
}

TEST(Flow, AutoLevels) {
  FlowParams p;
  EXPECT_EQ(p.resolve_levels(1200, 900), 6);
  EXPECT_EQ(p.resolve_levels(40, 40), 2);
}
