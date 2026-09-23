#include <gtest/gtest.h>

#include <algorithm>
#include <random>

#include "common/pixel_ops.hpp"

using namespace flipster;

TEST(PixelOps, Median9MatchesSort) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> uni(-5.f, 5.f);
  for (int trial = 0; trial < 20000; ++trial) {
    float v[9];
    for (float& x : v) x = (trial % 3 == 0) ? std::round(uni(rng)) : uni(rng);  // include ties
    float s[9];
    std::copy(v, v + 9, s);
    std::nth_element(s, s + 4, s + 9);
    ASSERT_EQ(px::median9(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]), s[4]);
  }
}

TEST(PixelOps, BilinearSampleClampsAndInterpolates) {
  const float img[] = {0.f, 1.f, 2.f, 3.f,   //
                       4.f, 5.f, 6.f, 7.f};  // 4 x 2
  EXPECT_FLOAT_EQ(px::sample_bilinear(img, 4, 2, 1.5f, 0.5f), 3.5f);
  EXPECT_FLOAT_EQ(px::sample_bilinear(img, 4, 2, -3.f, 0.f), 0.f);
  EXPECT_FLOAT_EQ(px::sample_bilinear(img, 4, 2, 10.f, 10.f), 7.f);
  EXPECT_FLOAT_EQ(px::sample_bilinear(img, 4, 2, 3.f, 1.f), 7.f);
}

TEST(PixelOps, GradientIsCorrectlyScaled) {
  // A ramp with slope 2 must give Ix = 2, not 16 (the v1 Sobel bug).
  float img[5 * 3];
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 5; ++x) img[y * 5 + x] = 2.f * x;
  float gx, gy;
  px::gradient(img, 5, 3, 2, 1, &gx, &gy);
  EXPECT_FLOAT_EQ(gx, 2.f);
  EXPECT_FLOAT_EQ(gy, 0.f);
}

TEST(PixelOps, LkUpdateSolvesTranslation) {
  // Pure x-gradient window: Gxx = 1, Gxy = Gyy = 0; residual says move by +1.5.
  float u = 0.f, v = 0.f;
  for (int i = 0; i < 50; ++i) {
    const float res_bx = -(1.5f - u) * 1.f - u * 1.f;  // sum g (It - g.u) with It = -(1.5 - u)
    px::lk_update(1.f, 0.f, 0.f, res_bx, 0.f, 0.05f, 1e-6f, 0.f, 4.f, &u, &v);
  }
  EXPECT_NEAR(u, 1.5f, 1e-4f);
  EXPECT_NEAR(v, 0.f, 1e-6f);  // unobservable direction stays put
}

TEST(PixelOps, ForwardBackwardReliability) {
  EXPECT_FLOAT_EQ(px::fb_reliability(3.f, -2.f, -3.f, 2.f, 0.05f, 1.f), 1.f);
  EXPECT_LT(px::fb_reliability(3.f, -2.f, 3.f, 2.f, 0.05f, 1.f), 0.01f);
}

TEST(PixelOps, GaussianKernelSumsToOne) {
  float s = 0.f;
  for (int k = -2; k <= 2; ++k) s += px::g5(k);
  EXPECT_FLOAT_EQ(s, 1.f);
}
