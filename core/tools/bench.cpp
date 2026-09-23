// Native micro-benchmark (no Python in the loop). Handy under Nsight:
//   nsys profile ./flipster_bench --backend cuda --size 1920x1080
//   ncu --set full ./flipster_bench --backend cuda --size 1920x1080 --reps 1
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "flipster/engine.hpp"
#include "test_util.hpp"

using namespace flipster;

int main(int argc, char** argv) {
  std::string backend = "cpu", variant = "optimized";
  int w = 1920, h = 1080, reps = 10;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--backend") backend = next();
    else if (a == "--variant") variant = next();
    else if (a == "--reps") reps = std::stoi(next());
    else if (a == "--size") {
      const std::string s = next();
      std::sscanf(s.c_str(), "%dx%d", &w, &h);
    } else {
      std::printf("usage: flipster_bench [--backend cpu|cuda] [--variant optimized|naive] [--size WxH] [--reps N]\n");
      return 1;
    }
  }
  testing::WaveTexture tex(11);
  const Plane a = tex.render(w, h), b = tex.render(w, h, 12.5f, -7.25f);
  auto engine = make_engine(backend, variant);
  engine->flow(a, b, FlowParams());  // warm-up (allocations, CUDA context)

  std::vector<double> wall, compute;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    engine->flow(a, b, FlowParams());
    wall.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    const auto& t = engine->timings();
    compute.push_back(t.count("kernel_ms") ? t.at("kernel_ms") : t.at("flow_ms"));
  }
  std::sort(wall.begin(), wall.end());
  std::sort(compute.begin(), compute.end());
  std::printf("%s/%s %dx%d  median wall %.2f ms  median compute %.2f ms  (%d reps)\n", backend.c_str(),
              variant.c_str(), w, h, wall[wall.size() / 2], compute[compute.size() / 2], reps);
  return 0;
}
