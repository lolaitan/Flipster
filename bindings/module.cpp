// pybind11 bindings: numpy (H, W[, C]) float32 arrays <-> planar flipster images.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>

#include "flipster/engine.hpp"

namespace py = pybind11;
using namespace flipster;

using FArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

Plane to_plane(const FArray& a, const char* what) {
  if (a.ndim() != 2) throw std::invalid_argument(std::string(what) + ": expected a 2-D (H, W) array");
  Plane p(static_cast<int>(a.shape(1)), static_cast<int>(a.shape(0)));
  std::memcpy(p.ptr(), a.data(), p.size() * sizeof(float));
  return p;
}

Plane to_optional_plane(const py::object& o, const char* what) {
  if (o.is_none()) return Plane();
  return to_plane(o.cast<FArray>(), what);
}

Image to_image(const FArray& a, const char* what) {
  if (a.ndim() != 3) throw std::invalid_argument(std::string(what) + ": expected a 3-D (H, W, C) array");
  const int h = static_cast<int>(a.shape(0)), w = static_cast<int>(a.shape(1)), c = static_cast<int>(a.shape(2));
  Image img(w, h, c);
  const float* src = a.data();
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int ch = 0; ch < c; ++ch)
        img.plane(ch)[static_cast<size_t>(y) * w + x] = src[(static_cast<size_t>(y) * w + x) * c + ch];
  return img;
}

Flow to_flow(const FArray& a, const char* what) {
  if (a.ndim() != 3 || a.shape(2) != 2) throw std::invalid_argument(std::string(what) + ": expected (H, W, 2) flow");
  const int h = static_cast<int>(a.shape(0)), w = static_cast<int>(a.shape(1));
  Flow f(w, h);
  const float* src = a.data();
  for (size_t k = 0; k < static_cast<size_t>(w) * h; ++k) {
    f.u.data[k] = src[2 * k];
    f.v.data[k] = src[2 * k + 1];
  }
  return f;
}

FArray from_plane(const Plane& p) {
  FArray out({p.height, p.width});
  std::memcpy(out.mutable_data(), p.ptr(), p.size() * sizeof(float));
  return out;
}

FArray from_image(const Image& img) {
  FArray out({img.height, img.width, img.channels});
  float* dst = out.mutable_data();
  const size_t n = img.plane_size();
  for (size_t k = 0; k < n; ++k)
    for (int c = 0; c < img.channels; ++c) dst[k * img.channels + c] = img.plane(c)[k];
  return out;
}

FArray from_flow(const Flow& f) {
  FArray out({f.height(), f.width(), 2});
  float* dst = out.mutable_data();
  for (size_t k = 0; k < f.u.size(); ++k) {
    dst[2 * k] = f.u.data[k];
    dst[2 * k + 1] = f.v.data[k];
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Flipster native engines (C++/OpenMP and CUDA)";

  py::class_<FlowParams>(m, "FlowParams")
      .def(py::init<>())
      .def_readwrite("levels", &FlowParams::levels)
      .def_readwrite("max_levels", &FlowParams::max_levels)
      .def_readwrite("min_size", &FlowParams::min_size)
      .def_readwrite("iterations", &FlowParams::iterations)
      .def_readwrite("window_radius", &FlowParams::window_radius)
      .def_readwrite("damping", &FlowParams::damping)
      .def_readwrite("damping_floor", &FlowParams::damping_floor)
      .def_readwrite("zero_pull", &FlowParams::zero_pull)
      .def_readwrite("max_step", &FlowParams::max_step)
      .def_readwrite("median", &FlowParams::median)
      .def("resolve_levels", &FlowParams::resolve_levels);

  py::class_<SplatParams>(m, "SplatParams")
      .def(py::init<>())
      .def_readwrite("softmax_beta", &SplatParams::softmax_beta)
      .def_readwrite("fb_alpha", &SplatParams::fb_alpha)
      .def_readwrite("fb_beta", &SplatParams::fb_beta)
      .def_readwrite("min_reliability", &SplatParams::min_reliability);

  py::class_<Engine>(m, "Engine")
      .def_property_readonly("name", &Engine::name)
      .def(
          "flow",
          [](Engine& e, const FArray& a, const FArray& b, const FlowParams& p) {
            Plane pa = to_plane(a, "a"), pb = to_plane(b, "b");
            Flow f;
            {
              py::gil_scoped_release release;
              f = e.flow(pa, pb, p);
            }
            return from_flow(f);
          },
          py::arg("a"), py::arg("b"), py::arg("params") = FlowParams())
      .def("warp",
           [](Engine& e, const FArray& img, const FArray& flow) {
             Image im = to_image(img, "img");
             Flow f = to_flow(flow, "flow");
             Image out;
             {
               py::gil_scoped_release release;
               out = e.warp(im, f);
             }
             return from_image(out);
           })
      .def(
          "set_pair",
          [](Engine& e, const FArray& c0, const FArray& c1, const py::object& imp0, const py::object& imp1,
             const FArray& f01, const FArray& f10, const SplatParams& sp) {
            Image i0 = to_image(c0, "c0"), i1 = to_image(c1, "c1");
            Plane p0 = to_optional_plane(imp0, "imp0"), p1 = to_optional_plane(imp1, "imp1");
            Flow a = to_flow(f01, "f01"), b = to_flow(f10, "f10");
            py::gil_scoped_release release;
            e.set_pair(i0, i1, p0, p1, a, b, sp);
          },
          py::arg("c0"), py::arg("c1"), py::arg("imp0"), py::arg("imp1"), py::arg("f01"), py::arg("f10"),
          py::arg("params") = SplatParams())
      .def("synthesize",
           [](Engine& e, float t) {
             Image out;
             {
               py::gil_scoped_release release;
               out = e.synthesize(t);
             }
             return from_image(out);
           })
      .def("reliability",
           [](Engine& e) {
             auto r = e.reliability();
             return py::make_tuple(from_plane(r.first), from_plane(r.second));
           })
      .def("timings", &Engine::timings);

  m.def("make_engine", &make_engine, py::arg("backend"), py::arg("variant") = "optimized");
  m.def("cuda_compiled", &cuda_compiled);
  m.def("cuda_available", &cuda_available);
  m.def("cuda_device_name", &cuda_device_name);
}
