# Flipster

**Turn a hand-drawn flipbook into smooth animation.** Flipster reads scans of notebook pages (or pages you draw in the
browser), tracks every pencil stroke from one page to the next with dense **pyramidal Lucas–Kanade optical flow**
written in **C++/OpenMP and CUDA**, and draws the in-between frames with **occlusion-aware forward splatting**.
On a free Colab Tesla T4 the CUDA engine computes a 1080p flow field in **16.6 ms**, 59× faster than the multithreaded
C++ engine ([benchmarks](#performance)).

<p align="center"><img src="docs/demo.gif" alt="20 scanned pages vs. the same flipbook with 3 generated in-betweens per page" width="860"></p>

<p align="center"><img src="docs/inbetween_pair14.png" alt="Cross-fade ghosts vs. Flipster's in-betweens for pages 14 to 15" width="100%"></p>

| | |
|---|---|
| **Frontend** | React 19 · TypeScript · Vite · Tailwind CSS · TanStack Query · dnd-kit · Vitest |
| **Backend** | FastAPI · background job queue · server-sent-event progress · GIF/MP4 export |
| **Compute core** | C++17 + OpenMP, CUDA 12 kernels, pybind11 bindings, scikit-build-core, GoogleTest |
| **Tooling** | Docker (CPU and CUDA images) · GitHub Actions (Linux + Windows C++, CUDA compile, Python, web, Docker) · ruff · oxlint |

---

## How it works

```mermaid
flowchart LR
  A["Scanned pages"] --> B["Clean-up<br/>ink extraction, hole and rule removal"]
  B --> C["Stabilize<br/>align on printed paper"]
  C --> D["Distance field<br/>exp(-d/σ) around strokes"]
  D --> E["Pyramidal LK<br/>forward + backward flow<br/>CPU or CUDA"]
  E --> F["Forward-backward check<br/>occlusion reliability"]
  F --> G["Softmax forward splatting<br/>both pages to time t"]
  G --> H["Ink-mass matching"] --> I["In-between frames"]
```

1. **Clean-up (Python/OpenCV).** Ink strength is darkness of `max(R,G,B)` against a local paper-brightness estimate, so
   blue ruled lines, lighting gradients and blue stains disappear. The red margin (which scanners darken) is removed by
   subtracting the colour a thin line adds over its surroundings. Binder holes, specks and haze are dropped.
2. **Stabilize.** Pages are hand-traced, so aligning on the ink drifts. Affine ECC read the character's walk as a page
   stretch, and phase correlation on the ink gave shifts of up to 110 px. The *printed paper*, though, is identical on
   every page, so the colourful structure is projected onto each axis and the 1-D profiles are cross-correlated. On the
   bundled scans this recovers the 4–14 px margin offsets exactly.
3. **Distance-field input.** A 2 px pencil line gives a 15×15 LK window almost nothing to work with. Flow is computed
   on `exp(-distance_to_ink / σ)`, which turns each stroke into a wide, smooth ridge.
4. **Dense pyramidal Lucas–Kanade** (the part in C++ and CUDA, [`core/`](core)):
   - Gaussian pyramid (5-tap binomial). Coarse-to-fine with ×2 flow upsampling. 3× more iterations at the coarsest
     level. 3×3 median after each level.
   - Each iteration warps `I1` by the current flow and solves one 2×2 system per pixel from windowed normal equations.
     The residual is **linearized around each pixel's own flow**, `I1(y+d) ≈ I0(y) + It(y) + ∇I(y)·(d − u(y))`, which
     makes the update a structure-weighted average of neighbouring flows. The naive `box(∇I·It)` update amplified
     high-frequency noise and diverged within ~20 iterations.
   - Samples that fall outside `I1` are **masked out** of both sides of the normal equations.
   - **Levenberg–Marquardt damping** scaled by the window's gradient energy, so every pyramid level converges at the
     same rate and flat windows stay stable.
5. **Synthesis.** Both pages are **forward-splatted** to time *t* (bilinear, atomics on the GPU) with
   **softmax weights** `exp(β·ink)` (Niklaus & Liu, CVPR 2020), so a moving stroke wins over the blank paper it lands on.
   Pixels that fail the **forward–backward consistency** check (Sundaram et al., ECCV 2010) get less weight, so strokes
   that appear or vanish fade instead of smearing. v1 used backward warping instead, which looks up the flow at the
   *destination* pixel. When a thin stroke has moved farther than its own width, that pixel is blank paper with no
   motion, so the stroke never moves. A final **ink-mass match** trims bilinear spill so in-betweens are the same
   weight as the pages.

The NumPy implementation in [`python/flipster/reference.py`](python/flipster/reference.py) is the executable spec. The
C++ and CUDA engines share every per-pixel formula through
[`core/src/common/pixel_ops.hpp`](core/src/common/pixel_ops.hpp) (`__host__ __device__` inline functions), and parity
tests hold all three backends to the same answer.

### CUDA engine

- One LK iteration is three launches:
  1. a **fused** warp + residual + normal-equation kernel that writes 5 planes,
  2. a box filter over all 5 planes at once,
  3. the per-pixel damped 2×2 solve.
- **Box filter.** The optimized path is separable:
  - The horizontal pass stages a 256-px row tile plus halo in **shared memory**.
  - The vertical pass is a **coalesced running sum**: each thread walks a column strip, and adjacent threads read
    adjacent addresses.
  - A naive 2-D global-memory variant is kept so the benchmark can show the difference.
- **Memory.** Device buffers are cached and grow-only, so repeated frames never hit `cudaMalloc`. `set_pair()` keeps
  both pages, flows and reliabilities resident, so N in-betweens cost N small splat launches.
- **Timing and parity.** Stages are timed with CUDA events (upload / pyramid / LK / download). The engine is built with
  `--fmad=false` so results stay comparable to the CPU engine.

## Where v1 went wrong

v1 was my 15-112 term project (Python + `cmu_graphics`, single-scale Lucas–Kanade). Rewriting it turned up three bugs,
each now pinned by a test in [`tests/test_legacy_bugs.py`](tests/test_legacy_bugs.py) against the original code in
[`legacy.py`](python/flipster/legacy.py):

1. **Flow was 8× too small.** `cv2.Sobel(ksize=3)` returns 8× the derivative while the temporal difference was
   unscaled. On a 2 px shift v1 estimated **0.25 px**, so the "optical flow" mode was really a cross-dissolve.
2. **The warp pointed the wrong way.** `remap(frame, grid + t·flow)` pushed both frames *away* from the midpoint (a blob
   moving 38 → 40 was drawn at 37 and 41 instead of 39).
3. **Single level, single iteration.** This only works for ~1 px of motion, which is why v1 shrank pages to 1/10 size.

It was also a per-pixel Python loop calling `cond()` and `pinv()`, estimated at ~225 s per flow field at 1080p on the
Colab machine below (~105 s on a faster 2-vCPU VM).

## Performance

`python bench/run_bench.py` runs every backend on the same textured image pair (480p to 4K) with a known 12.5 × −7.25 px
shift, so the table shows speed *and* end-point error. OpenCV's Farneback and DIS are included as familiar dense-flow
reference points (they are different algorithms).

Measured on a free Google Colab **Tesla T4** (Xeon @ 2.0 GHz, 2 threads)
([bench/results/colab-tesla-t4.md](bench/results/colab-tesla-t4.md)). CUDA rows are kernel time from CUDA events:

| method | 480p | 720p | 1080p | 4K | EPE (px) |
|---|---|---|---|---|---|
| v1 (15-112) Python loop | ~32.9 s (est.) | ~110.5 s (est.) | ~225.1 s (est.) | ~899 s (est.) | — |
| NumPy reference (vectorized) | 709 ms | 1.5 s | 3.3 s | 15.1 s | 0.045 |
| C++ / OpenMP (2 threads) | 132 ms | 425 ms | 975 ms | 4.6 s | 0.045 |
| **CUDA, shared-memory box filter** | **3.1 ms** | **7.8 ms** | **16.6 ms** | **51.3 ms** | 0.045 |
| CUDA, naive box filter | 15.3 ms | 24.2 ms | 46.6 ms | 204 ms | 0.045 |
| OpenCV Farneback | 96 ms | 309 ms | 819 ms | 3.5 s | 0.042 |
| OpenCV DIS (medium) | 45 ms | 132 ms | 540 ms | 1.5 s | 0.044 |

At 1080p the CUDA engine is:

- **59× faster than the multithreaded C++ engine** (16.6 ms vs 975 ms), or 29× counting PCIe copies (33.4 ms wall).
- **~200× faster than the vectorized NumPy reference** and **~13,500× faster than v1**.
- **2.8× faster than the naive kernel** thanks to the shared-memory / running-sum box filter (4.0× at 4K, 4.9× at
  480p), with the same result.

All four Flipster backends land on the same 0.045 px error, and the CUDA-vs-CPU parity tests pass on the T4. The
Farneback and DIS rows run on the CPU, so compare them with the C++ row, not the GPU rows.

Wall time with host↔device copies is 5.7 / 17.0 / 33.4 / 134 ms (480p → 4K), so at 1080p about half of the end-to-end
time is spent outside the kernels, mostly on PCIe transfers. That's why `set_pair()` uploads each page pair once and
keeps it resident for all of its in-betweens. On a faster 2-vCPU VM without a GPU the C++ engine does 1080p in 459 ms
([bench/results/cloud-2vcpu.md](bench/results/cloud-2vcpu.md)).

To reproduce: `python bench/run_bench.py --sizes 480p,720p,1080p,4k --out bench/results/<your-gpu>.md` (or run the
Colab notebook). For kernel profiles, use `nsys profile build/flipster_bench --backend cuda` or `ncu --set full ...`.

**Accuracy:** `python eval/middlebury.py --download --out eval/results.md` compares against ground truth
([eval/results.md](eval/results.md)). vision.middlebury.edu is often unreachable from cloud machines, so by default it
falls back to the copy of Middlebury's RubberWhale sequence in OpenCV's test data, plus leave-one-out interpolation on
two real video clips from the same folder (frames 0 and 2 in, frame 1 held out). Current results:

| flow, RubberWhale | end-point error (px) |
|---|---|
| **Flipster pyramidal LK** | **0.325** |
| OpenCV Farneback | 0.361 |
| OpenCV DIS (medium) | 0.223 |

| midpoint interpolation | Corridor-VGA | Street-720p | mean PSNR | mean SSIM |
|---|---|---|---|---|
| Cross-dissolve | 28.74 dB | 20.36 dB | 24.55 dB | 0.72 |
| Backward warp (v1's approach, bugs fixed) | 34.68 dB | 21.96 dB | 28.32 dB | 0.81 |
| **Flipster forward splatting** | 34.63 dB | 22.03 dB | **28.33 dB** | 0.81 |

The flow is more accurate than Farneback and less accurate than DIS. Both flow-based interpolators beat cross-dissolve
by ~3.8 dB. On textured photos, splatting and backward warping come out even: splatting pays off on thin line art,
where backward warping can't move a stroke farther than its own width. Natural video isn't what Flipster is tuned for,
so treat this as a sanity check of the core algorithm.

## Running it

### Google Colab (free NVIDIA GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lolaitan/Flipster/blob/main/notebooks/flipster_colab.ipynb)

[`notebooks/flipster_colab.ipynb`](notebooks/flipster_colab.ipynb) builds the CUDA engine on Colab's GPU, runs the C++
and Python suites (including CPU↔CUDA parity), benchmarks every backend up to 4K, runs the Middlebury evaluation,
renders the sample flipbook and can serve the web app through Colab's proxy. Pick *Runtime → Change runtime type → T4
GPU*, then *Run all*. The last cell downloads the results.

### Windows (one click)

Double-click **`scripts\windows\setup.cmd`**. It finds Python, creates `.venv`, builds the C++ engine with Visual
Studio's compiler (and the CUDA engine when an NVIDIA GPU and the CUDA Toolkit are present), builds the web app, runs
the tests, then opens http://127.0.0.1:8000. Without a C++ compiler it falls back to the NumPy engine, so the app
still runs. After that, `scripts\windows\run.cmd` just starts the app. The full log is in `data\setup\setup.log`.

### Docker

```bash
docker compose up                   # CPU build → http://localhost:8000
docker compose --profile gpu up     # CUDA build → http://localhost:8001 (needs the NVIDIA Container Toolkit)
```

### Local development

Requirements:

- Python ≥ 3.10
- CMake ≥ 3.20 and a C++17 compiler (GCC/Clang, or Visual Studio 2022 Build Tools on Windows)
- Node 22
- Optional: CUDA Toolkit 12.x. When `nvcc` is found, the CUDA backend is built automatically.

```bash
pip install -e ".[server,dev]"      # compiles flipster._core (CPU, + CUDA if nvcc is found)
uvicorn app.main:app --app-dir server --reload        # API on :8000

cd web && npm install && npm run dev                    # UI on :5173, proxies /api to :8000
```

Force a backend with `FLIPSTER_ENABLE_CUDA=ON|OFF pip install .`. Target only your own GPU with
`--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=native`.

**Windows + NVIDIA:** the simplest route is WSL2 (Ubuntu) with the CUDA toolkit for WSL, then the Linux steps above.
Native Windows works too: install Visual Studio 2022 Build Tools and the CUDA Toolkit, then run
`pip install -e .[server,dev]` from a *x64 Native Tools* prompt.

### Tests

```bash
pytest                                             # NumPy reference, native engines, parity, API
cmake -S . -B build -DFLIPSTER_BUILD_TESTS=ON && cmake --build build && ctest --test-dir build
cd web && npm test
```

On a CUDA machine, `ctest` also runs the CPU↔CUDA parity suite (optimized and naive kernels), and `pytest` includes
the `cuda` backend in every flow/synthesis test.

## Project layout

```
core/            C++17 / CUDA engine
  include/         public headers (Engine, FlowParams, images)
  src/common/      __host__ __device__ per-pixel math shared by both backends
  src/cpu/         OpenMP engine
  src/cuda/        CUDA engine + kernels
  tests/, tools/   GoogleTest suite, native benchmark
bindings/        pybind11 module (flipster._core)
python/flipster/ reference implementation, preprocessing, pipeline, export
server/          FastAPI app (projects, jobs, SSE, export) + API tests
web/             React + TypeScript app
bench/, eval/    throughput benchmark, Middlebury accuracy harness
samples/scans/   the 20 original notebook pages
docker/, .github/workflows/
```

## API

| method | path | |
|---|---|---|
| `GET` | `/api/system` | backends, GPU name, limits |
| `POST` | `/api/projects` | new project (`scan` or `drawing`) |
| `POST` | `/api/projects/{id}/frames` | upload pages (multipart) |
| `POST` | `/api/projects/{id}/samples` | load the bundled flipbook |
| `PUT` | `/api/projects/{id}/frames/order` | reorder |
| `POST` | `/api/projects/{id}/renders` | start a render → `job_id`, `render_id` |
| `GET` | `/api/jobs/{id}/events` | server-sent progress events |
| `GET` | `/api/renders/{id}` | frames, flow visualizations, timing summary |
| `GET` | `/api/renders/{id}/export?format=gif\|mp4` | download |

## Limitations

- Lucas–Kanade assumes locally smooth motion. Strokes that cross, rotate fast or change shape a lot between pages
  (limbs swinging) can tear. The consistency check turns those into a fade instead of a smear.
- Motion along a perfectly straight stroke is unobservable (the aperture problem). Such strokes take the motion of
  their surroundings.
- The scan clean-up is tuned for pencil or dark pen on ruled notebook paper.

## References

- B. D. Lucas and T. Kanade, *An Iterative Image Registration Technique with an Application to Stereo Vision*, 1981.
- J.-Y. Bouguet, *Pyramidal Implementation of the Lucas Kanade Feature Tracker*, Intel, 2000.
- S. Baker and I. Matthews, *Lucas-Kanade 20 Years On: A Unifying Framework*, IJCV 2004.
- N. Sundaram, T. Brox and K. Keutzer, *Dense Point Trajectories by GPU-Accelerated Large Displacement Optical Flow*,
  ECCV 2010.
- S. Niklaus and F. Liu, *Softmax Splatting for Video Frame Interpolation*, CVPR 2020.
- S. Baker et al., *A Database and Evaluation Methodology for Optical Flow*, IJCV 2011 (Middlebury).

MIT licensed.
