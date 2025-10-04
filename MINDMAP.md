```mermaid
mindmap
  root((Mandelbrot Otterdream))
    Renderer Orchestration & Windowing
      Renderer Window / Loop / Pipeline
      State (GL/CUDA) & Resources
    Frame Pipeline & Timing
      Per-frame execution flow
      FPS & Frame limiting
    CUDA Interop & Device Buffers
      PBO map/unmap, streams
      Device buffers (Hermelin)
    Capybara Compute Pipeline (Mandelbrot)
      Z-iteration, mapping, math
      Kernels & CUHs
    Colorization & Post-processing
      Iteration buffer → RGBA
    Overlays & HUD
      Heatmap (entropy/contrast)
      Text HUD (Warzenschwein)
    Project Settings, Common & PCH
      Global flags & constants
      Helpers / PCH
    GL Shaders
      Palette sampling, etc.
    Logging & Diagnostics (Luchs)
      Host & CUDA device log buffers
```

# Architecture Mindmap & Responsibilities

> Generated from your `/src.zip` snapshot.

## Module Buckets & Files
### Renderer Orchestration & Windowing
- High-level ownership of the render loop, window creation, input handling, and swapchain-ish flow.
- Wires together the `FramePipeline::execute` with GL resources and CUDA interop.
- Splits state into GL-facing and CUDA-facing structures where needed.

- `src/main.cpp`
- `src/renderer_core.cu`
- `src/renderer_core.hpp`
- `src/renderer_loop.cpp`
- `src/renderer_loop.hpp`
- `src/renderer_pipeline.cpp`
- `src/renderer_pipeline.hpp`
- `src/renderer_resources.cpp`
- `src/renderer_resources.hpp`
- `src/renderer_state.hpp`
- `src/renderer_state_cuda.cpp`
- `src/renderer_state_gl.cpp`
- `src/renderer_window.cpp`
- `src/renderer_window.hpp`

### Frame Pipeline & Timing
- *Single place* for per-frame control flow and timings — maps PBO → texture, draws base image, draws overlays, logs one fixed ASCII PERF line.
- Decides tile sizes, triggers zoom logic, maintains `FrameContext` as the authoritative per-frame data.
- Owns ring-usage stats and periodic logs.

- `src/capybara_frame_pipeline.cuh`
- `src/fps_meter.cpp`
- `src/fps_meter.hpp`
- `src/frame_context.cpp`
- `src/frame_context.hpp`
- `src/frame_limiter.hpp`
- `src/frame_pipeline.cpp`
- `src/frame_pipeline.hpp`

### CUDA Interop & Device Buffers
- Maps/unmaps the current PBO for CUDA access; launches Capybara kernels and colorization; unmaps; leaves GL upload to the pipeline.
- Holds the global pause toggle for zoom, and (optionally) mirrors iteration buffers to host for heatmap metrics.
- Encapsulates GL→CUDA resource interop via `bear_CudaPBOResource` and device memory via `Hermelin::CudaDeviceBuffer`.

- `src/bear_CudaPBOResource.cpp`
- `src/bear_CudaPBOResource.hpp`
- `src/cuda_interop.cu`
- `src/cuda_interop.hpp`
- `src/hermelin_buffer.cpp`
- `src/hermelin_buffer.hpp`

### Capybara Compute Pipeline (Mandelbrot)
- Kernel-side pipeline: complex math, coordinate mapping, and z-iteration (escape-time) specialized for performance and numeric stability.
- All heavy CUDA math lives here in `.cuh` headers and `.cu` kernels; host code should only depend on a narrow interface.

- `src/capybara_api.cuh`
- `src/capybara_integration.cuh`
- `src/capybara_mapping.cuh`
- `src/capybara_math.cuh`
- `src/capybara_pixel_iter.cuh`
- `src/capybara_render_kernel.cu`
- `src/capybara_ziter.cuh`

### Colorization & Post-processing
- Converts 16-bit iteration counts to RGBA into the mapped PBO, optionally applying a palette.

- `src/colorize_iterations.cu`
- `src/colorize_iterations.cuh`

### Overlays & HUD
- Heatmap overlay renders entropy/contrast tiles (host-side arrays) over the final texture; optional grid visualization.
- Warzenschwein overlay renders the HUD text (built in `hud_text`); draws last to stay on top; uses blending.

- `src/heatmap_overlay.cpp`
- `src/heatmap_overlay.hpp`
- `src/hud_text.cpp`
- `src/hud_text.hpp`
- `src/warzenschwein_overlay.cpp`
- `src/warzenschwein_overlay.hpp`

### Project Settings, Common & PCH
- Central compile-time options (performance logging, grid size policy, ring size); small helpers; precompiled header wiring.

- `src/common.hpp`
- `src/pch.hpp`
- `src/settings.hpp`

### Logging & Diagnostics (Luchs)
- Lightweight host logger and CUDA device log buffer utilities; use numeric RC codes and flush device log on error paths.

- `src/luchs_cuda_log_buffer.cu`
- `src/luchs_cuda_log_buffer.hpp`
- `src/luchs_log_device.hpp`
- `src/luchs_log_host.cpp`
- `src/luchs_log_host.hpp`

### Other
Miscellaneous utilities.
- `src/frame_capture.cpp`
- `src/frame_capture.hpp`
- `src/heatmap_utils.hpp`
- `src/warzenschwein_fontdata.hpp`
- `src/zoom_logic.cpp`
- `src/zoom_logic.hpp`

## Dependency Hot Spots (by `#include` fan-in/out)
Top *fan-in* (widely used headers):
- `src/luchs_log_host.hpp` used by **24** files
- `src/settings.hpp` used by **21** files
- `src/pch.hpp` used by **16** files
- `src/renderer_state.hpp` used by **12** files
- `src/cuda_interop.hpp` used by **8** files
- `src/capybara_math.cuh` used by **5** files
- `src/frame_context.hpp` used by **5** files
- `src/common.hpp` used by **4** files
- `src/luchs_cuda_log_buffer.hpp` used by **4** files
- `src/zoom_logic.hpp` used by **4** files
- `src/capybara_mapping.cuh` used by **3** files
- `src/capybara_ziter.cuh` used by **3** files
- `src/fps_meter.hpp` used by **3** files
- `src/heatmap_overlay.hpp` used by **3** files
- `src/hermelin_buffer.hpp` used by **3** files

Top *fan-out* (files that include many others):
- `src/frame_pipeline.cpp` includes **14** files
- `src/cuda_interop.cu` includes **11** files
- `src/renderer_loop.cpp` includes **11** files
- `src/capybara_render_kernel.cu` includes **9** files
- `src/renderer_core.cu` includes **9** files
- `src/renderer_state_gl.cpp` includes **8** files
- `src/main.cpp` includes **7** files
- `src/zoom_logic.cpp` includes **7** files
- `src/renderer_state_cuda.cpp` includes **6** files
- `src/renderer_window.cpp` includes **6** files
- `src/heatmap_overlay.cpp` includes **5** files
- `src/renderer_pipeline.cpp` includes **5** files
- `src/warzenschwein_overlay.cpp` includes **5** files
- `src/bear_CudaPBOResource.cpp` includes **4** files
- `src/capybara_pixel_iter.cuh` includes **4** files

## Ownership & Boundaries (no wild growth)
1. **FramePipeline** is the *only* place that sequences a frame: compute → PBO→texture upload → base draw → overlays → logging.
   - No CUDA calls from overlays; no GL in `cuda_interop.hpp` (header stays CUDA/host-only).
2. **CUDA interop** owns PBO map/unmap and kernel/colorize launches. It may expose *read-only* helpers to mirror device buffers to host for metrics.
   - No GL state changes here; GL upload is handled by the pipeline.
3. **Capybara kernels** stay in `.cu/.cuh` under a stable minimal interface; no host logging from within device code except via the device log buffer.
4. **Renderer_* (window/loop/resources/state)** wires OS/GL context, input events, and calls `FramePipeline::execute` each frame.
5. **Overlays** draw only after the base fullscreen quad. They must not mutate renderer state; inputs are plain data (text or metrics).
6. **Settings** is read-only during runtime; changing compile-time flags should not require touching multiple modules.
7. **PBO ring discipline**: only `FramePipeline` increments ring use/skip counters and emits ring logs.
8. **Logging**: exactly one ASCII PERF line format per N frames; avoid ad-hoc prints elsewhere.
9. **Headers**: keep them slim. Do not include GL headers in public CUDA interop headers; prefer forward decls.
10. **Include hygiene**: if a header gets >10 fan-in, freeze its surface (add wrappers elsewhere instead of piling more includes).
