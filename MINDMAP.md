```mermaid
mindmap
  root((Mandelbrot Otterdream))
    Build & Tooling
      Rust runner (otter_proc)
        Live progress (spinner/%/ETA)
        Color tags & ASCII bar
        Metrics seeding (.build_metrics)
        Env toggles (OTTER_COLOR/PROGRESS/ASCII)
      PowerShell build.ps1
      CMake + vcpkg (GLEW dynamic)
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
      Heatmap (preview, EC off)
      Text HUD (Warzenschwein)
    Project Settings, Common & PCH
      Global flags & constants
      Helpers / PCH
    GL Shaders
      Palette sampling, etc.
    Logging & Diagnostics (Luchs)
      Host & CUDA device log buffers
      ASCII-only, PERF line cadence
```
# Architecture Mindmap & Responsibilities (Updated 2025-11-01)

> Reflektiert den aktuellen Stand inkl. **Rust-Runner mit Live-Progress**, **GLEW dynamisch**, **Heatmap = Preview (EC off)**.

## Module Buckets & Files

### Build & Tooling
- **Zweck:** Komfortabler Build mit Live-Progress, ETA, farbigen Tags; stabile CMake/VCPKG-Kette.
- **Artefakte:** `.build_metrics/metrics.json` (ETA-Seeding), ANSI-Farben per `ENABLE_VIRTUAL_TERMINAL_PROCESSING`.
- **Toggles:** `OTTER_PROGRESS=0`, `OTTER_COLOR=0`, `OTTER_ASCII=1`.

- `build.ps1`
- `rust/otter_proc/src/runner.rs`
- `rust/otter_proc/src/runner_progress.rs`
- `rust/otter_proc/src/runner_term.rs`
- `rust/otter_proc/src/runner_classify.rs`

### Renderer Orchestration & Windowing
- High-level Render-Loop, Window, Input, „swapchain“-Flow.
- Verbindet `FramePipeline::execute` mit GL-Ressourcen und CUDA-Interop.

- `src/main.cpp`
- `src/renderer_core.cu` / `src/renderer_core.hpp`
- `src/renderer_loop.cpp` / `src/renderer_loop.hpp`
- `src/renderer_pipeline.cpp` / `src/renderer_pipeline.hpp`
- `src/renderer_resources.cpp` / `src/renderer_resources.hpp`
- `src/renderer_state.hpp`
- `src/renderer_state_cuda.cpp`
- `src/renderer_state_gl.cpp`
- `src/renderer_window.cpp` / `src/renderer_window.hpp`

### Frame Pipeline & Timing
- Einziger Ort für pro-Frame Ablauf & Timings: compute → PBO→Tex-Upload → Base-Draw → Overlays → **eine feste ASCII-PERF-Zeile**.
- Pflegt `FrameContext` und Ring-Statistiken; triggert Zoom-Logik.

- `src/capybara_frame_pipeline.cuh`
- `src/fps_meter.cpp` / `src/fps_meter.hpp`
- `src/frame_context.cpp` / `src/frame_context.hpp`
- `src/frame_limiter.hpp`
- `src/frame_pipeline.cpp` / `src/frame_pipeline.hpp`

### CUDA Interop & Device Buffers
- PBO map/unmap für CUDA; startet Capybara-Kernels & Colorizer; unmap; GL-Upload macht die Pipeline.
- `bear_CudaPBOResource` kapselt GL↔CUDA; `Hermelin::CudaDeviceBuffer` kapselt Device-Speicher.

- `src/bear_CudaPBOResource.cpp` / `src/bear_CudaPBOResource.hpp`
- `src/cuda_interop.cu` / `src/cuda_interop.hpp`
- `src/hermelin_buffer.cpp` / `src/hermelin_buffer.hpp`

### Capybara Compute Pipeline (Mandelbrot)
- Device-seitige Iteration (Escape-Time), Mapping, Zahlentricks.
- Host hängt nur an einer schmalen API.

- `src/capybara_api.cuh`
- `src/capybara_integration.cuh`
- `src/capybara_mapping.cuh`
- `src/capybara_math.cuh`
- `src/capybara_pixel_iter.cuh`
- `src/capybara_render_kernel.cu`
- `src/capybara_ziter.cuh`

### Colorization & Post-processing
- 16-bit Iterationen → RGBA in den gemappten PBO, optional Palette.

- `src/colorize_iterations.cu` / `src/colorize_iterations.cuh`

### Overlays & HUD
- **Heatmap (Preview):** EC aktuell **aus**; Overlay zeigt vorberechnete Tiles.
- **Warzenschwein:** HUD-Text; letzter Draw-Pass mit Blending.

- `src/heatmap_overlay.cpp` / `src/heatmap_overlay.hpp`
- `src/hud_text.cpp` / `src/hud_text.hpp`
- `src/warzenschwein_overlay.cpp` / `src/warzenschwein_overlay.hpp`

### Project Settings, Common & PCH
- Zentrale Compile-Time-Schalter, Helpers, PCH.

- `src/common.hpp`
- `src/pch.hpp`
- `src/settings.hpp`

### Logging & Diagnostics (Luchs)
- Host-Logger (ASCII-only) & CUDA-Device-Logpuffer.
- Numerische Return-Codes; Device-Flush nur bei Fehlerpfaden/gezielt.

- `src/luchs_cuda_log_buffer.cu` / `src/luchs_cuda_log_buffer.hpp`
- `src/luchs_log_device.hpp`
- `src/luchs_log_host.cpp` / `src/luchs_log_host.hpp`

### Other
- `src/frame_capture.cpp` / `src/frame_capture.hpp`
- `src/heatmap_utils.hpp`
- `src/warzenschwein_fontdata.hpp`
- `src/zoom_logic.cpp` / `src/zoom_logic.hpp`

## Ownership & Boundaries (konkret)
1. **FramePipeline** sequenziert den Frame allein; Overlays rufen **keine** CUDA auf, `cuda_interop.hpp` bleibt GL-frei.
2. **CUDA Interop**: nur Map/Unmap + Launches; GL-Uploads macht die Pipeline.
3. **Capybara** bleibt in `.cu/.cuh`; Host-Logs aus Device nur via Log-Puffer.
4. **Renderer*** (Window/Loop/Resources/State) verdrahtet OS/GL, triggert pro Frame `FramePipeline::execute`.
5. **Overlays** zeichnen nach dem Base-Quad; keine Mutation von Renderer-State.
6. **Settings** zur Laufzeit read-only; Compile-Flags dürfen nicht quer durchs Projekt angepasst werden müssen.
7. **PBO-Ring**: Inkrement/Logs ausschließlich in der Pipeline.
8. **Logging**: genau *ein* ASCII-PERF-Format; kein Wildwuchs.
9. **Header-Hygiene**: GL nicht in öffentlichen CUDA-Headers; schmale Oberflächen bevorzugen.
