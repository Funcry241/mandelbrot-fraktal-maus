<!-- Datei: README.md -->
<!-- 🐭 Maus-Kommentar: README für Alpha 81+ – CI-validiert, Silk-Lite Zoom integriert, Capybara Single-Path (keine EC/Wrapper), Logs als Epoch-Millis. CUDA 13 Pflicht; GLEW dynamisch. Schneefuchs: „Nur was synchron ist, bleibt stabil.“ -->

# 🦦 OtterDream Mandelbrot Renderer (CUDA + OpenGL)

[![Build Status](https://github.com/Funcry241/mandelbrot-fraktal-maus/actions/workflows/ci.yml/badge.svg)](https://github.com/Funcry241/mandelbrot-fraktal-maus/actions/workflows/ci.yml)
![CUDA](https://img.shields.io/badge/CUDA-13%2B-76b900?logo=nvidia)
![C%2B%2B](https://img.shields.io/badge/C%2B%2B-20-blue)
![OpenGL](https://img.shields.io/badge/OpenGL-4.3%2B-3D9DD6)
![Platforms](https://img.shields.io/badge/Platforms-Windows%20%7C%20Linux-informational)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="assets/hero_russelwarze.jpg" alt="OtterDream Mandelbrot – Rüsselwarze Mode" width="85%">
</p>

Ein ultraschneller Mandelbrot-Renderer mit CUDA-Beschleunigung und OpenGL-Anzeige für moderne NVIDIA-GPUs. Der Renderer zoomt automatisch in interessante Regionen und erhöht fortlaufend die Detailtiefe.
Seit **Alpha 81**: CI-validiert, deterministisch, sanfter **Silk-Lite**-Zoom - und kompakte **Epoch-Millis**-Logs.

> **Wichtig (Änderung)**: Ab diesem Stand rendert OtterDream über einen **einzigen aktiven Pfad**:  
> **Capybara → Iterationen → Colorizer → PBO**.  
> Es gibt **keinen Referenz-Orbit / keine Perturbation** und **keine EC/Wrapper** im aktiven Code.

---

## 🧠 Features

* **🚀 CUDA Rendering (Capybara)**  
  Iterations-Render über Capybara, ereignisbasiertes **Event-Timing** via CUDA-Events (ohne globales `cudaDeviceSynchronize()` im Normalpfad).  
  * **Survivor-Black**: unfertige Pixel sofort schwarz -> *kein Ghosting* zwischen Slices.  
  * **WARP_CHUNK**-basiertes Pacing (warp-synchron).

* **🪶 Silk-Lite Motion Planner (Auto-Zoom)**  
  Sanfte Schwenks, **Yaw-Rate-Limiter** (rad/s) + Längendämpfung, relative Hysterese & kurzer Lock gegen Flip-Flop.  
  **Hinweis:** Die frühere Entropie/Kontrast-Analyse ist aktuell **deaktiviert**; es wirkt der **ForceAlwaysZoom**-Fallback für stetige Bewegung.

* **🕳️ Anti-Black-Guard (Cardioid/Bulb-Avoidance)**  
  Warm-up-Drift und **Void-Bias** schieben den Fokus verlässlich aus Innenbereichen -> *kein „Zoom ins Schwarze“*.

* **📈 Progressive Iterationen (Zoom-abhängig)**  
  Iterationszahl steigt automatisch mit dem Zoom-Level. **Standardmäßig aktiv** (abschaltbar).

* **🎨 GT-Palette (Cyan→Amber) + Smooth Coloring**  
  Interpolation im **Linearraum** gegen Banding, **Smooth Coloring** via `it - log2(log2(|z|))`.  
  **Streifen-Shading** optional – **standardmäßig aus** (`stripes = 0.0f`) für ringfreie Darstellung.  
  **Mapping-Vertrag:** *Innenpunkte schreiben `iterOut = maxIter`*, Escape schreibt die Iterationsnummer.

* **🖼️ Echtzeit-OpenGL + CUDA-Interop**  
  Anzeige via Fullscreen-Quad, direkte PBO-Verbindung (`cudaGraphicsGLRegisterBuffer`).

* **📊 Heatmap-Overlay (Eule – Preview)**  
  GPU-Shader-Variante im Aufbau; **derzeit ohne EC-Signal**.

* **🧰 HUD & ASCII-Debug (Warzenschwein)**  
  FPS, Zoom, Offset – optional. **Logging ist ASCII-only** und wirkt nicht auf Berechnungs-/Render-Pfade.

* **🤖 Auto-Tuner**  
  Findet ohne Neustart zyklisch sinnvolle Ziel-/Zoom-Parameter und schreibt sie ins Log (kein JSON-Reload nötig).

---

## 🆕 Neu in dieser Version (Alpha 81+)

* **Single-Path Renderer**: Capybara → Colorizer → PBO (klassischer/perturbierter Pfad sowie EC-Wrapper entfernt)
* **Survivor-Black** (ghosting-frei) & **Event-Timing** (CUDA-Events)
* **Anti-Black-Guard** (Warm-up-Drift + Void-Bias)
* **Yaw-Limiter** + **Längendämpfung**, **Hysterese/Lock** & dyn. **Retarget-Throttle**
* **Softmax-Sparsification** (Designbestandteil; aktuell ohne EC-Eingang aktiv)  
* **Epoch-Millis-Logging** (UTC-Millis seit 1970) - kompakt, sortier- & skriptfreundlich

---

## 🖥️ Systemvoraussetzungen

* Windows 10/11 **oder** Linux
* **NVIDIA GPU** mit CUDA (Compute Capability **8.0+**, empfohlen **8.6+**)
* **CUDA Toolkit v13.0+ (erforderlich)** – inkl. `nvcc`
* Visual Studio 2022 **oder** GCC 11+
* CMake (Version **≥ 3.28**), Ninja
* vcpkg (für GLFW, GLEW; **GLEW dynamisch**, kein `GLEW_STATIC`)

> ⚠️ GPUs unter Compute Capability 8.0 (z. B. Kepler, Maxwell) werden **nicht** unterstützt.  
> ⚠️ OpenGL **4.3 Core** wird vorausgesetzt.

---

## 📦 Abhängigkeiten (via vcpkg)

* [GLFW](https://www.glfw.org/) – Fenster-/Eingabe-Handling  
* [GLEW](http://glew.sourceforge.net/) – OpenGL-Extension-Management (**dynamisch**)

---

## 🦀 Rust Build Runner – Live Progress (optional)

Der **Rust-Runner `otter_proc`** orchestriert den Build mit **Live-Progress** (Spinner, **%**, **ETA**, ASCII-Bar) und farbigen Tags (`[PS]`, `[RUST]`, `[PROC]`).  
Robuste Parser erkennen `68%` und **Ratio** `[17/45]`; Animation rate-limited auf **200 ms**.  
**Metrik-Seeding:** `.build_metrics/metrics.json` speichert Laufzeiten pro Phase zur besseren ETA.

**Toggles (Environment):**
- `OTTER_PROGRESS=0` – Progress-UI aus (Default: an)  
- `OTTER_COLOR=0` – Farben aus (Default: an)  
- `OTTER_ASCII=1` – ASCII-Spinner/Balken

---

## 🔧 Build-Anleitung

> **Hinweis:** Der Build läuft vollständig über **Standard-CMake** (host-agnostisch).
> Ein **optionales** PowerShell-Skript `build.ps1` kann vorhanden sein, wird aber nicht benötigt.

### 1) Repository & vcpkg holen

```bash
git clone --recurse-submodules https://github.com/Funcry241/mandelbrot-fraktal-maus.git
cd mandelbrot-fraktal-maus
# vcpkg lokal bootstrappen (unter Windows .bat verwenden)
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh            # Linux/macOS
bootstrap-vcpkg.bat             # Windows (PowerShell oder CMD)
cd ..
```

### 2) Windows (MSVC + Ninja)

```powershell
cmake -S . -B build -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="${PWD}/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# (optional) Installationsbaum erzeugen
cmake --install build --prefix .\dist
# Ausführen
.\dist\mandelbrot_otterdream.exe
```

### 3) Linux (GCC + Ninja)

```bash
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE="$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# (optional) Installationsbaum erzeugen
cmake --install build --prefix ./dist
# Ausführen
./dist/mandelbrot_otterdream
```

> **Tipp:** Abweichende Compute Capability beim Konfigurieren überschreiben:
>
> ```bash
> cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_TOOLCHAIN_FILE="$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake" -DCMAKE_BUILD_TYPE=Release
> ```

---

### ⌨️ Keyboard Controls

* `P`: Auto-Zoom pausieren/fortsetzen
* `H`: Heatmap-Overlay ein/aus (derzeit ohne EC-Daten)
* `T`: HUD (Warzenschwein) ein/aus

> Hinweis: `Space` ist derzeit **nicht** gemappt (kein Alias zu `P`).

---

## 🌊 Das Robbe-Prinzip (API-Synchronität)

**Seit Alpha 41 gilt:** Header und Source bleiben **synchron**. Kein Drift, kein API-Bruch. Die Robbe wacht.

> „API-Änderung ohne Header-Update? Dann OOU-OOU und Build-Fehler!“

**Referenz-Signaturen (aktuell):**
* `src/capybara_frame_pipeline.cuh` → **`capy_render(...)`**
* `src/cuda_interop.hpp` → **`renderCudaFrame(...)`** (Overloads)

---

## 🦝 Waschbär-Prinzip (Auto-Fix & Hygiene)

**Ab Alpha 53:** Der Build prüft auf bekannte Toolchain-Fallen (z. B. `glew32d.lib`) und hält CMake-Hygiene hoch - **ohne** projektexterne Skripte.

---

## 🔎 Qualitäts-Guards (Kurzüberblick)

* **Anti-Black-Guard**: Warm-up-Drift & Void-Bias – kein „Zoom ins Schwarze“
* **Survivor-Black**: Ghosting-freie Slices
* **Hysterese/Lock**: verhindert Ziel-Flip-Flops
* **Retarget-Throttle**: CPU-schonend, ruhiger Kurs
* **Softmax-Sparsification**: ignoriert irrelevante Tails (EC aktuell deaktiviert)

---

## 🧭 Zoomgerichtet & geschmacksgetestet

Silk-Lite koppelt **Zielwahl** und **Bewegung**. Der Designpfad sieht Entropie/Kontrast als Signalquelle vor; aktuell ist EC **deaktiviert**.  
Der Planner arbeitet daher mit **ForceAlwaysZoom**, Yaw-Limiter, Dämpfung, Hysterese/Lock und Retarget-Throttle für ruhige, stetige Kamerafahrten - ohne „ins Schwarze“ zu kippen.

---

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
- *Single place* for per-frame control flow and timings - maps PBO → texture, draws base image, draws overlays, logs one fixed ASCII PERF line.
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

---

## 📄 Lizenz

MIT-Lizenz – siehe [LICENSE](LICENSE).

---

**OtterDream** – von der Raupe zum Fraktal-Schmetterling 🦋  
*Happy Zooming!*

🐭 Maus sorgt für Fokus und ASCII-Sauberkeit.  
🦊 Schneefuchs bewacht die Präzision.  
🦦 Otter treibt den Zoom unaufhaltsam.  
🦭 Robbe schützt die API-Würde.  
🦝 Waschbär hält den Build hygienisch.  
🦉 Eule sorgt für Überblick in Heatmap & Koordinaten.
