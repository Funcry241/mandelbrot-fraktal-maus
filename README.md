<!-- Datei: README.md -->
<!-- 🐭 Maus-Kommentar: README für Alpha 81+ – CI-validiert, Silk-Lite Zoom integriert, Capybara Single-Path (keine EC/Wrapper), Logs als Epoch-Millis. CUDA 13 Pflicht für den Build; zum Ausführen reicht der NVIDIA-Treiber. GLEW dynamisch, DIST enthält die nötigen DLLs. -->

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
Seit **Alpha 81**: CI-validiert, deterministisch, sanfter **Silk-Lite**-Zoom – und kompakte **Epoch-Millis**-Logs.

> **Wichtig (Änderung)**: Ab diesem Stand rendert OtterDream über einen **einzigen aktiven Pfad**:  
> **Capybara -> Iterationen -> Colorizer -> PBO**.  
> Es gibt **keinen Referenz-Orbit / keine Perturbation** und **keine EC/Wrapper** im aktiven Code.

---

## ✨ Sofort testen (Windows, **ohne Setup**)

Wenn du das Projekt **nur ausprobieren** möchtest, brauchst du **nichts zu installieren**.  
Im Repository/Arbeitsverzeichnis liegt ein **`dist\`**-Ordner mit einer **portablen Windows-Build**:

```
dist\
  mandelbrot_otterdream.exe   ← doppelklicken & starten
  glew32.dll                  ← bereits mitgeliefert
  glfw3.dll                   ← bereits mitgeliefert
```

**Voraussetzung zum Ausführen:** Ein aktueller **NVIDIA-Grafiktreiber** und OpenGL 4.3 (Bestandteil von Windows-Treiberpaketen).  
**Nicht nötig zum Ausführen:** Visual Studio, vcpkg oder das CUDA Toolkit (wir linken die CUDA Runtime statisch).

> Falls `dist\mandelbrot_otterdream.exe` fehlt: einmal bauen (siehe unten „Automatischer Build“) – der Build füllt `dist\` automatisch.

---

## 🔧 Automatischer Build (Windows) – `build.ps1`

Der einfachste Weg, den Build zu starten, ist das **PowerShell-Skript** `build.ps1`.  
Es orchestriert alles: vcpkg-Abhängigkeiten, CMake-Konfiguration/Build, und das **Befüllen von `dist\`** (EXE + benötigte DLLs).

```powershell
# Aus dem Repo-Root
powershell -ExecutionPolicy Bypass -File .\build.ps1
# Optional: Konfiguration wählen (Default: RelWithDebInfo)
powershell -ExecutionPolicy Bypass -File .\build.ps1 -Configuration Release
```

**Was `build.ps1` für dich erledigt**
- Öffnet die passende **VS-Entwicklungsumgebung**, konfiguriert **CMake + Ninja**
- Installiert/prüft **GLFW** & **GLEW** via **vcpkg** (dynamisch)
- Baut das Projekt und kopiert **EXE + DLLs nach `dist\`**
- Zeigt Live-Progress (Spinner, %, ETA) über den **Rust-Runner** mit farbigen `[PS]/[RUST]/[RUNNER]`-Tags

> Du brauchst lediglich **PowerShell 5.1**, Visual Studio 2022 Build-Tools und das **CUDA Toolkit 13** (nur zum **Bauen**; zum **Ausführen** nicht nötig).

---

## 🔧 Manueller Build (CMake)

> Der Build läuft vollständig über **Standard-CMake** (host-agnostisch). `build.ps1` ist nur Komfort.

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
# Installationsbaum erzeugen (optional)
cmake --install build --prefix .\dist
# Starten
.\dist\mandelbrot_otterdream.exe
```

### 3) Linux (GCC + Ninja)

```bash
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE="$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# Installationsbaum erzeugen (optional)
cmake --install build --prefix ./dist
# Starten
./dist/mandelbrot_otterdream
```

> **Tipp:** Compute Capability beim Konfigurieren überschreiben:
>
> ```bash
> cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_TOOLCHAIN_FILE="$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake" -DCMAKE_BUILD_TYPE=Release
> ```

---

## 🧠 Features

* **🚀 CUDA Rendering (Capybara)** – schnelle Iterationen, Event-Timing via CUDA-Events (kein globales `cudaDeviceSynchronize()` im Normalpfad), **Survivor-Black** gegen Ghosting.
* **🪶 Silk-Lite Motion Planner (Auto-Zoom)** – sanft, yaw-limitiert, hysterese-/lock-gestützt (EC-Signal derzeit deaktiviert, **ForceAlwaysZoom** aktiv).
* **🕳️ Anti-Black-Guard** – Warm-up-Drift + Void-Bias: kein „Zoom ins Schwarze“.
* **📈 Progressive Iterationen** – Zoom-abhängige Iterationen; **standardmäßig aktiv**.
* **🎨 GT-Palette + Smooth Coloring** – linearer Farbraum, `it - log2(log2(|z|))`; Streifen optional (off).
* **🖼️ Echtzeit-OpenGL + CUDA-Interop** – PBO-interoperabel (`cudaGraphicsGLRegisterBuffer`).
* **📊 Heatmap-Overlay (Eule – Preview)** – GPU-Shader-Basis; derzeit ohne EC-Eingang.
* **🧰 HUD & ASCII-Debug (Warzenschwein)** – FPS/Zoom/Offset; Logs sind **ASCII-only**.
* **🤖 Auto-Tuner** – zyklische Parameterausgabe ins Log, kein JSON-Hot-Reload nötig.

---

## 🖥️ Systemvoraussetzungen

* Windows 10/11 **oder** Linux
* **NVIDIA GPU** mit CUDA (Compute Capability **8.0+**, empfohlen **8.6+**)
* **Für den Build:** **CUDA Toolkit v13.0+**, Visual Studio 2022 (MSVC) bzw. GCC 11+, CMake ≥ 3.28, Ninja, vcpkg
* **Für das Ausführen (nur Windows, via `dist\`):** **nur NVIDIA-Treiber** (OpenGL 4.3 Core)

> ⚠️ GPUs unter Compute Capability 8.0 (z. B. Kepler/Maxwell) werden **nicht** unterstützt.

---

## 📦 Abhängigkeiten (via vcpkg)

* [GLFW](https://www.glfw.org/) – Fenster-/Eingabe-Handling  
* [GLEW](http://glew.sourceforge.net/) – OpenGL-Extension-Management (**dynamisch**, DLL im `dist\`)

---

### ⌨️ Keyboard Controls

* `P` – Auto-Zoom pausieren/fortsetzen  
* `H` – Heatmap-Overlay an/aus (derzeit ohne EC-Daten)  
* `T` – HUD (Warzenschwein) an/aus  

> Hinweis: `Space` ist derzeit **nicht** gemappt (kein Alias zu `P`).

---

## 🌊 Das Robbe-Prinzip (API-Synchronität)

Header und Source bleiben **synchron**. Kein Drift, kein API-Bruch. Die Robbe wacht.

**Referenz-Signaturen (Auszug):**
* `src/cuda_interop.hpp` -> **`renderCudaFrame(...)`**
* `src/capybara_frame_pipeline.cuh` -> **`capy_render(...)`**

---

## 🔎 Qualitäts-Guards (Kurzüberblick)

* **Anti-Black-Guard** – Warm-up-Drift & Void-Bias  
* **Survivor-Black** – Ghosting-freie Slices  
* **Hysterese/Lock** – kein Flip-Flop  
* **Retarget-Throttle** – ruhige Kurswechsel  
* **Softmax-Sparsification** – ignoriert irrelevante Tails (EC aktuell deaktiviert)

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
