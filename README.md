<!-- Datei: README.md -->
<!-- Zeilen: 129 -->
<!-- 🐭 Maus-Kommentar: README für Alpha 20 – Build-Anleitung jetzt vollständig und CI-kompatibel, mit klarem vcpkg-Weg und aktualisierter CMake-Version. Schneefuchs: „Wer bauen will, muss vorher graben – im richtigen Verzeichnis.“ -->

# 🦦 OtterDream Mandelbrot Renderer (CUDA + OpenGL)

[![Build Status](https://github.com/Funcry241/otterdream-mandelbrot/actions/workflows/ci.yml/badge.svg)](https://github.com/Funcry241/otterdream-mandelbrot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ein ultraschneller Mandelbrot-Renderer mit CUDA-Beschleunigung und OpenGL-Anzeige, entwickelt für moderne NVIDIA-GPUs. Der Renderer zoomt automatisch in interessante Regionen hinein und erhöht dabei fortlaufend die Detailtiefe.

---

## 🧠 Features

- **🚀 CUDA Rendering**  
  Fraktale GPU-beschleunigt berechnen (Blockweise, Tile-optimiert)
- **🎯 Auto-Zoom mit Entropieanalyse**  
  Erkennt kontrastreiche Bereiche und zoomt hinein
- **📈 Progressive Iterationen**  
  Iterationszahl steigt nur bei Zoom automatisch
- **🎨 Smooth Coloring**  
  Sanfte Farbverläufe (smoothed iteration count)
- **🔍 Adaptive Tile-Größe**  
  Passt Tile-Größe an Zoomlevel an (mehr Details bei starker Vergrößerung)
- **🖼️ Echtzeit-OpenGL**  
  Anzeige über Fullscreen-Quad, keine Altlasten (Core Profile 4.3)
- **🔄 Fenster-Resize & dynamischer Viewport**
- **🔗 CUDA/OpenGL Interop über `cudaGraphicsGLRegisterBuffer`**
- **🧰 HUD & Debug-Ausgaben (via stb_easy_font, optional)**

---

## 🖥️ Systemvoraussetzungen

- Windows 10 oder 11 **oder Linux**
- **NVIDIA GPU** mit CUDA (Compute Capability **8.0+**, empfohlen **8.6+**)
- CUDA Toolkit (empfohlen: **v12.9**)
- Visual Studio 2022 **oder** GCC 11+
- CMake (Version **≥3.28**), Ninja
- vcpkg (für GLFW, GLEW)

> ⚠️ Hinweis: GPUs unter Compute Capability 8.0 (z. B. Kepler, Maxwell) werden **nicht unterstützt**.

---

## 📦 Abhängigkeiten (via vcpkg)

- [GLFW](https://www.glfw.org/) – Fenster- und Eingabe-Handling  
- [GLEW](http://glew.sourceforge.net/) – OpenGL-Extension-Management  
- [stb_easy_font](https://github.com/nothings/stb/blob/master/stb_easy_font.h) – Schriftanzeige im HUD *(optional)*

---

## 🔧 Build-Anleitung

### 📁 Vcpkg Setup

```bash
git clone --recurse-submodules https://github.com/Funcry241/otterdream-mandelbrot.git
cd otterdream-mandelbrot
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh   # oder .bat unter Windows
cd ..
```

---

### 🪟 Windows Build

```powershell
cmake --preset windows-msvc
cmake --build --preset windows-msvc
cmake --install build/windows --prefix ./dist
.\dist\mandelbrot_otterdream.exe
```

> 🛠 `build.ps1` automatisiert alle Schritte (Build, Install, Run)

---

### 🐧 Linux Build

> Voraussetzung: CUDA 12.9, GCC, Ninja, CMake ≥3.28, OpenGL-Treiber, vcpkg

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build libglfw3-dev libglew-dev libxmu-dev libxi-dev libglu1-mesa-dev xorg-dev pkg-config
```

```bash
cmake --preset linux-build
cmake --build --preset linux-build
cmake --install build/linux --prefix ./dist
./dist/mandelbrot_otterdream
```

---

### ⌨️ Keyboard Controls

- `P` or `Space`: Pause/resume automatic zoom
- `H`: Toggle heatmap overlay (entropy/contrast)

---

### ⚙️ Customizing CUDA Architectures

By default, this project targets CUDA compute capabilities 8.0, 8.6, 8.9, and 9.0 (i.e. architectures 80;86;89;90).
If your GPU has a different compute capability (e.g. RTX 4090 with Arch 90), override it like this:

```bash
cmake --preset windows-release -DCMAKE_CUDA_ARCHITECTURES=90
```

Find your GPU's capability [here](https://developer.nvidia.com/cuda-gpus).

---

## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz – siehe [LICENSE](LICENSE) für Details.

---

**OtterDream** – von der Raupe zum Fraktal-Schmetterling 🦋  
*Happy Zooming!*


🐭 This project owes a mouse more than it admits.
