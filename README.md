<!-- Datei: README.md -->
<!-- Zeilen: 119 -->
<!-- 🐭 Maus-Kommentar: README für Alpha 13 – bereinigt um macOS-Fehlinformation und veraltete Submodul-Anweisung. Schneefuchs: „Nur was stimmt, darf auch in der README glänzen.“ -->

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

- Windows 10 oder 11
- **NVIDIA GPU** mit CUDA (Compute Capability **8.0+**, empfohlen **8.6+**)
- CUDA Toolkit (empfohlen: **v12.9**)
- Visual Studio 2022 mit C++ & CUDA-Komponenten
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
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat
vcpkg integrate install
vcpkg install glfw3 glew
```

---

### 🪟 Windows Build

```bash
git clone https://github.com/Funcry241/otterdream-mandelbrot.git
cd otterdream-mandelbrot
cmake --preset windows-msvc
cmake --build --preset windows-msvc
cmake --install build/windows --prefix ./dist
.\dist\mandelbrot_otterdream.exe
```

> 🛠 Tipp: `build.ps1` ist ein PowerShell-Skript für automatisierten Build & Install

---

### 🐧 Linux Build

> Voraussetzung: CUDA, GCC, Ninja, CMake ≥3.28, OpenGL-Treiber, GLFW & GLEW

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build libglfw3-dev libglew-dev
```

```bash
git clone https://github.com/Funcry241/otterdream-mandelbrot.git
cd otterdream-mandelbrot
cmake --preset linux-build
cmake --build --preset linux-build
cmake --install build/linux --prefix ./dist
./dist/bin/mandelbrot_otterdream
```

---

## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz – siehe [LICENSE](LICENSE) für Details.

---

**OtterDream** – von der Raupe zum Fraktal-Schmetterling 🦋  
*Happy Zooming!*
