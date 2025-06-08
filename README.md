# 🦦 OtterDream Mandelbrot Renderer (CUDA + OpenGL)

[![Build Status](https://github.com/dein-benutzername/otterdream-mandelbrot/actions/workflows/ci.yml/badge.svg)](https://github.com/dein-benutzername/otterdream-mandelbrot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ein schneller, moderner Mandelbrot-Renderer für Windows basierend auf CUDA und OpenGL 4.3 Core Profile. Dynamisches Auto-Zooming, sanfte Farbverläufe und ein flexibler Renderer.

---

## Features

- **CUDA-Optimiert**: Schnelles Mandelbrot-Rendering mit progressiver Verfeinerung.
- **OpenGL 4.3 Core Profile**: Moderne Shader-Pipeline ohne Fixed-Function OpenGL.
- **Auto-Zoom**: Automatisches Zoomen auf interessante Bildregionen.
- **Dynamic Hue Coloring**: Farbverlauf abhängig vom Zoom-Level.
- **Smooth Iteration Coloring**: Feine Farbübergänge für hohe Zoomstufen.
- **Progressive Iterationen**: Automatisches Hochzählen der Iterationen.
- **Resizing**: Fenstergrößenänderung mit dynamischem Viewport.
- **GPU-Kompatibilität**: Läuft auf GPUs ab **Compute Capability 3.0**.

---

## Voraussetzungen

- Windows 10/11
- NVIDIA CUDA Toolkit (v12.9 empfohlen)
- Visual Studio 2022 (mit C++ und CUDA Support)
- CMake **>4.0** und Ninja
- vcpkg (für GLFW, GLEW)

> **Hinweis:** Keine Dynamic Parallelism-Unterstützung erforderlich — der Renderer läuft auf GPUs ab Compute Capability 3.0.

---

## Abhängigkeiten (über vcpkg)

- **GLFW**: Fenster und Eingabe
- **GLEW**: OpenGL Extension Wrangler
- **STB Easy Font**: Leichtgewichtiges Text-Rendering *(optional für späteres HUD)*

---

## Build-Anleitung

### Vcpkg Setup

```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat
vcpkg integrate install
vcpkg install glfw3 glew
```

---

### Windows Build

```bash
git clone https://github.com/dein-benutzername/otterdream-mandelbrot.git
cd otterdream-mandelbrot
cmake --preset windows-msvc
cmake --build --preset windows-msvc
.\dist\mandelbrot_otterdream.exe
```

> **Hinweis:** PowerShell-Skript `build.ps1` vorhanden für komfortablen Build-Prozess.

---

### Linux Build

> **Erforderlich**: CUDA Toolkit, GCC, CMake **>4.0**, Ninja, OpenGL-Treiber, GLFW und GLEW Dev-Pakete.

Installiere benötigte Pakete:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build libglfw3-dev libglew-dev
```

Build:
```bash
git clone https://github.com/dein-benutzername/otterdream-mandelbrot.git
cd otterdream-mandelbrot
cmake --preset linux-gcc
cmake --build --preset linux-gcc
./dist/mandelbrot_otterdream
```

---

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz – siehe [LICENSE](LICENSE) für Details.

---

Happy Fractaling! 🚀🦦
