# 🦦 OtterDream Mandelbrot Renderer (CUDA + OpenGL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ein schneller, moderner Mandelbrot-Renderer für Windows basierend auf CUDA und OpenGL 4.3 Core Profile. Dynamisches Auto-Zooming, sanfte Farbverläufe und ein leichtgewichtiges HUD.

---

## Features

- **CUDA-Optimiert**: Schnelles Mandelbrot-Rendering mit progressiver Verfeinerung.
- **OpenGL 4.3 Core Profile**: Moderne Shader-Pipeline ohne Fixed-Function OpenGL.
- **Auto-Zoom**: Automatisches Zoomen auf interessante Bildregionen.
- **Dynamic Hue Coloring**: Farbverlauf abhängig vom Zoom-Level.
- **HUD (FPS/Zoom)**: Eingeblendetes HUD via Shader (keine veralteten Funktionen).
- **Resizing**: Fenstergrößenänderung mit dynamischem Viewport.
- **Smooth Iteration Coloring**: Feine Farbübergänge für hohe Zoomstufen.
- **Progressive Iterationen**: Automatisches Hochzählen der Iterationen.

---

## Voraussetzungen

- Windows 10/11
- NVIDIA CUDA Toolkit (v12.9 empfohlen)
- Visual Studio 2022 (mit C++ und CUDA Support)
- CMake 3.24+ und Ninja
- Vcpkg (für GLFW, GLEW)

---

## Abhängigkeiten (über vcpkg)

- **GLFW**: Fenster und Eingabe
- **GLEW**: OpenGL Extension Wrangler
- **STB Easy Font**: Leichtgewichtiges Text-Rendering

---

## Build-Anleitung

### Vcpkg Setup

```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat
vcpkg integrate install
vcpkg install glfw3 glew
