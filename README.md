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
- CMake 3.24+ und Ninja
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
