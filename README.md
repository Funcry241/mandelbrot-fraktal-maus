# 🦦 OtterDream Mandelbrot Renderer (CUDA + OpenGL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ein schneller, moderner Mandelbrot-Renderer für Windows basierend auf CUDA und OpenGL 4.3 Core Profile. Dynamisches Auto-Zooming, sanfte Farbverläufe und ein leichtgewichtiges HUD.

---

## Features

- **CUDA-Optimiert**: Schnelles Mandelbrot-Rendering mit progressiver Verfeinerung.
- **OpenGL 4.3 Core Profile**: Moderne Shader-Pipeline — keine veralteten OpenGL-Features.
- **Auto-Zoom**: Automatisches Zoomen auf interessante Regionen des Fraktals.
- **Dynamic Hue Coloring**: Farbverlauf abhängig vom Zoom-Level.
- **HUD (FPS, Zoom, Offset)**: Über Shader gerendertes Head-Up-Display — kompatibel mit Core Profile.
- **Resizing**: Anpassung an Fenstergrößenänderung mit dynamischem glViewport.
- **Smooth Iteration Coloring**: Feine Farbübergänge für hohe Zoomstufen.
- **Progressive Iterationen**: Iterationsanzahl erhöht sich automatisch für feinere Details.
- **Double-Präzision**: CUDA- und CPU-Operationen mit hoher Genauigkeit.

---

## Voraussetzungen

- Windows 10 oder 11
- NVIDIA CUDA Toolkit (v12.9 empfohlen)
- Visual Studio 2022 (C++ Desktop Development + CUDA-Support)
- CMake 3.24+ und Ninja
- [vcpkg](https://github.com/microsoft/vcpkg) (Paketmanager für C++-Bibliotheken)

---

## Abhängigkeiten (über vcpkg)

- **GLFW**: Fenster- und Eingabeverwaltung
- **GLEW**: OpenGL-Extension-Management
- **STB Easy Font**: Leichtgewichtiges Text-Rendering für das HUD

---

## Build-Anleitung

### 1. vcpkg Setup

```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat
vcpkg integrate install
vcpkg install glfw3 glew
