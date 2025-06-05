# 🦦 OtterDream Mandelbrot Renderer (CUDA + OpenGL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ein schneller, moderner Mandelbrot-Renderer für Windows basierend auf **CUDA** und **OpenGL 4.3 Core Profile**. Dynamisches Auto-Zooming, sanfte Farbverläufe und ein leichtgewichtiges, shaderbasiertes HUD.

---

## ✨ Features

- **CUDA-Optimiert**: Schnelles Mandelbrot-Rendering mit progressiver Verfeinerung.
- **OpenGL 4.3 Core Profile**: Moderne Shader-Pipeline ohne Fixed-Function OpenGL.
- **Auto-Zoom**: Automatisches Zoomen auf interessante Bildregionen basierend auf Komplexität.
- **Dynamisches Hue Coloring**: Farbverlauf abhängig vom Zoom-Level.
- **HUD (FPS, Zoom, Offset)**: Eingeblendetes HUD mit modernem OpenGL (Shader-basiert).
- **Fenster-Resizing**: Dynamische Anpassung des Viewports bei Größenänderung.
- **Smooth Iteration Coloring**: Weiche Farbübergänge für hohe Zoomstufen.
- **Progressive Iterationen**: Automatische Erhöhung der Iterationszahl für immer feinere Details.

---

## 🛠️ Voraussetzungen

- Windows 10/11
- NVIDIA CUDA Toolkit (12.9 empfohlen)
- Visual Studio 2022 (mit C++ und CUDA-Support)
- CMake 3.24+ und Ninja
- Vcpkg (für Abhängigkeitsverwaltung)

---

## 📦 Abhängigkeiten (über vcpkg)

- **GLFW** – Fenster- und Eingabemanagement
- **GLEW** – OpenGL Extension Wrangler
- **STB Easy Font** – Leichtgewichtiges Text-Rendering für HUD

---

## ⚙️ Build-Anleitung

### 1. Vcpkg Setup

```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat
vcpkg integrate install
vcpkg install glfw3 glew
