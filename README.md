<!-- Datei: README.md -->
<!-- Zeilen: 159 -->
<!-- 🐭 Maus-Kommentar: README für Alpha 53.2 – Waschbär integriert, GLEW-Fallback entschärft, CI-ready, Patchsystem dokumentiert. Schneefuchs sagt: „Erst putzen, dann patchen.“ -->

# 🦦 OtterDream Mandelbrot Renderer (CUDA + OpenGL)

[![Build Status](https://github.com/Funcry241/otterdream-mandelbrot/actions/workflows/ci.yml/badge.svg)](https://github.com/Funcry241/otterdream-mandelbrot/actions/workflows/ci.yml)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ein ultraschneller Mandelbrot-Renderer mit CUDA-Beschleunigung und OpenGL-Anzeige, entwickelt für moderne NVIDIA-GPUs. Der Renderer zoomt automatisch in interessante Regionen hinein und erhöht dabei fortlaufend die Detailtiefe.

---

## 🧠 Features

- **🚀 CUDA Rendering**  
  Fraktale GPU-beschleunigt berechnen (Blockweise, Tile-optimiert)

- **🎯 Auto-Zoom mit Entropie- und Kontrastanalyse**  
  Erkennt kontrastreiche und strukturreiche Bereiche, zoomt fokussiert hinein

- **📈 Progressive Iterationen (Zoom-abhängig)**  
  Iterationszahl steigt mit dem Zoom-Level automatisch

- **🎨 Smooth Coloring**  
  Sanfte Farbverläufe mit stabilisiertem Betrag (kein Farbflimmern)

- **🔍 Adaptive Tile-Größe**  
  Automatische Tile-Anpassung für bessere Detailauswertung bei starkem Zoom

- **🖼️ Echtzeit-OpenGL + CUDA-Interop**  
  Anzeige über Fullscreen-Quad, keine Altlasten, direkte PBO-Verbindung via `cudaGraphicsGLRegisterBuffer`

- **📊 Heatmap-Overlay**  
  Entropie/Kontrast pro Tile sichtbar gemacht – Debug & Analyse

- **🧰 HUD & ASCII-Debug**  
  FPS, Zoom, Offset, optional aktivierbar

- **🦝 Build-Fallback-Logik (Waschbär)**  
  Automatische Bereinigung von vcpkg/glew-Bugs (z. B. `glew32d.lib`)

- **🖋️ Eigenes Font-Overlay (Warzenschwein)**
  HUD-Schrift ohne ImGui oder externe Fontlibs – direkt per OpenGL-Shader

---

## 🖥️ Systemvoraussetzungen

- Windows 10 oder 11 **oder** Linux
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
.uild.ps1
```

> 🧼 Der Build-Skript erkennt und behebt automatisch bekannte Fallstricke:
>
> - `glew32d.lib`-Bug (vcpkg-Falle)  
> - inkonsistente CMake-Caches  
> - fehlende CUDA-Pfade  
>  
> Kein zweiter Durchlauf nötig – dank 🐭-Patchlogik und 🦝 Waschbär-Watchdog.

---

### 🐧 Linux Build

> Voraussetzung: CUDA 12.9, GCC, Ninja, CMake ≥3.28, OpenGL-Treiber, vcpkg

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build libglfw3-dev libglew-dev libxmu-dev libxi-dev libglu1-mesa-dev xorg-dev pkg-config libcuda1-525  # oder libcuda1-545, je nach Treiberversion
```

> *Hinweis:* Je nach Distribution kann die CUDA-Runtime-Bibliothek anders heißen (z. B. `libcuda1-545`)

```bash
cmake --preset linux-build
cmake --build --preset linux-build
cmake --install build/linux --prefix ./dist
./dist/mandelbrot_otterdream
```

---

### ⌨️ Keyboard Controls

- `P` oder `Space`: Auto-Zoom pausieren/fortsetzen  
- `H`: Heatmap-Overlay ein-/ausschalten

---

### ⚙️ Customizing CUDA Architectures

By default, this project targets CUDA compute capabilities 8.0, 8.6, 8.9, and 9.0 (architectures `80;86;89;90`).

If your GPU has a different compute capability, override like this:

```bash
cmake --preset windows-release -DCMAKE_CUDA_ARCHITECTURES=90
```

Find your GPU's capability [here](https://developer.nvidia.com/cuda-gpus).

---

## 🌊 Das Robbe-Prinzip (API-Synchronität)

**Seit Alpha 41 gilt:**  
**Header und Source werden immer synchron gepflegt. Kein Drift, kein API-Bruch. Die Robbe wacht.**

> „API-Änderung ohne Header-Update? Dann OOU-OOU und Build-Fehler!“

---

## 🦝 Waschbär-Prinzip (Auto-Fix & Hygiene)

**Ab Alpha 53:**  
Der Build prüft automatisch auf bekannte Toolchain-Fallen.  
Wenn z. B. `glew32d.lib` referenziert wird, wird der Eintrag gelöscht,  
der Cache invalidiert und der Build neu aufgesetzt – ganz ohne Nutzerinteraktion.

> „Sieht unscheinbar aus, aber hat alles im Griff.“ – 🦝

---

## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz – siehe [LICENSE](LICENSE) für Details.

---

**OtterDream** – von der Raupe zum Fraktal-Schmetterling 🦋  
*Happy Zooming!*

🐭 This project owes a mouse more than it admits.  
🦊 With Schneefuchs’ sharp eyes and Otter’s relentless zoom.  
🦭 The Robbe ensures API dignity.  
🦝 And Waschbär… keeps things **really clean**.
