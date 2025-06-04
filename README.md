# 🦦 OtterDream Mandelbrot Renderer (CUDA + Double)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Dieser Renderer nutzt CUDA und `double`-Präzision für schnelles, interaktives Mandelbrot-Rendering mit Auto-Zoom. OpenGL und ImGui sorgen für eine moderne Darstellung auf Windows.

## Voraussetzungen

- Windows 10/11
- NVIDIA CUDA Toolkit (v12.9 empfohlen) — **nvcc** muss im `PATH` verfügbar sein.
- Visual Studio Build Tools (2022) mit **C++ Desktop Development**
- [vcpkg](https://github.com/microsoft/vcpkg) installiert

## Abhängigkeiten

- GLFW
- GLEW
- ImGui
- Boost (`multiprecision` für CPU-Zoomsteuerung)

Alle Bibliotheken werden über `vcpkg` eingebunden.

## Erforderliche Umgebungsvariablen

Folgende Umgebungsvariablen müssen gesetzt sein:

- `CUDA_PATH` — Verzeichnis der installierten CUDA-Toolchain, z.B. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`
- `VCPKG_ROOT` — Verzeichnis der lokalen vcpkg-Installation, z.B. `C:\vcpkg`

> Unter Windows kannst du die Variablen unter **Systemsteuerung → System → Erweiterte Systemeinstellungen → Umgebungsvariablen** setzen.

## Build mit CMake

```bash
cmake --preset=windows-msvc
cmake --build --preset=build
```

## Dateien

- `main.cpp`: GUI, Panning, Zoom, CUDA-Interop
- `mandelbrot.cu`: CUDA-Renderer mit Supersampling & Distance Estimation (`double`)
- `gui.cpp/hpp`: HUD via ImGui
- `CMakeLists.txt` + `CMakePresets.json`: Build-Setup
- `README.MAUS`: Interne KI-Dokumentation, nicht für Menschen bestimmt

## Besonderheiten

- **Auto-Zoom & Auto-Pan** basierend auf Gradientendichte
- **2×2 Supersampling** für glatte Kanten
- **Distance Estimation** für schönere Farbverläufe
- **Boost BigFloat** für CPU-Koordinaten (nicht im CUDA-Kern)
- **Sanfte Farbgebung** basierend auf Sinusverlauf

## Bekannte Einschränkungen

- Kein echtes Double-Double (zugunsten von Geschwindigkeit)
- Kein Farbschema-Wechsel über das HUD
- Kein Multithreading auf der CPU (GPU parallelisiert)
- Dragging per Maus (noch deaktiviert)

## Lizenz

Dieses Projekt steht unter der [MIT License](LICENSE).

## Autor

OtterDream & ChatGPT (2025)

---

**Hinweis:** Das Programm stürzte früher ab, weil bei hohen Zoom-Faktoren durch Null geteilt wurde – **jetzt mit Sicherung**. 🦦