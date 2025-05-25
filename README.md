# OtterDream Mandelbrot Renderer (CUDA + Double)

Dieser Renderer nutzt CUDA und `double`-Präzision für schnelles, interaktives Mandelbrot-Rendering mit Auto-Zoom. OpenGL und ImGui sorgen für eine moderne Darstellung auf Windows.

## Voraussetzungen

- Windows 10/11
- NVIDIA CUDA Toolkit (v12.8 empfohlen)
- Visual Studio Build Tools (2022)
- [vcpkg](https://github.com/microsoft/vcpkg) installiert

## Abhängigkeiten

- GLFW
- GLEW
- ImGui
- Boost (nur `multiprecision` für CPU-Zoomsteuerung)

Alle Bibliotheken werden über `vcpkg` eingebunden.

## Build mit CMake

```bash
cmake --preset=windows-msvc
cmake --build --preset=build
```

## Dateien

- `main.cpp`: GUI, Panning, Zoom, CUDA-Interop
- `mandelbrot.cu`: CUDA-Renderer mit Supersampling & Distance Estimation (double)
- `gui.cpp/hpp`: HUD via ImGui
- `CMakeLists.txt` + `CMakePresets.json`: Build-Setup
- `README.MAUS`: Interne KI-Dokumentation, nicht für Menschen bestimmt

## Besonderheiten

- **Auto-Zoom & Auto-Pan** basierend auf Gradientendichte
- **2×2 Supersampling** für glatte Kanten
- **Distance Estimation** für schöne Farbverläufe
- **Boost BigFloat** nur für CPU-Koordinaten, nicht im CUDA-Kern
- **Sanfte Farbgebung** basierend auf Sinusverlauf

## Bekannte Einschränkungen

- Kein echtes Double-Double mehr – zugunsten von Geschwindigkeit
- Kein Farbschema-Wechsel über HUD
- Kein Multithreading auf der CPU (GPU parallelisiert)
- Kein Dragging per Maus (noch nicht reaktiviert)

## Autor

OtterDream & ChatGPT (2025)

---

Fragen oder Ideen? Einfach den Otter fragen. 🦦


**Hinweis:** Das Programm stürzte ab, weil irgendwann durch null geteilt wurde – ohne Sicherung.