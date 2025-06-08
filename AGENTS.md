# 👩‍💻 OtterDream Build Agents

Diese Datei dokumentiert die automatisierten Prozesse und verwendeten Tools für den Build und die Pflege des OtterDream Mandelbrot-Renderers.

---

## Overview

Das Projekt nutzt folgende Agents und Tools:

| Agent                | Zweck                            | Trigger            | Aktionen                         |
|----------------------|----------------------------------|--------------------|----------------------------------|
| GitHub Actions (CI)   | Build-Überprüfung bei Push/PR    | Push auf `main`    | CMake Konfiguration, Ninja Build |
| Dependabot           | Abhängigkeits-Updates für vcpkg  | Wöchentlich        | Überwachung von `vcpkg.json`     |

---

## Tools und Versionen

| Tool              | Mindestversion  | Hinweise                                 |
|-------------------|-----------------|------------------------------------------|
| CUDA Toolkit      | 12.0+            | Erforderlich für GPU-Rendering           |
| OpenGL            | 4.3+             | Core Profile benötigt                   |
| Visual Studio     | 2022             | Mit CUDA- und C++-Support                |
| CMake             | >4.0             | Mindestversion laut Build Requirements   |
| Ninja             | 1.10+            | Für schnellen Build-Prozess              |
| vcpkg             | aktuell          | Abhängigkeitsmanagement für Libraries    |

---

## Lokaler Build

### Windows

```bash
cmake --preset windows-msvc
cmake --build --preset windows-msvc
```

### Linux

> **Voraussetzung:** Installiere CUDA Toolkit, CMake **>4.0**, Ninja, OpenGL-Treiber, GLFW und GLEW.

```bash
cmake --preset linux-gcc
cmake --build --preset linux-gcc
```

Ergebnisse befinden sich im `./dist` Verzeichnis.

---

## Hinweise

- **CI Builds** laufen bei jedem Push auf den `main` Branch.
- **Abhängigkeitsüberwachung** erfolgt automatisch via Dependabot.
- **Build-Presets** sind vordefiniert in `CMakePresets.json`.
- **Keine** Dynamic Parallelism Anforderungen: Funktioniert mit GPUs ab Compute Capability 3.0.

---

Happy Building! 🦦🚀
