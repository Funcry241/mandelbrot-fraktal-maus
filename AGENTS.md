# 👩‍💻 OtterDream Build Agents

Diese Datei dokumentiert die automatisierten Prozesse und Tools für den Build und die Pflege des OtterDream Mandelbrot-Renderers.

---

## 🧭 Overview

Das Projekt verwendet folgende Agents und Werkzeuge:

| Agent                | Zweck                             | Trigger             | Aktionen                          |
|----------------------|-----------------------------------|---------------------|-----------------------------------|
| GitHub Actions (CI)  | Build-Überprüfung bei Push/PR     | Push auf `main`     | CMake-Konfiguration, Ninja-Build  |
| Dependabot           | Abhängigkeits-Updates für vcpkg   | Wöchentlich         | Überwachung von `vcpkg.json`      |

---

## 🧰 Tools & Versionen

| Tool              | Mindestversion  | Hinweise                                  |
|-------------------|-----------------|-------------------------------------------|
| CUDA Toolkit      | 12.0+            | Erforderlich für GPU-Rendering            |
| OpenGL            | 4.3+             | Benötigt Core Profile                     |
| Visual Studio     | 2022             | Inklusive C++- und CUDA-Support           |
| CMake             | ≥3.25            | Empfehlung: aktuelle stabile Version      |
| Ninja             | 1.10+            | Für schnelle parallele Builds             |
| vcpkg             | aktuell          | Verwaltung von Drittanbieter-Bibliotheken |

---

## ⚙️ Lokaler Build

### 🪟 Windows

```bash
cmake --preset windows-msvc
cmake --build --preset windows-msvc
