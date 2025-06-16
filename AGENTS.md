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
| CUDA Toolkit      | 12.0+           | Erforderlich für GPU-Rendering            |
| OpenGL            | 4.3+            | Benötigt Core Profile                     |
| Visual Studio     | 2022            | Inklusive C++- und CUDA-Support           |
| CMake             | ≥3.25           | Empfehlung: aktuelle stabile Version      |
| Ninja             | 1.10+           | Für schnelle parallele Builds             |
| vcpkg             | aktuell         | Verwaltung von Drittanbieter-Bibliotheken |

---

## ⚙️ Lokaler Build

### 🪟 Windows

```bash
cmake --preset windows-msvc
cmake --build --preset windows-msvc
```

### 🐧 Linux

1. **Voraussetzungen installieren** (einmalig):
```bash
sudo apt update
sudo apt install build-essential cmake git libglfw3-dev libglew-dev libcuda1-525 nvidia-cuda-toolkit
```

2. **Repository klonen & vcpkg initialisieren**:
```bash
git clone https://github.com/dein-username/otterdream-mandelbrot.git
cd otterdream-mandelbrot
./vcpkg/bootstrap-vcpkg.sh
```

3. **Projekt konfigurieren & bauen**:
```bash
cmake -B build/linux -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/linux --parallel
```

4. **Starten**:
```bash
cd build/linux
./mandelbrot_otterdream
```

---

## 🌐 CI/CD Pipelines

- **GitHub Actions**: 
  - `.github/workflows/ci.yml` führt bei jedem Push auf `main` einen vollständigen Build und Tests durch.
- **Dependabot**:
  - Automatisches Update der vcpkg-Abhängigkeiten wöchentlich.

