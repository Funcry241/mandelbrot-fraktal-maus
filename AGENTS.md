# 👩‍💻 OtterDream Build Agents

Diese Datei dokumentiert die automatisierten Prozesse und Tools für den Build und die Pflege des OtterDream Mandelbrot-Renderers.

---

## 🧑‍🔬 Overview

Das Projekt verwendet folgende Agents und Werkzeuge:

| Agent               | Zweck                           | Trigger         | Aktionen                                   |
| ------------------- | ------------------------------- | --------------- | ------------------------------------------ |
| GitHub Actions (CI) | Build- & Install-Check bei Push | Push auf `main` | CMake-Konfiguration, Ninja-Build, `--install` |
| Dependabot          | Abhängigkeits-Updates für vcpkg | Wöchentlich     | Überwachung von `vcpkg.json`               |

---

## 🧰 Tools & Versionen

| Tool          | Mindestversion | Hinweise                                  |
| ------------- | -------------- | ----------------------------------------- |
| CUDA Toolkit  | 12.9+          | Erforderlich für GPU-Rendering            |
| OpenGL        | 4.3+           | Benötigt Core Profile                     |
| Visual Studio | 2022           | Inklusive C++- und CUDA-Support           |
| CMake         | ≥3.28          | Install-Ziel via `--install`              |
| Ninja         | 1.10+          | Für schnelle parallele Builds             |
| vcpkg         | aktuell        | Verwaltung von Drittanbieter-Bibliotheken |

---

## ⚙️ Lokaler Build

### 🪟 Windows

```bash
cmake --preset windows-msvc
cmake --build --preset windows-msvc
cmake --install build/windows --prefix ./dist
```

### 🐗 Linux

1. **Voraussetzungen installieren** (einmalig):

```bash
sudo apt update
sudo apt install build-essential cmake git libglfw3-dev libglew-dev libcuda1-525
```

2. **Repository klonen & vcpkg initialisieren**:

```bash
git clone https://github.com/Funcry241/otterdream-mandelbrot.git
cd otterdream-mandelbrot
./vcpkg/bootstrap-vcpkg.sh
```

3. **Projekt konfigurieren & bauen**:

```bash
cmake --preset linux-build
cmake --build --preset linux-build
cmake --install build/linux --prefix ./dist
```

4. **Starten**:

```bash
./dist/bin/mandelbrot_otterdream
```

---

## 🌐 CI/CD Pipelines

* **GitHub Actions**:

  * `.github/workflows/ci.yml` führt bei jedem Push auf `main` einen vollständigen Build und anschließenden `cmake --install` aus.
* **Dependabot**:

  * Automatisches Update der vcpkg-Abhängigkeiten wöchentlich.
