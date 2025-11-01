<!-- Datei: AGENTS.md -->
<!-- 🐭 Maus-Kommentar: Dokumentiert Buildprozesse und Toolchains für OtterDream. Aktualisiert: Single-Path-Render (Capybara), kein EC/Wrapper, core_kernel entfernt, LUCHS_LOG-Trennung, kompakte PERF-Logs (Epoch-Millis). Schneefuchs flüstert: „Ein Agent kennt die versteckten Knöpfe und sorgt für saubere Übergänge.“ -->

# 👩‍💻 OtterDream Build Agents

Diese Datei beschreibt die automatisierten Prozesse, lokalen Helfer und Regeln rund um Build, Logging und Pflege des **OtterDream Mandelbrot-Renderers**.

**Seit Alpha 41** gilt das **Robbe-Prinzip**: *Header & Source bleiben synchron. Kein Drift, kein API-Bruch.*  
**Seit Alpha 81** zusätzlich relevant: *Silk-Lite Zoom*, *Frame-Budget-Pacing* und **ASCII-only Logs** ohne Seiteneffekt.  
**Neu (Renderer-Pfad)**: Einziger aktiver Pfad ist **Capybara → Iterationen → Colorizer → PBO** (kein Referenz-Orbit, **keine Perturbation**, **kein EC/Wrapper**).

---

## 🧑‍🔬 Overview

| Agent/Tool              | Zweck                         | Trigger            | Aktionen                                                |
| ----------------------- | ----------------------------- | ------------------ | ------------------------------------------------------- |
| **GitHub Actions (CI)** | Build-, Test-, Install-Check  | Push/PR auf `main` | CMake Configure → Ninja Build → `cmake --install`       |
| **Dependabot**          | Abhängigkeits-Updates (vcpkg) | Wöchentlich        | PRs für `vcpkg.json`, CI baut PR                        |
| **Waschbär-Watchdog**   | Hygiene & Auto-Fixes (lokal)  | On-Demand          | Räumt CMake-Caches, fixt typische GLEW/vcpkg-Fallen     |

> CI stellt sicher, dass **Debug-/Perf-Logging keine Seiteneffekte** erzeugt (keine erzwungenen Synchronisationen im Hot-Path).

---

## 🧰 Tools & Versionen

| Tool          | Mindestversion | Hinweise                         |
| ------------- | -------------- | -------------------------------- |
| CUDA Toolkit  | **13.0+**      | `nvcc` v13 lokal erforderlich    |
| OpenGL        | 4.3+           | Core Profile                     |
| Visual Studio | 2022           | C++ + CUDA                       |
| CMake         | ≥3.28          | Presets & `--install`            |
| Ninja         | 1.10+          | Schneller Parallel-Build         |
| vcpkg         | aktuell        | Drittanbieter-Libs               |

### ⚠️ CUDA erforderlich

Ohne lokal installiertes **CUDA 13** (inkl. `nvcc`) startet der Build nicht.

---

## 🧠 CUDA-Architekturen

Standard: `80;86;89;90` (Ampere+). Abweichungen pro Preset überschreiben:

```bash
cmake --preset windows-release -DCMAKE_CUDA_ARCHITECTURES=90
```

Die passende Compute Capability deiner GPU findest du in NVIDIAs Übersicht.

---

## 🧯 Host/Device-Logging (LUCHS_LOG)

* **Host**: `LUCHS_LOG_HOST(...)` - ASCII-only, **eine Zeile pro Event**, Zeitstempel als **Epoch-Millis**.
* **Device**: `LUCHS_LOG_DEVICE(msg)` - schreibt in den Device-Puffer; Flush auf Host synchronisiert **außerhalb** des Hot-Paths.  
  *Hinweis:* Nachricht mit `snprintf` zusammenbauen ist ok - der **finale** Aufruf ist genau **ein** `LUCHS_LOG_DEVICE(const char*)`.
* **Kein `printf/fprintf`** im Produktionspfad. Logs dürfen **keine** impliziten Synchronisationen auslösen.
* **Zwei Schalter** (`Settings`):  
  `performanceLogging` → kompakte Messwerte via CUDA-Events; **ASCII**  
  `debugLogging` → detaillierter, ggf. langsamer

---

## ⏱️ Frame-Budget-Pacing (Silk-Lite kompatibel)

Der Mandelbrot-Pfad hält sich an ein weiches Zeitbudget pro Frame. Silk-Lite steuert Bewegung (Yaw-Limiter + Dämpfung).  
**Regel**: Pacing misst mit CUDA-Events (kostenarm) und **erzwingt keine** globale Synchronisation.

---

## 🎨 Renderer-Pfad & Farbgebung (Status)

* **Aktiver Pfad**: **Capybara-Iteration** (Float), Escape-Test **vor** dem Update (`|z|^2 > 4`).  
  → *Innen* schreibt `iterOut = maxIter`, *Escape* schreibt den Iterationsindex.  
* **Pipeline**: `capy_render(...)` (Iterations) → `colorize_iterations_to_pbo(...)` → PBO (GL-Interop).  
* **Palette**: **GT (Cyan→Amber)** mit Interpolation im **Linearraum** gegen Banding.  
  **Stripes** sind **standardmäßig aus** (`stripes = 0.0f`) für ringfreie Darstellung.  
* **Mapping**: Projektweit über `screenToComplex(...)` (Koordinaten-Harmonisierung, „Eule“).

> Hinweis: Entropie/Kontrast (EC) ist **deaktiviert/entfernt**. Heatmap-Overlay existiert optional, liefert derzeit **kein** Signal.

---

## ⌨️ Hotkeys (Runtime)

| Taste   | Funktion                       |
| ------- | ------------------------------ |
| `P`     | Auto-Zoom pausieren/fortsetzen |
| `H`     | Heatmap-Overlay toggeln (derzeit ohne EC-Daten) |
| `T`     | HUD (Warzenschwein) toggeln    |

> Hinweis: `Space` ist derzeit **nicht** gemappt (kein Alias zu `P`).

---

## ⚙️ Lokaler Build

### 🪟 Windows

**A) Komfort (optional, falls vorhanden)**

```powershell
./build.ps1
```

**B) Manuell mit Presets**

```powershell
cmake --preset windows-msvc
cmake --build --preset windows-msvc
cmake --install --preset windows-msvc --prefix .\dist
.\dist\mandelbrot_otterdream.exe
```

> Hinweis: In manchen Repos ist `build.ps1` absichtlich **nicht** eingecheckt. Dann bitte Weg **B)** verwenden.

### 🐧 Linux

1. Pakete (Beispiel Debian/Ubuntu):

```bash
sudo apt update
sudo apt install build-essential cmake git ninja-build libglfw3-dev libglew-dev libxmu-dev libxi-dev libglu1-mesa-dev xorg-dev pkg-config
```

2. Klonen & vcpkg bootstrap (wie in der README):

```bash
git clone --recurse-submodules https://github.com/Funcry241/mandelbrot-fraktal-maus.git
cd mandelbrot-fraktal-maus
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh
```

3. Bauen & installieren:

```bash
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE="$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake --install build --prefix ./dist
./dist/mandelbrot_otterdream
```

---

## 🧷 Toolchain & Hardening (Windows)

* **CRT vereinheitlicht**: `/MT` (inkl. NVCC-Host) → keine LNK2038-Mismatches.
* **`CUDA::cudart_static`**: passt zum `/MT`-CRT.
* **GLEW dynamisch**: **kein** `GLEW_STATIC`; vcpkg-Triplet passend wählen.
* **Hardening nur im Host-Link**: `/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf` über `$<HOST_LINK:...>`.
* **Separable Compilation** + **Device-Symbols** aktiviert (CMake Properties).

---

## 🌊 Robbe-Prinzip (API-Synchronität)

> Jede Änderung an Signaturen/Interfaces wird **zeitgleich** in Header **und** Source umgesetzt (und gemeinsam committed). Abweichungen sind Build-Fehler – Robbe sagt **OOU-OOU**.

* Kein schleichender Drift  
* Saubere öffentliche API  
* **Referenz:** `src/capybara_frame_pipeline.cuh` (Signatur **`capy_render(...)`**) und `src/cuda_interop.hpp` (Signaturen **`renderCudaFrame(...)`**).

---

## 🧪 Logging-Formate (Kern, kompakt)

* **ASCII-only**, **eine Zeile pro Logeintrag**.
* **Epoch-Millis** (UTC) als Zeitstempel.
* **Keine Seiteneffekte** im Hot-Path (keine globale Sync).

### Kompakte PERF-Zeile (Render/Colorize)

```
[<epoch-ms>][cuda_interop.cu][line]: [PERF] capy=<ms> col=<ms> b=<budget-ms> ema=<x.xxx> bh=<0/1>
```

**Legende (Kurz):** `capy` Render via Capybara, `col` Colorizer, `b` Budget, `ema` geglättete Budget-Auslastung, `bh` Budget-Hit (1 = Budget erschöpft).

> EC-spezifische Metriken sind derzeit **entfernt**.

---

## 🌐 CI/CD Pipelines

**GitHub Actions** (`.github/workflows/ci.yml`)

* Configure → Build (Ninja) → Install
* Artefakte: Install-Tree unter `dist/`
* Prüfungen:
  * CUDA-Kompilation für Presets
  * konsistente CMake-Presets
  * deterministische Builds (gleiche Inputs → gleiche Outputs)

**Dependabot**

* PRs für `vcpkg.json` (wöchentlich)
* CI baut und verifiziert

---

## ❓ Troubleshooting (Kurz)

* **`nvcc` fehlt** → **CUDA 13** installieren, PATH/INCLUDE/LIB prüfen
* **GLEW-Mismatch (z. B. `glew32d.lib`)** → auf **dynamisches GLEW** wechseln und Triplet/Cache prüfen
* **Schwarze Frames** bei extremem Pan/Zoom → Silk-Lite/Anti-Black-Guard aktiv lassen; Messläufe ohne Debug-Logs
* **CUDA-Interop Stalls** → PBO-Ring (≥3), `WriteDiscard`, persistentes Mapping, Fences

---

**Agenten-Motto:** Maus bringt Fokus 🐭, Schneefuchs Präzision 🦊, Robbe API-Disziplin 🦭, Waschbär Hygiene 🦝. 💫
