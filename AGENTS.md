<!-- Datei: AGENTS.md -->

<!-- 🐭 Maus-Kommentar: Dokumentiert Buildprozesse und Toolchains für OtterDream. Jetzt mit Hotkey-Doku, CUDA-Architektur-Hinweis, Frame‑Budget‑Pacing‑Hinweis, Robbe‑Regel und kompakten PERF‑Logs (Epoch‑Millis). Schneefuchs flüstert: „Ein Agent kennt die versteckten Knöpfe und sorgt für saubere Übergänge.“ -->

# 👩‍💻 OtterDream Build Agents

Diese Datei beschreibt die automatisierten Prozesse, lokalen Helfer und Regeln rund um Build, Logging und Pflege des **OtterDream Mandelbrot‑Renderers**.

**Seit Alpha 41** gilt das **Robbe‑Prinzip**: *Header & Source bleiben synchron. Kein Drift, kein API‑Bruch.*
**Seit Alpha 81** zusätzlich relevant: *Silk‑Lite Zoom*, *Frame‑Budget‑Pacing* und **ASCII‑only Logs** ohne Seiteneffekt.

---

## 🧑‍🔬 Overview

| Agent/Tool              | Zweck                         | Trigger         | Aktionen                                            |
| ----------------------- | ----------------------------- | --------------- | --------------------------------------------------- |
| **GitHub Actions (CI)** | Build-, Test-, Install‑Check  | Push auf `main` | CMake Configure → Ninja Build → `cmake --install`   |
| **Dependabot**          | Abhängigkeits‑Updates (vcpkg) | Wöchentlich     | PRs für `vcpkg.json`, CI baut PR                    |
| **Waschbär‑Watchdog**   | Hygiene & Auto‑Fixes (lokal)  | On‑Demand       | Räumt CMake‑Caches, fixt typische GLEW/vcpkg‑Fallen |

> CI stellt sicher, dass **Debug-/Perf‑Logging keine Seiteneffekte** erzeugt (keine erzwungenen Synchronisationen im Hot‑Path).

---

## 🧰 Tools & Versionen

| Tool          | Mindestversion | Hinweise                  |
| ------------- | -------------- | ------------------------- |
| CUDA Toolkit  | 12.9+          | `nvcc` lokal erforderlich |
| OpenGL        | 4.3+           | Core Profile              |
| Visual Studio | 2022           | C++ + CUDA                |
| CMake         | ≥3.28          | Presets & `--install`     |
| Ninja         | 1.10+          | Schneller Parallel‑Build  |
| vcpkg         | aktuell        | Drittanbieter‑Libs        |

### ⚠️ CUDA erforderlich

Ohne lokal installiertes CUDA (inkl. `nvcc`) startet der Build nicht.

---

## ⌨️ Hotkeys (Runtime)

| Taste   | Funktion                       |
| ------- | ------------------------------ |
| `P`     | Auto‑Zoom pausieren/fortsetzen |
| `Space` | Alternativ zu `P`              |
| `H`     | Heatmap‑Overlay toggeln        |
| `T`     | HUD (Warzenschwein) toggeln    |

> **Silk‑Lite** sorgt für sanfte Richtungswechsel (Yaw‑Limiter + Dämpfung), unabhängig vom Logging.

---

## 🧠 CUDA‑Architekturen

Standard: `80;86;89;90`. Abweichungen pro Preset überschreiben:

```bash
cmake --preset windows-release -DCMAKE_CUDA_ARCHITECTURES=90
```

Die passende CC deiner GPU findest du in NVIDIAs Übersicht.

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
cmake --install build\windows --prefix .\dist
./dist/mandelbrot_otterdream.exe
```

> Hinweis: In manchen Repos ist `build.ps1` absichtlich **nicht** eingecheckt. Dann bitte Weg **B)** verwenden.

### 🐧 Linux

1. Pakete (Beispiel Debian/Ubuntu):

```bash
sudo apt update
sudo apt install build-essential cmake git ninja-build \
  libglfw3-dev libglew-dev libxmu-dev libxi-dev libglu1-mesa-dev xorg-dev pkg-config libcuda1-525
```

> Je nach Treiber: `libcuda1-545` o. ä.

2. Klonen & vcpkg bootstrap:

```bash
git clone --recurse-submodules https://github.com/Funcry241/otterdream-mandelbrot.git
cd otterdream-mandelbrot
./vcpkg/bootstrap-vcpkg.sh
```

3. Bauen & installieren:

```bash
cmake --preset linux-build
cmake --build --preset linux-build
cmake --install build/linux --prefix ./dist
./dist/mandelbrot_otterdream
```

---

## 🌊 Robbe‑Prinzip (API‑Synchronität)

> Jede Änderung an Signaturen/Interfaces wird **zeitgleich** in Header **und** Source umgesetzt (und gemeinsam committed). Abweichungen sind Build‑Fehler – Robbe sagt **OOU‑OOU**.

* Kein schleichender Drift
* Saubere öffentliche API

---

## 🧪 Logging‑Regeln & Formate (Alpha 81)

* **ASCII‑only**, keine binären Dumps, **eine Zeile pro Logeintrag**.
* **Keine Seiteneffekte**: Logs verändern keinen Zustand und erzwingen keine Synchronisationen im Hot‑Path.
* **Zwei Schalter** (in `Settings`):

  * `performanceLogging` → kompakte Messwerte via CUDA‑Events
  * `debugLogging` → detailliertere Diagnose (zur Not langsamer)

### Zeitstempel

* **Epoch‑Millis** (UTC/Local egal für Parsing) statt Langformat.
* Beispiel‑Prefix: `\[1693243285061][core_kernel.cu][676]: ...`

### Kompakte PERF‑Zeilen (Kern)

Ein Eintrag bündelt das Wesentliche pro Frame:

```
[<epoch-ms>][core_kernel.cu][line]: [PERF] k=<ms> b=<budget-ms> wu=<it> sv=<n>(<%>) sl=<slices> st0=<it0> stN=<itN> stMax=<itMax> ch=<n> rem=<n> ema=<x.xxx> bh=<0/1>
```

**Legende (Kurz):**

* `k` Kernel‑Gesamtzeit (ms), `b` Kernel‑Budget (ms)
* `wu` Warmup‑Iterationen
* `sv` Survivor nach Pass 1 (Anzahl & Anteil)
* `sl` Slices ausgeführt
* `st0` Start‑SliceIt, `stN` letztes SliceIt, `stMax` höchstes SliceIt
* `ch` Anzahl SliceIt‑Anpassungen
* `rem` verbleibende Survivors nach letzter Slice
* `ema` geglättete Drop‑Rate
* `bh` Budget‑Hit (1 = Budget erschöpft)

### Entropie/Kontrast (GPU‑Metriken)

Separat und knapp:

```
[<epoch-ms>][core_kernel.cu][line]: [PERF] en=<ms> ct=<ms>
```

> Tipp: Für Volumen‑Reduktion **Sampling‑Rate** des Perf‑Loggers anheben (z. B. jede n‑te Frame‑Zeile), Debug‑Logs aus.

---

## 🌐 CI/CD Pipelines

**GitHub Actions** (`.github/workflows/ci.yml`)

* Configure → Build (Ninja) → Install
* Artefakte: Install‑Tree unter `dist/`
* Prüfungen:

  * CUDA‑Kompilation für Presets
  * konsistente CMake‑Presets
  * deterministische Builds (gleiche Inputs → gleiche Outputs)

**Dependabot**

* PRs für `vcpkg.json` (wöchentlich)
* CI baut und verifiziert

---

## ❓ Troubleshooting (Kurz)

* **`nvcc` fehlt** → CUDA 12.9 installieren, PATH prüfen
* **`glew32d.lib` verlinkt** → Triplet prüfen; notfalls Build‑Cache löschen (Preset neu)
* **Schwarze Frames** bei extremem Pan/Zoom → Silk‑Lite/Anti‑Black‑Guard aktiv lassen; Messläufe ohne Debug‑Logs

---

**Agenten‑Motto:** Maus bringt Fokus 🐭, Schneefuchs Präzision 🦊, Robbe API‑Disziplin 🦭, Waschbär Hygiene 🦝. 💫
