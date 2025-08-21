<!-- Datei: README.md -->
<!-- 🐭 Maus-Kommentar: README für Alpha 81 – CI-validiert, Silk-Lite Zoom integriert, Auto-Tuner statt JSON-Reload, Heatmap-Shader in Arbeit. Schneefuchs sagt: „Nur was synchron ist, bleibt stabil.“ -->

# 🦦 OtterDream Mandelbrot Renderer (CUDA + OpenGL)

[![Build Status](https://github.com/Funcry241/otterdream-mandelbrot/actions/workflows/ci.yml/badge.svg)](https://github.com/Funcry241/otterdream-mandelbrot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ein ultraschneller Mandelbrot-Renderer mit CUDA-Beschleunigung und OpenGL-Anzeige, entwickelt für moderne NVIDIA-GPUs.
Der Renderer zoomt automatisch in interessante Regionen hinein und erhöht dabei fortlaufend die Detailtiefe.
Seit **Alpha 81**: CI-validiert, deterministisch und mit sanftem „Silk-Lite“-Zoom.

---

## 🧠 Features

* **🚀 CUDA Rendering**
  2‑Pass Mandelbrot-Pipeline (Warmup + Sliced Finish), warp-synchron, CHUNKed (`WARP_CHUNK=64`).

  * **Survivor‑Black**: Noch nicht fertig berechnete Pixel werden sofort geschwärzt → *kein Ghosting* zwischen Slices.
  * **Event‑Timing & L1‑Cache Pref**: Eventbasierte Messung (ohne `cudaDeviceSynchronize`) & `cudaFuncSetCacheConfig(..., PreferL1)`.

* **🎯 Auto‑Zoom mit Entropie- und Kontrastanalyse**
  Softmax-Schwerpunkt über **Median/MAD**‑normalisierte Entropie/Kontrast‑Scores; Softmax‑Sparsification für ruhige Ziele.

* **🪶 Silk‑Lite Motion Planner**
  Sanfte Schwenks, **Yaw‑Rate‑Limiter (rad/s)** + Längendämpfung, relative Hysterese & kurzer Lock gegen Flip‑Flop.

* **🕳️ Anti‑Black‑Guard (Cardioid/Bulb‑Avoidance)**
  Warm‑up‑Drift und **Void‑Bias** schieben den Fokus verlässlich aus Innenbereichen → *kein „Zoom ins Schwarze“*.

* **⏱️ Frame‑Budget‑Pacing**
  Der Mandelbrot‑Pfad nutzt nur einen Anteil des Frame‑Budgets (Default **62%**). Budget via `capFramerate/capTargetFps` ableitbar.

* **📈 Progressive Iterationen (Zoom‑abhängig)**
  Iterationszahl steigt automatisch mit dem Zoom‑Level.

* **🎨 Rüsselwarze‑Farbmodus**
  Innen dunkel, außen strukturierte Chaoswellen (Smooth Coloring mit Streifen‑Shading).

* **🔍 Adaptive Tile‑Größe**
  Automatische Tile‑Anpassung für bessere Detailauswertung bei starkem Zoom.

* **🖼️ Echtzeit‑OpenGL + CUDA‑Interop**
  Anzeige über Fullscreen‑Quad, direkte PBO‑Verbindung via `cudaGraphicsGLRegisterBuffer`.

* **📊 Heatmap‑Overlay (Projekt Eule)**
  Visualisierung von Entropie/Kontrast pro Tile, aktuell CPU‑basiert. GPU‑Shader (Glow/Transparenz) in Arbeit.

* **🧰 HUD & ASCII‑Debug (Warzenschwein)**
  FPS, Zoom, Offset – optional. **Logging ist ASCII‑only und ohne Seiteneffekte** (keine funktionale Beeinflussung der Pfade).

* **🦝 Build‑Fallback‑Logik (Waschbär)**
  Automatische Bereinigung typischer Toolchain‑Fallen (z. B. `glew32d.lib`).

* **🤖 Auto‑Tuner**
  Findet ohne Neustart zyklisch optimale Zoom-/Analyseparameter und schreibt sie ins Log (kein JSON‑Reload nötig).

---

## 🆕 Neu in dieser Version (Alpha 81+)

* **Sliced Survivor Finish** mit **Survivor‑Black** (ghosting‑frei)
* **Frame‑Budget‑Pacing** mit eventbasiertem Timing (kostenarme Budgetkontrolle)
* **Anti‑Black‑Guard** (Warm‑up‑Drift + Void‑Bias gegen Cardioid/Bulb‑Hänger)
* **Yaw‑Limiter** (rad/s → rad/Frame per `dt`) + **Längendämpfung** bei großen Drehwinkeln
* **Hysterese/Lock & dyn. Retarget‑Throttle** für ruhiges Zielhalten
* **Softmax‑Sparsification** und robuste **Median/MAD**‑Statistik (konsistente Scores)

---

## 🖥️ Systemvoraussetzungen

* Windows 10 oder 11 **oder** Linux
* **NVIDIA GPU** mit CUDA (Compute Capability **8.0+**, empfohlen **8.6+**)
* CUDA Toolkit (empfohlen: **v12.9**)
* Visual Studio 2022 **oder** GCC 11+
* CMake (Version **≥3.28**), Ninja
* vcpkg (für GLFW, GLEW)

> ⚠️ GPUs unter Compute Capability 8.0 (z. B. Kepler, Maxwell) werden **nicht unterstützt**.

---

## 📦 Abhängigkeiten (via vcpkg)

* [GLFW](https://www.glfw.org/) – Fenster- und Eingabe‑Handling
* [GLEW](http://glew.sourceforge.net/) – OpenGL‑Extension‑Management

---

## 🔧 Build‑Anleitung

### 📁 vcpkg Setup

```bash
git clone --recurse-submodules https://github.com/Funcry241/otterdream-mandelbrot.git
cd otterdream-mandelbrot
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh   # oder .bat unter Windows
cd ..
```

---

## Plattformkompatibilität

* Zeitformatierung plattformunabhängig via `getLocalTime(...)`
* MSVC‑spezifische `#pragma warning(...)` sind in `#ifdef _MSC_VER` gekapselt

### 🪟 Windows Build

```powershell
.build.ps1
```

> 🧼 Das Build‑Skript erkennt und behebt automatisch bekannte Fallstricke:
>
> * `glew32d.lib`‑Bug (vcpkg‑Falle)
> * inkonsistente CMake‑Caches
> * fehlende CUDA‑Pfade
>
> Kein zweiter Durchlauf nötig – dank 🐭‑Patchlogik und 🦝 Waschbär‑Watchdog.

---

### 🐧 Linux Build

> Voraussetzung: CUDA 12.9, GCC, Ninja, CMake ≥3.28, OpenGL‑Treiber, vcpkg

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build libglfw3-dev libglew-dev libxmu-dev libxi-dev libglu1-mesa-dev xorg-dev pkg-config libcuda1-525
```

> *Hinweis:* Je nach Distribution kann die CUDA‑Runtime‑Bibliothek anders heißen (z. B. `libcuda1-545`)

```bash
cmake --preset linux-build
cmake --build --preset linux-build
cmake --install build/linux --prefix ./dist
./dist/mandelbrot_otterdream
```

---

### ⌨️ Keyboard Controls

* `P` oder `Space`: Auto‑Zoom pausieren/fortsetzen
* `H`: Heatmap‑Overlay ein-/ausschalten

---

### ⚙️ Customizing CUDA Architectures

Standardmäßig werden Compute‑Capabilities **80;86;89;90** gebaut.

Wenn Ihre GPU eine andere Capability hat, überschreiben Sie sie so:

```bash
cmake --preset windows-release -DCMAKE_CUDA_ARCHITECTURES=90
```

Ihre Capability finden Sie in NVIDIAs Übersicht.

---

## 🌊 Das Robbe‑Prinzip (API‑Synchronität)

**Seit Alpha 41 gilt:**
Header und Source werden **immer synchron** gepflegt. Kein Drift, kein API‑Bruch.
Die Robbe wacht.

> „API‑Änderung ohne Header‑Update? Dann OOU‑OOU und Build‑Fehler!“

---

## 🦝 Waschbär‑Prinzip (Auto‑Fix & Hygiene)

**Ab Alpha 53:**
Der Build prüft automatisch auf bekannte Toolchain‑Fallen.
Wenn z. B. `glew32d.lib` referenziert wird, wird der Eintrag gelöscht,
der Cache invalidiert und der Build neu aufgesetzt – ohne Nutzerinteraktion.

---

## 🔭 Zoomgerichtet & geschmacksgetestet

**Seit Alpha 81 (Silk‑Lite Zoom):**
Das Zoomziel wird per Softmax‑Ranking, Entropie-/Kontrastanalyse und Motion‑Planner bestimmt.
Yaw‑Limiter, Mikro‑Deadband und Acc-/Vel‑Clamp verhindern Ruckler & Flip‑Flops.
Optional sorgt der **Auto‑Tuner** dafür, dass die Parameter im laufenden Betrieb feingeschliffen werden.

> Ergebnis: Immer der spannendste Bildausschnitt, nie das Gefühl von „lost in fractal space“.

---

## 🔎 Qualitäts‑Guards (Kurzüberblick)

* **Anti‑Black‑Guard**: Warm‑up‑Drift & Void‑Bias – kein „Zoom ins Schwarze“
* **Survivor‑Black**: Ghosting‑freie Slices
* **Hysterese/Lock**: verhindert Ziel‑Flip‑Flops
* **Retarget‑Throttle**: CPU‑schonend, ruhiger Kurs
* **Softmax‑Sparsification**: ignoriert irrelevante Tails

---

## ⚙️ Konfigurationshinweise

* **Logging**: ASCII‑only; *keine* Seiteneffekte auf Berechnungs‑ oder Render‑Pfade.
  Aktivieren Sie `debugLogging` nur für Diagnosen; `performanceLogging` misst budgetschonend via Events.
* **Framerate‑Cap**: `capFramerate` + `capTargetFps` steuern das Frame‑Budget; der Mandelbrot‑Pfad nutzt davon standardmäßig \~62%.
* **ForceAlwaysZoom**: hält den Zoomfluss aktiv (mit weicher Drift, falls kein starkes Signal vorliegt).

---

## 📄 Lizenz

Dieses Projekt steht unter der MIT‑Lizenz – siehe [LICENSE](LICENSE) für Details.

---

**OtterDream** – von der Raupe zum Fraktal‑Schmetterling 🦋
*Happy Zooming!*

🐭 Maus sorgt für Fokus und ASCII‑Sauberkeit.
🦊 Schneefuchs bewacht die Präzision.
🦦 Otter treibt den Zoom unaufhaltsam.
🦭 Robbe schützt die API‑Würde.
🦝 Waschbär hält den Build hygienisch.
🦉 Eule sorgt für Überblick in Heatmap & Koordinaten.
