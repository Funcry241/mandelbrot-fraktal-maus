<!-- Datei: README.md -->
<!-- 🐭 Maus-Kommentar: README für Alpha 81+ – CI-validiert, Silk-Lite Zoom integriert, Capybara Single-Path (keine EC/Wrapper), Logs als Epoch-Millis. CUDA 13 Pflicht; GLEW dynamisch. Schneefuchs: „Nur was synchron ist, bleibt stabil.“ -->

# 🦦 OtterDream Mandelbrot Renderer (CUDA + OpenGL)

[![Build Status](https://github.com/Funcry241/mandelbrot-fraktal-maus/actions/workflows/ci.yml/badge.svg)](https://github.com/Funcry241/mandelbrot-fraktal-maus/actions/workflows/ci.yml)
![CUDA](https://img.shields.io/badge/CUDA-13%2B-76b900?logo=nvidia)
![C%2B%2B](https://img.shields.io/badge/C%2B%2B-20-blue)
![OpenGL](https://img.shields.io/badge/OpenGL-4.3%2B-3D9DD6)
![Platforms](https://img.shields.io/badge/Platforms-Windows%20%7C%20Linux-informational)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="assets/hero_russelwarze.jpg" alt="Otterdream Mandelbrot – Rüsselwarze Mode" width="85%">
</p>

Ein ultraschneller Mandelbrot-Renderer mit CUDA-Beschleunigung und OpenGL-Anzeige für moderne NVIDIA-GPUs. Der Renderer zoomt automatisch in interessante Regionen und erhöht fortlaufend die Detailtiefe.
Seit **Alpha 81**: CI-validiert, deterministisch, sanfter **Silk-Lite**-Zoom — und kompakte **Epoch-Millis**-Logs.

> **Wichtig (Änderung)**: Ab diesem Stand rendert OtterDream über einen **einzigen aktiven Pfad**:  
> **Capybara → Iterationen → Colorizer → PBO**.  
> Es gibt **keinen Referenz-Orbit / keine Perturbation** und **keine EC/Wrapper** im aktiven Code.

---

## 🧠 Features

* **🚀 CUDA Rendering (Capybara)**  
  Iterations-Render über Capybara, ereignisbasiertes **Event-Timing** via CUDA-Events (ohne globales `cudaDeviceSynchronize()` im Normalpfad).  
  * **Survivor-Black**: unfertige Pixel sofort schwarz -> *kein Ghosting* zwischen Slices.  
  * **WARP_CHUNK**-basiertes Pacing (warp-synchron).

* **🪶 Silk-Lite Motion Planner (Auto-Zoom)**  
  Sanfte Schwenks, **Yaw-Rate-Limiter** (rad/s) + Längendämpfung, relative Hysterese & kurzer Lock gegen Flip-Flop.  
  **Hinweis:** Die frühere Entropie/Kontrast-Analyse ist aktuell **deaktiviert**; es wirkt der **ForceAlwaysZoom**-Fallback für stetige Bewegung.

* **🕳️ Anti-Black-Guard (Cardioid/Bulb-Avoidance)**  
  Warm-up-Drift und **Void-Bias** schieben den Fokus verlässlich aus Innenbereichen -> *kein „Zoom ins Schwarze“*.

* **📈 Progressive Iterationen (Zoom-abhängig)**  
  Iterationszahl steigt automatisch mit dem Zoom-Level. **Standardmäßig aktiv** (abschaltbar).

* **🎨 GT-Palette (Cyan→Amber) + Smooth Coloring**  
  Interpolation im **Linearraum** gegen Banding, **Smooth Coloring** via `it - log2(log2(|z|))`.  
  **Streifen-Shading** optional – **standardmäßig aus** (`stripes = 0.0f`) für ringfreie Darstellung.  
  **Mapping-Vertrag:** *Innenpunkte schreiben `iterOut = maxIter`*, Escape schreibt die Iterationsnummer.

* **🖼️ Echtzeit-OpenGL + CUDA-Interop**  
  Anzeige via Fullscreen-Quad, direkte PBO-Verbindung (`cudaGraphicsGLRegisterBuffer`).

* **📊 Heatmap-Overlay (Eule – Preview)**  
  GPU-Shader-Variante im Aufbau; **derzeit ohne EC-Signal**.

* **🧰 HUD & ASCII-Debug (Warzenschwein)**  
  FPS, Zoom, Offset – optional. **Logging ist ASCII-only** und wirkt nicht auf Berechnungs-/Render-Pfade.

* **🤖 Auto-Tuner**  
  Findet ohne Neustart zyklisch sinnvolle Ziel-/Zoom-Parameter und schreibt sie ins Log (kein JSON-Reload nötig).

---

## 🆕 Neu in dieser Version (Alpha 81+)

* **Single-Path Renderer**: Capybara → Colorizer → PBO (klassischer/perturbierter Pfad sowie EC-Wrapper entfernt)
* **Survivor-Black** (ghosting-frei) & **Event-Timing** (CUDA-Events)
* **Anti-Black-Guard** (Warm-up-Drift + Void-Bias)
* **Yaw-Limiter** + **Längendämpfung**, **Hysterese/Lock** & dyn. **Retarget-Throttle**
* **Softmax-Sparsification** (Designbestandteil; aktuell ohne EC-Eingang aktiv)  
* **Epoch-Millis-Logging** (UTC-Millis seit 1970) — kompakt, sortier- & skriptfreundlich

---

## 🖥️ Systemvoraussetzungen

* Windows 10/11 **oder** Linux
* **NVIDIA GPU** mit CUDA (Compute Capability **8.0+**, empfohlen **8.6+**)
* **CUDA Toolkit v13.0+ (erforderlich)** – inkl. `nvcc`
* Visual Studio 2022 **oder** GCC 11+
* CMake (Version **≥ 3.28**), Ninja
* vcpkg (für GLFW, GLEW; **GLEW dynamisch**, kein `GLEW_STATIC`)

> ⚠️ GPUs unter Compute Capability 8.0 (z. B. Kepler, Maxwell) werden **nicht** unterstützt.  
> ⚠️ OpenGL **4.3 Core** wird vorausgesetzt.

---

## 📦 Abhängigkeiten (via vcpkg)

* [GLFW](https://www.glfw.org/) – Fenster-/Eingabe-Handling  
* [GLEW](http://glew.sourceforge.net/) – OpenGL-Extension-Management (**dynamisch**)

---

## 🔧 Build-Anleitung

> **Hinweis:** Der Build läuft vollständig über **Standard-CMake** (host-agnostisch).
> Ein **optionales** PowerShell-Skript `build.ps1` kann vorhanden sein, wird aber nicht benötigt.

### 1) Repository & vcpkg holen

```bash
git clone --recurse-submodules https://github.com/Funcry241/mandelbrot-fraktal-maus.git
cd mandelbrot-fraktal-maus
# vcpkg lokal bootstrappen (unter Windows .bat verwenden)
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh            # Linux/macOS
bootstrap-vcpkg.bat             # Windows (PowerShell oder CMD)
cd ..
```

### 2) Windows (MSVC + Ninja)

```powershell
cmake -S . -B build -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="${PWD}/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# (optional) Installationsbaum erzeugen
cmake --install build --prefix .\dist
# Ausführen
.\dist\mandelbrot_otterdream.exe
```

### 3) Linux (GCC + Ninja)

```bash
cmake -S . -B build -G Ninja   -DCMAKE_TOOLCHAIN_FILE="$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake"   -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# (optional) Installationsbaum erzeugen
cmake --install build --prefix ./dist
# Ausführen
./dist/mandelbrot_otterdream
```

> **Tipp:** Abweichende Compute Capability beim Konfigurieren überschreiben:
>
> ```bash
> cmake -S . -B build -G Ninja >   -DCMAKE_CUDA_ARCHITECTURES=90 >   -DCMAKE_TOOLCHAIN_FILE="$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake" >   -DCMAKE_BUILD_TYPE=Release
> ```

---

### ⌨️ Keyboard Controls

* `P`: Auto-Zoom pausieren/fortsetzen
* `H`: Heatmap-Overlay ein/aus (derzeit ohne EC-Daten)
* `T`: HUD (Warzenschwein) ein/aus

> Hinweis: `Space` ist derzeit **nicht** gemappt (kein Alias zu `P`).

---

## 🌊 Das Robbe-Prinzip (API-Synchronität)

**Seit Alpha 41 gilt:** Header und Source bleiben **synchron**. Kein Drift, kein API-Bruch. Die Robbe wacht.

> „API-Änderung ohne Header-Update? Dann OOU-OOU und Build-Fehler!“

**Referenz-Signaturen (aktuell):**
* `src/capybara_frame_pipeline.cuh` → **`capy_render(...)`**
* `src/cuda_interop.hpp` → **`renderCudaFrame(...)`** (Overloads)

---

## 🦝 Waschbär-Prinzip (Auto-Fix & Hygiene)

**Ab Alpha 53:** Der Build prüft auf bekannte Toolchain-Fallen (z. B. `glew32d.lib`) und hält CMake-Hygiene hoch — **ohne** projektexterne Skripte.

---

## 🔎 Qualitäts-Guards (Kurzüberblick)

* **Anti-Black-Guard**: Warm-up-Drift & Void-Bias – kein „Zoom ins Schwarze“
* **Survivor-Black**: Ghosting-freie Slices
* **Hysterese/Lock**: verhindert Ziel-Flip-Flops
* **Retarget-Throttle**: CPU-schonend, ruhiger Kurs
* **Softmax-Sparsification**: ignoriert irrelevante Tails (EC aktuell deaktiviert)

---

## 🧭 Zoomgerichtet & geschmacksgetestet

Silk-Lite koppelt **Zielwahl** und **Bewegung**. Der Designpfad sieht Entropie/Kontrast als Signalquelle vor; aktuell ist EC **deaktiviert**.  
Der Planner arbeitet daher mit **ForceAlwaysZoom**, Yaw-Limiter, Dämpfung, Hysterese/Lock und Retarget-Throttle für ruhige, stetige Kamerafahrten — ohne „ins Schwarze“ zu kippen.

---

## ⚙️ Konventionshinweise

* **Logging**: ASCII-only; strikt **einzeilig** pro Event. Zeitstempel sind **Epoch-Millis (UTC)**.  
  `debugLogging` für Diagnose; `performanceLogging` misst budgetschonend via CUDA-Events.
* **Tier-Codename-Pflege**: Nicht genutzte/retirierte Namen stehen im **Friedhof** (Doppelvergabe vermeiden).

---

## 📄 Lizenz

MIT-Lizenz – siehe [LICENSE](LICENSE).

---

**OtterDream** – von der Raupe zum Fraktal-Schmetterling 🦋  
*Happy Zooming!*

🐭 Maus sorgt für Fokus und ASCII-Sauberkeit.  
🦊 Schneefuchs bewacht die Präzision.  
🦦 Otter treibt den Zoom unaufhaltsam.  
🦭 Robbe schützt die API-Würde.  
🦝 Waschbär hält den Build hygienisch.  
🦉 Eule sorgt für Überblick in Heatmap & Koordinaten.
