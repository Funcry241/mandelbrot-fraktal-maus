<!-- Datei: AGENTS.md -->
<!-- 🐭 Maus-Kommentar: Dokumentiert Buildprozesse und Toolchains für OtterDream. Aktualisiert: Single-Path-Render (Capybara), LUCHS_LOG-Trennung, kompakte PERF-Logs (Epoch-Millis). Schneefuchs flüstert: „Ein Agent kennt die versteckten Knöpfe und sorgt für saubere Übergänge.“ -->

# 👩‍💻 OtterDream Build Agents

Diese Datei beschreibt die automatisierten Prozesse, lokalen Helfer und Regeln rund um Build, Logging und Pflege des **OtterDream Mandelbrot-Renderers**.

**Seit Alpha 41** gilt das **Robbe-Prinzip**: *Header & Source bleiben synchron. Kein Drift, kein API-Bruch.*  
**Seit Alpha 81** zusätzlich relevant: *Silk-Lite Zoom*, *Frame-Budget-Pacing* und **ASCII-only Logs** ohne Seiteneffekt.  
**Aktiver Renderer-Pfad:** **Capybara → Iterationen → Colorizer → PBO** (ein Pfad).

---

## 🧑‍🔬 Overview

| Agent/Tool               | Zweck                           | Trigger                 | Aktionen                                                             |
| ------------------------ | ------------------------------- | ----------------------- | -------------------------------------------------------------------- |
| **GitHub Actions (CI)**  | Build-, Test-, Install-Check    | Push/PR auf `main`      | CMake Configure → Ninja Build → `cmake --install`                    |
| **Dependabot**           | Abhängigkeits-Updates (vcpkg)   | Wöchentlich             | PRs für `vcpkg.json`, CI baut PR                                     |
| **Waschbär-Watchdog**    | Hygiene & Auto-Fixes (lokal)    | On-Demand               | Räumt CMake-Caches, fixt typische GLEW/vcpkg-Fallen                  |
| **Autogit (lokal)**      | Mini-CI für Commits/Push        | Nach erfolgreichem Build| `git add -A` → `git commit -m "<msg>"` → `git push` (https Fallback) |
| **Rust Runner (lokal)**  | Komfort-Build mit Live-Progress | Manuell (CLI/PS)        | Farben/Spinner/%/ETA, Log-Tags `[PS]/[RUST]/[PROC]`, ETA aus Metrics |

> CI stellt sicher, dass **Debug-/Perf-Logging keine Seiteneffekte** erzeugt (keine erzwungenen Synchronisationen im Hot-Path).

---

## 🦀 Rust Build Runner (otter_proc)

**Zweck:** Lokale Orchestrierung mit **Live-Progress** (Spinner, **%**, **ETA**, ASCII-Bar), farbigen Tags und stabilen Logs.  
**Tags:** `[PS]` = Shell/Script, `[RUST]` = Runner selbst, `[PROC]` = Kindprozess (CMake/Ninja).

**Signal-Parsers:**  
- Prozent **`68%`** und Ratio **`[17/45]`** (CMake/Ninja/MSBuild-ähnlich).  
- Debounce/Rate-Limit: **200 ms** Animationstakt.

**ETA-Seeding:**  
- Datei: `.build_metrics/metrics.json` (pro Arbeitsverzeichnis).  
- Key-Signatur: `exe:phase` (z. B. `cmake:configure`, `cmd:build`).  
- Zeitbasierte Fortschritts-Absicherung (Zeit-Prozent max mit Builder-Prozent fusioniert).

**Env-Toggles:**  
- `OTTER_PROGRESS=0` → Progress-UI aus (Default: **an**)  
- `OTTER_COLOR=0` → Farben aus (Default: **an**)  
- `OTTER_ASCII=1` → ASCII-Spinner/Balken erzwingen

**Cache-Schutz:**  
- Erkanntes **CMake-Cache-Mismatch** (Repo-Root-Wechsel) ⇒ Runner löscht `build/` sicher und konfiguriert neu.

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

---

## 🧯 Host/Device-Logging (LUCHS_LOG)

* **Host**: `LUCHS_LOG_HOST(...)` – ASCII-only, **eine Zeile pro Event**, Zeitstempel als **Epoch-Millis**.
* **Device**: `LUCHS_LOG_DEVICE(msg)` – schreibt in den Device-Puffer; Flush auf Host **außerhalb** des Hot-Paths.  
  *Hinweis:* Nachricht mit `snprintf` zusammenbauen ist ok – der **finale** Aufruf ist genau **ein** `LUCHS_LOG_DEVICE(const char*)`.
* **Keine** `printf/fprintf` im Produktionspfad. Logs dürfen **keine** impliziten Synchronisationen auslösen.
* **Schalter (Settings)**:  
  `performanceLogging` → kompakte Messwerte via CUDA-Events (ASCII)  
  `debugLogging` → detaillierter, ggf. langsamer

---

## ⏱️ Frame-Budget-Pacing (Silk-Lite kompatibel)

Der Mandelbrot-Pfad hält sich an ein weiches **Zeitbudget** pro Frame. Silk-Lite steuert Bewegung (Yaw-Limiter + Dämpfung).  
**Regel:** Pacing misst mit **CUDA-Events** (kostenarm) und **erzwingt keine** globale Synchronisation.

---

## 🎨 Renderer-Pfad & Farbgebung (Status)

* **Aktiver Pfad:** **Capybara-Iteration** (Float), Escape-Test **vor** dem Update (`|z|^2 > 4`).  
  – *Innen* schreibt `iterOut = maxIter`, *Escape* schreibt den Iterationsindex.  
* **Pipeline:** `capy_render(...)` (Iterations) → `colorize_iterations_to_pbo(...)` → PBO (GL-Interop).  
* **Palette:** **GT (Cyan→Amber)**, Interpolation im **Linearraum** (Banding-mindernd).  
  **Stripes** sind **standardmäßig aus** (`stripes = 0.0f`) für ringfreie Darstellung.  
* **Mapping:** Projektweit über `screenToComplex(...)` (Koordinaten-Harmonisierung, „Eule“).

---

## ⌨️ Hotkeys (Runtime)

| Taste   | Funktion                               |
| ------- | -------------------------------------- |
| `P`     | Auto‑Zoom pausieren/fortsetzen         |
| `H`     | Heatmap‑Overlay toggeln                |
| `T`     | HUD (Warzenschwein) toggeln            |
| `Ctrl+P`| **Perturb‑Gate** toggeln               |

---

## 🧷 Toolchain & Hardening (Windows)

* **CRT**: `/MD` (DLL) – konsistent zum Host-Link; keine LNK2038-Mismatches.  
* **CUDA Runtime**: **Shared**; Runtime‑DLL wird in `dist\` kopiert (falls nötig).  
* **GLEW dynamisch**: kein `GLEW_STATIC`; vcpkg‑Triplet entsprechend.  
* **Hardening nur im Host-Link**: `/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf` über `$<HOST_LINK:...>`.  
* **Separable Compilation** + **Device-Symbole** aktiviert (CMake Properties).

---

## 🌊 Robbe-Prinzip (API-Synchronität)

> Jede Änderung an Signaturen/Interfaces wird **zeitgleich** in Header **und** Source umgesetzt (und gemeinsam committed). Abweichungen sind Build-Fehler – Robbe sagt **OOU-OOU**.

* Kein schleichender Drift  
* Saubere öffentliche API  
* **Referenz:** `src/capybara_frame_pipeline.cuh` (Signatur **`capy_render(...)`**) und `src/cuda_interop.hpp` (Signaturen **`renderCudaFrame(...)`**).

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
