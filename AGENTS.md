<!-- Datei: AGENTS.md -->
<!-- 🐭 Maus-Kommentar: Dokumentiert Buildprozesse und Toolchains für OtterDream. Jetzt mit Hotkey-Doku, CUDA-Architektur-Hinweis, Frame-Budget-Pacing-Hinweis und Robbe-Regel für API-Synchronität. Schneefuchs flüstert: „Ein Agent kennt auch die versteckten Knöpfe und sorgt für saubere Übergänge.“ -->

# 👩‍💻 OtterDream Build Agents

Diese Datei dokumentiert die automatisierten Prozesse und Tools für den Build und die Pflege des OtterDream Mandelbrot-Renderers. **Ab Alpha 41 gilt das "Robbe-Prinzip": Alle Header-/Source-Schnittstellen werden IMMER synchron gepflegt. Kein Drift, kein API-Bruch.**

Seit **Alpha 81** zusätzlich relevant:

* **Silk‑Lite Zoom** im Runtime‑Pfad (zeitstabile Drehraten, Längendämpfung)
* **Frame‑Budget‑Pacing** im Kernel-Wrapping (CI geprüft)
* **Logging ohne Seiteneffekte** (reine ASCII‑Logs; Verhalten der Pipelines bleibt unverändert)

---

## 🧑‍🔬 Overview

Das Projekt verwendet folgende Agents und Werkzeuge:

| Agent               | Zweck                                  | Trigger         | Aktionen                                            |
| ------------------- | -------------------------------------- | --------------- | --------------------------------------------------- |
| GitHub Actions (CI) | Build-, Test- & Install-Check bei Push | Push auf `main` | CMake-Konfiguration, Ninja-Build, `cmake --install` |
| Dependabot          | Abhängigkeits-Updates für vcpkg        | Wöchentlich     | Überwachung/PRs für `vcpkg.json`                    |
| Waschbär-Watchdog   | Hygiene & Auto-Fixes (lokal)           | On-Demand       | Bereinigt vcpkg/GLEW-Fallen, räumt CMake-Caches auf |

> CI erzeugt deterministische Artefakte und prüft zusätzlich, dass **Debug-/Perf-Logging keine Seiteneffekte** (z. B. Barrieren, Zustandsänderungen) hat.

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

### ⚠️ CUDA erforderlich

> ❗ **Hinweis:** Für den Build ist eine **lokal installierte CUDA-Toolchain (z. B. `nvcc`) zwingend erforderlich**.
> Ohne CUDA kann der Buildprozess **nicht gestartet** werden.

---

## ⌨️ Keyboard Controls

Diese Tastenkürzel sind während der Laufzeit verfügbar:

| Taste   | Funktion                            |
| ------- | ----------------------------------- |
| `P`     | Auto-Zoom pausieren oder fortsetzen |
| `Space` | Alternativ zu `P`                   |
| `H`     | Heatmap-Overlay ein-/ausschalten    |
| `T`     | HUD (WarzenschweinOverlay) toggeln  |

> Hinweis: Die **Silk‑Lite**-Planung sorgt bei Richtungswechseln für sanfte Übergänge (Yaw‑Limiter + Dämpfung). Das Verhalten ist unabhängig von Debug‑Logs.

---

## 🧠 CUDA Architekturen

Standardmäßig ist in den CMake-Presets die Architektur `80;86;89;90` gesetzt.
Für andere GPUs kann diese wie folgt überschrieben werden:

```bash
cmake --preset windows-release -DCMAKE_CUDA_ARCHITECTURES=90
```

Die passende Architektur für deine GPU findest du auf der offiziellen NVIDIA-Liste.

---

## ⚙️ Lokaler Build

### 🪟 Windows (zwei Wege)

**A) Komfortskript**

```powershell
./build.ps1
```

**B) Manuell mit Presets**

```powershell
cmake --preset windows-msvc
cmake --build --preset windows-msvc
cmake --install build/windows --prefix ./dist
./dist/mandelbrot_otterdream.exe
```

> Das Skript erkennt bekannte Fallstricke (z. B. `glew32d.lib`), bereinigt CMake-Caches und setzt die Pfade für CUDA automatisch.

### 🐧 Linux

1. **Voraussetzungen installieren** (einmalig):

```bash
sudo apt update
sudo apt install build-essential cmake git ninja-build libglfw3-dev libglew-dev libxmu-dev libxi-dev libglu1-mesa-dev xorg-dev pkg-config libcuda1-525
```

> *Hinweis:* Je nach Distribution kann die CUDA-Runtime-Bibliothek anders heißen (z. B. `libcuda1-545`).

2. **Repository klonen & vcpkg initialisieren**:

```bash
git clone --recurse-submodules https://github.com/Funcry241/otterdream-mandelbrot.git
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
./dist/mandelbrot_otterdream
```

---

## 🌊 Das Robbe-Prinzip (API-Synchronität)

Ab Alpha 41 gilt:

> **Jede Änderung an Funktionssignaturen, Headern oder APIs wird immer gleichzeitig in Header- und Source-Dateien umgesetzt und committed. Kein Drift!**

* Nie wieder schleichende Bugs durch asynchrone Schnittstellen.
* Funktionsänderungen, die Robbe nicht sieht, werden nicht gebaut.

Robbe wacht über jede Funktion. Wenn Header und Source abweichen, watschelt sie quer durch den Commit und macht lautstark **OOU‑OOU!**

---

## 🧪 Logging-Regeln (seit Alpha 81)

* **ASCII‑only** – keine binären Dumps im Hot‑Path.
* **Zero Side‑Effects** – Logs dürfen **keine** Zustände verändern, keine Synchronisationspunkte erzwingen und sind klar hinter Performance‑kritischen Pfaden platziert.
* **Performance‑Logging** und **Debug‑Logging** sind strikt getrennt und können unabhängig voneinander aktiviert werden.

---

## 🌐 CI/CD Pipelines

**GitHub Actions**

* Workflow: `.github/workflows/ci.yml`
* Schritte: Configure → Build (Ninja) → Install
* Artefakte: Install‑Tree unter `dist/`
* Prüft zusätzlich:

  * erfolgreiche CUDA‑Kompilation für Presets
  * konsistente CMake‑Presets
  * deterministische Builds (gleiche Inputs → gleiche Outputs)

**Dependabot**

* Automatisches Update der vcpkg‑Abhängigkeiten (wöchentlich)
* PRs werden vom CI‑Workflow gebaut

---

## ❓ Troubleshooting (Kurz)

* **nvcc nicht gefunden** → CUDA 12.9 installieren und PATH prüfen.
* **Linker findet `glew32d.lib`** → vcpkg‑Triplet auf Release prüfen; im Zweifel `build.ps1` nutzen (räumt auf).
* **Schwarze Frames bei extremer Kamerabewegung** → sicherstellen, dass Runtime‑Einstellungen (Silk‑Lite/Anti‑Black‑Guard) aktiv sind; Logging muss aus sein bei Performance‑Messungen.

---

**Agenten‑Motto:** Maus sorgt für Fokus, Schneefuchs für Präzision, Robbe für API‑Disziplin, Waschbär für Hygiene. 💫
