<!-- Otter: KI-Verhalten & Projektregeln für OtterDream/Fraktal -->
<!-- Schneefuchs: Leitplanken für Logging, Numerik, Zoom & Telemetrie -->
<!-- Maus: Zusammenarbeit Mensch ↔ KI, keine heimlichen Abkürzungen -->
<!-- Datei: KI-RULES.md -->

# KI-RULES

Diese Datei beschreibt, wie eine KI (z.B. ChatGPT) im Kontext des OtterDream/Fraktal-Projekts arbeiten soll.  
Ziel: reproduzierbare, nachvollziehbare, saubere Beiträge ohne „magische“ Abkürzungen.

---

## 1. Zweck & Geltungsbereich

1. Diese Regeln gelten für **alle Vorschläge**, die die KI zum Projekt macht:
   - Code (C/C++/CUDA/GLSL, PowerShell, CMake, Markdown-Dokumentation, …)
   - Architektur- und Designentscheidungen
   - Logging-, Performance- und Diagnosepfade
2. Die Regeln betreffen **sowohl**:
   - Verhalten der KI im Chat
   - Struktur und Inhalte der erzeugten Dateien

---

## 2. Grundprinzipien

1. **Keine Lazy-Lösungen**  
   Vorschläge müssen auf einem **state-of-the-art**-Niveau sein, nicht als schnelle Notlösung.
2. **Analyse vor Aktion**  
   Besonders im Projekt „Regenwurm/Pyramiden-Otter“:
   - Erst Diagnoseplan → Datensammlung → Auswertung → Maßnahmenvorschlag.
   - Keine Symptom-Pflaster ohne belegte Ursache.
3. **Keine stillen Regressionen**  
   - Keine Performance-, Stabilitäts- oder Qualitätsregression ohne klare Kennzeichnung und explizite Zustimmung.
4. **Ein Pfad, klare Nutzung (Genfer Großente)**  
   - Neue Kernpfade (Orbit, DE, Capybara, Progressive, etc.) werden sauber angebunden, haben Telemetrie und klare Fallbacks.
   - Keine toten Schalter, keine geheimen Zweitpfade.

---

## 3. Logging-Regeln

### 3.1 Exklusive LUCHS-Logging-Pfade

1. **Host-Logging**: ausschließlich `LUCHS_LOG_HOST(...)`.
2. **Device-Logging**: ausschließlich `LUCHS_LOG_DEVICE(...)`.
3. Keine direkten `printf`, `fprintf`, `std::cout`, `OutputDebugString`, o.Ä. im Runtime-Code.

### 3.2 Device-Logging-Pattern

1. In `__device__` Code ist `snprintf` **erlaubt**, aber nur im Muster:
   ```cpp
   char msg[512];
   snprintf(msg, sizeof(msg), "text it=%d", it);
   LUCHS_LOG_DEVICE(msg);
   ```
2. Es gibt immer genau **einen** Makro-Aufruf mit `const char*` pro Log-Nachricht.

### 3.3 Sprache & Zeichensatz der Logs

1. Alle Runtime-Logs (Host & Device) sind:
   - **Englisch**
   - **ASCII-only** (keine Umlaute, kein Unicode-Sonderzoo).
2. Dies betrifft insbesondere:
   - Telemetrie-Zeilen
   - Fehlermeldungen
   - Debug-Ausgaben

### 3.4 Logging darf Verhalten nicht ändern

1. Logging darf Numerik, Timings und Funktionalität nicht merklich beeinflussen.
2. Heißt konkret:
   - Debug-Logs gehören hinter Flags wie `Settings::debugLogging`.
   - Performance-Telemetrie ggf. hinter `Settings::performanceLogging`.
3. Rate-Limits für Device-Logs sind erwünscht, um Warp-/Frame-Spam zu vermeiden.

### 3.5 Performance-Logging-Default

1. `Settings::performanceLogging` ist standardmäßig **true**.
2. Telemetrie ist:
   - Kompakte ASCII-Einzeiler (Frame-Time, FPS, Budget, wichtige Flags).
   - Lesbar und stabil, damit Log-Auswertungen reproduzierbar bleiben.

---

## 4. Code-, Datei- & Header-Regeln

### 4.1 Komplettdateien statt Snippets

1. Wenn die KI eine Datei ändert oder neu vorschlägt:
   - Es wird **immer** die **vollständige** Datei geliefert (keine ausgelassenen Zeilen).
2. Änderungen werden so präsentiert, dass sie direkt ins Repo übernommen werden können.

### 4.2 Dateiname vor Codeblöcken

1. Vor jedem Codeblock in der Chat-Antwort steht genau **eine** Zeile:
   - `Datei: <relativer/pfad/zur/datei.ext>`
2. Pro Antwort und Datei nur eine solche Zeile (keine Wiederholung vor jedem Block).

### 4.3 Kanonischer Dateikopf im Code

1. Jede Source-/Header-/Shader-/Script-Datei beginnt mit genau 4 Kommentarzeilen:
   1. `Otter: ...`
   2. `Schneefuchs: ...`
   3. `Maus: ...`
   4. `Datei: <relative/path/to/file>`
2. Kommentarprefix je nach Sprache:
   - C/C++/CUDA/GLSL: `/////`
   - Scripts/CMake: `#####`
   - PowerShell: `#`
   - Markdown: HTML-Kommentare `<!-- ... -->` (wie in dieser Datei).

### 4.4 5-Zeichen-Prefix

1. Ein 5-Zeichen-Block am Zeilenanfang (z.B. `/////`, `#####`) ist für Meta-Kommentare reserviert.
2. Die KI nutzt diesen Prefix **konsistent** für:
   - Otter/Schneefuchs/Maus-Tags
   - sonstige Projekt-Metakommentare

### 4.5 Kommentar-Tags

1. Kommentare werden markiert mit:
   - **Otter**: Ideen/Änderungen, die vom Benutzer ausgehen.
   - **Schneefuchs**: Hinweise aus pedantischer Analyse / Code-Audit.
   - **Maus**: kleine Metakommentare, Checks, Randnotizen.

### 4.6 `settings.hpp`-Dokumentationspflicht

1. Jede Einstellung in `settings.hpp` muss dokumentiert sein:
   - Funktion / Wirkung
   - Sinnvolle Wertebereiche (Min–Max oder „typisch“)
   - Effekt bei Erhöhung/Verringerung (z.B. mehr Qualität, weniger FPS)
2. Änderungen an `settings.hpp` ohne passende Doku sind **nicht** erlaubt.

### 4.7 Header/Source-Synchronität

1. Header (.hpp) und Implementierung (.cpp/.cu) müssen immer synchron bleiben:
   - Signaturen, Parameterlisten, `const`-Qualifizierungen, Namespaces.
2. Die KI ändert **nie** nur den Header oder nur die Source – immer beide zusammen, wenn nötig.

---

## 5. Build-, Toolchain- & Skript-Regeln

### 5.1 GLEW & OpenGL

1. GLEW wird **dynamisch** verwendet (kein `GLEW_STATIC` als Standard).
2. Mindestanforderung: **OpenGL 4.3 Core**.

### 5.2 CUDA_CHECK

1. Es gibt genau **eine** Definition von `CUDA_CHECK`, in `luchs_log_host.hpp`.
2. Die KI führt **keine** weiteren, konkurrierenden Macros ein.

### 5.3 PowerShell 5.1 Kompatibilität

1. Alle PowerShell-Snippets müssen mit **Windows PowerShell 5.1** funktionieren:
   - Kein ternärer `?:`-Operator.
   - Keine PS7-Pipeline-Chain-Operatoren `&&` / `||`.
   - Kein `??` / `??=`.
   - Kein `ForEach-Object -Parallel`.
2. Sprachfeatures >5.1 dürfen nur benutzt werden, wenn explizit freigegeben.

### 5.4 build.ps1 & CI

1. Das große `build.ps1` ist stabiler Kern; Änderungen erfolgen nur bewusst und begründet.
2. CI-/vcpkg-/DLL-Logik wird vorsichtig angepasst – keine spontanen Umbauten.

### 5.5 Parameter-Tuning (kein JSON-Hot-Reload)

1. JSON-Hot-Reload wird **nicht** verwendet.
2. Tuning läuft über:
   - Auto-Tuner mit einfachem booleschen Schalter (z.B. in `zoom_logic.cpp`).
   - Auto-Tuner loggt regelmäßig seine besten Parameter per ASCII-Zeile.

---

## 6. Laufzeit-Defaults & Feature-Schalter

1. **HUD-Overlays**
   - Warzenschwein-HUD (Text-HUD) ist standardmäßig aktiviert.
   - Heatmap-Overlay ist standardmäßig aktiviert.
2. **Zoom-Logik**
   - Zoom V3 besitzt ein Warm-Up-Freeze (keine Richtungswechsel zu Beginn).
   - `Settings::ForceAlwaysZoom` steht standardmäßig auf **true**.
3. **Alte Debugpfade**
   - `debugGradient`, Testkernel & ähnliche Debugpfade sind vollständig entfernt.
   - Die KI führt diese Pfade nicht stillschweigend wieder ein.

---

## 7. Architektur- & Designregeln

### 7.1 FrameContext & RendererState

1. **FrameContext**
   - Zentrale Sammlung von Analysewerten (Entropy, Contrast, Zoomziel etc.).
   - Teilt Daten zwischen CUDA, Zoom-Logik, HUD, Heatmap.
2. **RendererState**
   - Kümmert sich um Plattform-/Ressourcendetails (GL-Kontext, PBOs, Texturen, etc.).
   - Enthält **keine** High-Level-Analyse- oder Zoomlogik.
3. **tileSize**
   - Wird zentral in der Frame-Pipeline bestimmt.
   - Wird explizit an `setupCudaBuffers(...)` und `launch_mandelbrotHybrid(...)` übergeben.
   - Kein „Eigenleben“ mit eigenen Berechnungen in Komponenten.

### 7.2 Zoom-Logik (Rullmolder/Otter/Regenwurm)

1. Zoom-Logik (z.B. `zoom_logic.cpp`) darf als Input **nur** nutzen:
   - Heatmap (Entropy/Contrast-Grid).
   - Daraus abgeleitete Signale (z.B. Interest-Map, Crosshair).
2. Sie darf **nicht**:
   - Direkt in das gerenderte RGB-Bild schauen.
   - Irgendwelche „geheimen“ Zusatzkanäle nutzen.
3. Die aktuelle Steuerung (PD-Planer, Softmax-Ziel, Hysterese, Retarget-Throttle etc.) ist Baseline:
   - Tuning nur vorsichtig und nachvollziehbar.
   - API/Headers werden nicht ohne explizite Freigabe angefasst.

### 7.3 Genfer Großente (Deep-Zoom-Regelwerk)

1. Leitprinzipien für tiefe Zooms:
   - **Ein Pfad**: keine konkurrierenden, halbfertigen Alternativwege.
   - **Sofortige Nutzung**: neue Numerik wird auch wirklich im Bild verwendet.
   - **Budget-Gate**: tiefe Orbit-/DE-Berechnungen unterliegen einem Frame-Budget.
   - **Lauter Fallback**: bei Problemen gibt es klare ASCII-Fallback-Logs, kein stummes Versagen.
2. Weitere Punkte:
   - Orbit-Pipelining in Segmenten.
   - Konsistente δ-Transformation, konsistentes `z/z'`.
   - DE-Soft-Edge statt „Zufalls-SSAA“.

### 7.4 Projekt Capybara (Numerik ohne BigFloat/Perturbation)

1. Ziel:
   - Höhere Genauigkeit ohne BigFloat/Perturbation.
2. Mittel:
   - Binäre Skalierung über `frexp/ldexp`, regelmäßige Renormalisierung.
   - Kompensierte Summen (hi+lo) an Hotspots (Mapping, frühe Iterationen).
   - Einsatz von FMA, wo sinnvoll.
3. Steuerung:
   - Renorm-Trigger und Early-Iter-Limits sind Settings-gesteuert.
4. Telemetrie:
   - CAPY-ASCII-Zeilen mit Exponent, Renorm-Zählung etc.

### 7.5 Projekt Nacktmull (Performance-Kader)

1. „Nacktmull“ steht für den 9-Dateien-Optimierungssatz (Warp Exit, Block-Layout, Unroll, Cardioid/Bulb-Tests, Log-Rate-Limits, etc.).
2. Regeln:
   - Nur zielgerichtete Performance-Optimierungen, keine Verunstaltung der Lesbarkeit.
   - Logs stark rate-limitiert (keine Spam-Flut).
   - Keine Rückkehr zu alten langsamen Pfaden ohne guten Grund.

---

## 8. Daten, Artefakte & Anhänge

1. Alte Logs, ZIPs, Screenshots aus früheren Chats gelten **nicht automatisch**.
2. Die KI verlässt sich nur auf Artefakte, die:
   - Im aktuellen Thread zur Verfügung stehen, oder
   - Explizit referenziert / erneut hochgeladen wurden.

---

## 9. Zusammenarbeit Mensch ↔ KI

### 9.1 Bestätigung vor Code

1. Die KI fragt grundsätzlich:
   - „Soll ich dir den Code / die Datei jetzt schicken?“
   - Bevor vollständige Dateien oder größere Codeblöcke gesendet werden.
2. Ausnahmen können durch Systemvorgaben erzwungen sein; die Absicht bleibt, Code nicht „heimlich“ zu liefern.

### 9.2 Schrittweise Vorgehensweise

1. Bevor Code geändert wird:
   - Analyse/Audit (z.B. „Specht“, „Regenwurm L1“).
   - Klarer Plan: welche Datei, welcher Abschnitt, erwarteter Effekt.
2. Änderungen erfolgen in kleinen, nachvollziehbaren Schritten:
   - „Erst dieses Modul, dann jenes.“
   - Klare Markierung der betroffenen Bereiche.

### 9.3 Keine heimlichen Änderungen

1. Die KI benennt immer:
   - Welche Dateien betroffen sind.
   - Welche Funktionen / Strukturen verändert werden.
2. Keine stillen Nebenwirkungen:
   - Keine versteckten API-Änderungen.
   - Keine „Bonus“-Umbauten außerhalb des besprochenen Bereichs.

### 9.4 Scriban-Templates

1. In Scriban-Templates werden **keine Inline-Kommentare** verwendet.
2. Erklärungen erfolgen, falls nötig:
   - Über `sys.log(...)`-Ausgaben.
   - Oder externe Dokumentation (z.B. README-Abschnitt).

---

## 10. Telemetrie & Statuszeilen

1. Pro Frame existiert (falls aktiv) eine konsistente ASCII-Telemetriezeile:
   - Frame-Time, FPS.
   - Zoom, Offset.
   - Entropy[0]/Contrast[0] oder zentrale Kennzahlen.
   - Statusflags (CAPY, DE, Progressive, Survivor, …).
2. Telemetrie ist stabil genug, um Log-Vergleiche zwischen Builds/Branches zu ermöglichen.

---

## 11. Pflege & Erweiterung dieser Datei

1. Neue Regeln werden nur hinzugefügt, wenn sie:
   - Wiederkehrende Probleme lösen, oder
   - Eine etablierte Praxis offiziell machen.
2. Beim Ändern dieser Datei:
   - Otter/Schneefuchs/Maus-Kommentar im Kopf aktualisieren.
   - Klar kennzeichnen, ob Regeln ersetzt oder ergänzt wurden.
3. Diese Datei ist verbindliche Referenz für das KI-Verhalten im Projekt.
