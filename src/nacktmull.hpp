// ============================================================================
// Datei: src/nacktmull.hpp
// Projekt Nacktmull – Perturbation/Series-Engine für unendliches Zoomen
// ----------------------------------------------------------------------------
// 🦙 Idee (kurz): Ein Referenzzentrum c0 wird hochpräzise auf der CPU iteriert.
// Für jedes Pixel gilt c = c0 + δc. Auf der GPU wird nur δz iteriert
// (Perturbation mit Quadratik-Term), während z⁰_n (Orbit um c0) aus einem
// vorab berechneten Puffer gelesen wird. Bei zu großem Fehler/δc wird re-centered.
// ----------------------------------------------------------------------------
// Vorgabe: Diese Engine ersetzt den bisherigen Hybrid-Renderer vollständig.
// (Kein Feature-Flag, kein Fallback – Integration erfolgt im core_kernel-Pfad.)
// ----------------------------------------------------------------------------
// ASCII-only, header-only API. Implementierung folgt in nacktmull.cu/.cpp.
// ============================================================================

#pragma once

#include <cuda_runtime.h>    // float2, double2, uchar4
#include <cstdint>
#include <cstddef>
#include <vector>

// Vorwärtsdeklaration Host-Logger (bestehendes Projekt-API)
namespace LuchsLogger { void logMessage(const char* file, int line, const char* fmt, ...); }
#ifndef LUCHS_LOG_HOST
#define LUCHS_LOG_HOST(...) ::LuchsLogger::logMessage(__FILE__, __LINE__, __VA_ARGS__)
#endif

namespace Nacktmull {

// ---------------------------------------------------------------------------
// Konfiguration (Host)
// ---------------------------------------------------------------------------
struct Config {
    // Max. Länge des Referenz-Orbits (CPU, hohe Präzision)
    int     maxOrbitTerms      = 200000;   // kann dynamisch gekürzt werden
    // Re-Centering-Schwelle (|δc|): darüber neues c0 bestimmen
    double  recenterThreshold  = 1e-12;
    // Max. erlaubter Perturbationsfehler (heuristisch, in |δz|)
    double  maxDeltaError      = 1e-10;
    // Max. Anzahl Re-Centerings pro Frame (Schutz gegen Ping-Pong)
    int     maxRecentersPerFrame = 2;
    // Budget für CPU-Referenz (ms); 0 = adaptiv (aus Framebudget ableiten)
    double  refOrbitBudgetMs   = 0.0;
};

// Minimaler Telemetriedatensatz für eine kompakte PERF-Zeile
struct Stats {
    double ref_ms     = 0.0;  // CPU-Referenzzeit
    double kern_ms    = 0.0;  // GPU-Kernelzeit gesamt
    int    centers    = 0;    // Anzahl Re-Centerings in diesem Frame
    int    orbitN     = 0;    // Länge des verwendeten Orbits
    double err_max    = 0.0;  // gemessene maximale |δz|-Fehlergröße
};

// ---------------------------------------------------------------------------
// Host-Seite: Referenz-Orbit Speicher
// ---------------------------------------------------------------------------
struct RefOrbitHost {
    // z⁰_n als double2 (Host-Repräsentation); bei echter High-Precision werden
    // diese Werte nach Konvertierung auf double in die GPU übertragen.
    std::vector<double2> z;   // Länge = orbitN (inkl. z0 = 0)
};

// ---------------------------------------------------------------------------
// Device-Seite: Referenz-Orbit Buffer (RAII-light via Methoden)
// ---------------------------------------------------------------------------
struct RefOrbitDevice {
    double2* d_z   = nullptr; // Geräteseitiger Puffer für z⁰_n
    int      count = 0;       // Anzahl Elemente (Orbit-Länge)

    void free() noexcept {
        if (d_z) { cudaFree(d_z); d_z = nullptr; }
        count = 0;
    }

    // Allokiert (oder vergrößert) den Orbit-Puffer auf 'n' Elemente
    void ensure_capacity(int n) {
        if (n <= count) return;
        if (d_z) cudaFree(d_z);
        cudaMalloc(&d_z, static_cast<size_t>(n) * sizeof(double2));
        count = n;
    }
};

// ---------------------------------------------------------------------------
// Host-API: Engine-Lebenszyklus
// ---------------------------------------------------------------------------
struct View {
    int    width  = 0;
    int    height = 0;
    double zoom   = 1.0;      // View-Zoom (wie bisher)
    float2 offset = make_float2(0.f, 0.f);
};

struct Center {
    // Aktuelles Referenzzentrum c0 in doppelter Präzision (GPU bekommt double)
    double c0x = -0.5;  // Standard-Mandelbrot
    double c0y = 0.0;
};

class Engine {
public:
    Engine() = default;
    ~Engine() { device_.free(); }

    // Initialisiert View/Config; invalidiert ggf. bestehende Orbitdaten
    void initialize(const View& v, const Config& cfg);

    // Setzt neue Ansicht und markiert Rebuild-Bedarf, wenn c0 zu weit weg ist
    void update_view(const View& v);

    // Haupt-Einstieg: Sorgt für Referenz-Orbit (CPU) + Upload + GPU-Rendering
    // Schreibt Iterationen/Pixel (wie bisher: out + d_iters) über Perturbation.
    void render_frame(uchar4* d_out, int* d_iters, int maxIter, Stats& outStats);

    // Zugriff auf internes Zentrum (c0)
    const Center& center() const { return center_; }

private:
    // CPU: Referenz-Orbit neu aufbauen (ggf. in High-Precision und danach auf double konvertieren)
    void build_ref_orbit_cpu(int maxIter, double budgetMs, Stats& stats);

    // GPU: Orbit an Gerät senden
    void upload_ref_orbit_gpu();

    // GPU: Perturbations-Render starten (ein oder mehrere Kernel, Budget-aware außerhalb steuern)
    void launch_perturbation_kernel(uchar4* d_out, int* d_iters, int maxIter, Stats& stats);

    // Heuristik: Re-Centering nötig?
    bool recenter_required() const;

private:
    Config          cfg_{};
    View            view_{};
    Center          center_{};     // aktuelles c0
    RefOrbitHost    hostOrbit_{};  // Host-Puffer
    RefOrbitDevice  device_{};     // Device-Puffer
    bool            needRecenter_ = true; // erzwinge Aufbau beim ersten Frame
};

// ---------------------------------------------------------------------------
// GPU-Schnittstellen (Definition in nacktmull.cu)
// ---------------------------------------------------------------------------
// Kernel: Perturbationsiteration mit vorgegebenem Referenz-Orbit z⁰_n
// - d_refOrbit: z⁰_n als double2; Länge = orbitCount
// - c0: Referenzzentrum
// - View/Geometrie: wie gewohnt (w,h, zoom, offset)
// - maxIter: iteratives Limit/Bailout-Guard
// Ergebnis: d_out (RGBA), d_iters (Iteration pro Pixel)

void launchPerturbationPass(
    uchar4* d_out,
    int*    d_iters,
    int     w,
    int     h,
    double  zoom,
    float2  offset,
    int     maxIter,
    const double2* d_refOrbit,
    int     orbitCount,
    double  c0x,
    double  c0y,
    double& outKernelMs  // gefüllte Kernelzeit in Millisekunden
);

// Hilfsfunktion: Pixel → komplexe Zahl (double), im Nacktmull-Koordinatensystem
__host__ __device__ inline double2 pixelToComplexD(
    double px, double py, int w, int h, double spanX, double spanY, double2 off)
{
    return make_double2(
        (px / double(w) - 0.5) * spanX + off.x,
        (py / double(h) - 0.5) * spanY + off.y
    );
}

} // namespace Nacktmull
