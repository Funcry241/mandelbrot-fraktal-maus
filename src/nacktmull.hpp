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
/* ASCII-only, header-only API. Implementierung folgt in nacktmull.cu/.cpp. */
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
    int     maxOrbitTerms        = 200000; // maximale Referenz-Orbit-Länge (CPU)
    double  recenterThreshold    = 1e-12;  // |δc|-Schwelle für Re-Centering
    double  maxDeltaError        = 1e-10;  // tolerierter |δz|-Fehler (heuristisch)
    int     maxRecentersPerFrame = 2;      // Schutz gegen Ping-Pong
    double  refOrbitBudgetMs     = 0.0;    // Budget für CPU-Referenz (0 = adaptiv)
};

// Minimaler Telemetriedatensatz für kompakte PERF-Zeilen
struct Stats {
    double ref_ms  = 0.0;  // CPU-Referenzzeit
    double kern_ms = 0.0;  // GPU-Kernelzeit gesamt
    int    centers = 0;    // Anzahl Re-Centerings in diesem Frame
    int    orbitN  = 0;    // Länge des verwendeten Orbits
    double err_max = 0.0;  // gemessene maximale |δz|-Fehlergröße
};

// ---------------------------------------------------------------------------
// Host-Seite: Referenz-Orbit Speicher
// Hinweis: Aktuelle GPU-Pipeline nutzt float2-Orbit (kompakt, schnell).
// Bei späterer Double-Pipeline kann dieser Typ auf double2 angehoben werden.
// ---------------------------------------------------------------------------
struct RefOrbitHost {
    std::vector<float2> z;  // Länge = orbitN (inkl. z0 = 0)
};

// ---------------------------------------------------------------------------
// Device-Seite: Referenz-Orbit Buffer (RAII-light via Methoden)
// ---------------------------------------------------------------------------
struct RefOrbitDevice {
    float2* d_z   = nullptr; // Geräteseitiger Puffer für z⁰_n
    int     count = 0;       // Anzahl Elemente (Orbit-Länge)

    void free() noexcept {
        if (d_z) { cudaFree(d_z); d_z = nullptr; }
        count = 0;
    }
    // Allokiert (oder vergrößert) den Orbit-Puffer auf 'n' Elemente
    void ensure_capacity(int n) {
        if (n <= count) return;
        if (d_z) cudaFree(d_z);
        cudaMalloc(&d_z, static_cast<size_t>(n) * sizeof(float2));
        count = n;
    }
};

// ---------------------------------------------------------------------------
// Host-API: Engine-Lebenszyklus
// ---------------------------------------------------------------------------
struct View {
    int    width  = 0;
    int    height = 0;
    double zoom   = 1.0;          // View-Zoom (wie bisher)
    float2 offset {0.f, 0.f};     // Kamerazentrum (Bildmitte im Fraktalraum)
};

struct Center {
    // Aktuelles Referenzzentrum c0 (Host-seitig in doppelter Präzision verwaltet)
    double c0x = -0.5;
    double c0y =  0.0;
};

class Engine {
public:
    Engine() = default;
    ~Engine() { device_.free(); }

    // Initialisiert View/Config; invalidiert ggf. bestehende Orbitdaten
    void initialize(const View& v, const Config& cfg);

    // Setzt neue Ansicht und markiert Rebuild-Bedarf, wenn c0 zu weit weg ist
    void update_view(const View& v);

    // Haupt-Einstieg: CPU-Referenz-Orbit (falls nötig) + Upload + GPU-Perturbation
    // Schreibt Iterationen/Pixel (wie bisher: d_out + d_iters).
    void render_frame(uchar4* d_out, int* d_iters, int maxIter, Stats& outStats);

    // Zugriff auf internes Zentrum (c0)
    const Center& center() const { return center_; }

private:
    // CPU: Referenz-Orbit neu aufbauen (High-Precision intern, aktuell → float2)
    void build_ref_orbit_cpu(int maxIter, double budgetMs, Stats& stats);

    // GPU: Orbit an Gerät senden
    void upload_ref_orbit_gpu();

    // GPU: Perturbations-Render starten
    void launch_perturbation_kernel(uchar4* d_out, int* d_iters, int maxIter, Stats& stats);

    // Heuristik: Re-Centering nötig?
    bool recenter_required() const;

private:
    Config         cfg_{};
    View           view_{};
    Center         center_{};     // aktuelles c0
    RefOrbitHost   hostOrbit_{};  // Host-Puffer
    RefOrbitDevice device_{};     // Device-Puffer
    bool           needRecenter_ = true; // erster Frame erzwingt Aufbau
};

// ---------------------------------------------------------------------------
// GPU-Schnittstelle (Definition in nacktmull.cu)
// Hinweis: Orbit ist derzeit float2 (GPU-seitig wie oben).
// ---------------------------------------------------------------------------
void launchPerturbationPass(
    uchar4*      d_out,
    int*         d_iters,
    int          w,
    int          h,
    double       zoom,
    float2       offset,
    int          maxIter,
    const float2* d_refOrbit,
    int          orbitCount,
    double       c0x,
    double       c0y,
    double&      outKernelMs  // gefüllte Kernelzeit (ms)
);

// Hilfsfunktion: Pixel → komplexe Zahl (double), Nacktmull-Koordinaten
__host__ __device__ inline double2 pixelToComplexD(
    double px, double py, int w, int h, double spanX, double spanY, double2 off)
{
    double2 r;
    r.x = (px / (w > 0 ? double(w) : 1.0) - 0.5) * spanX + off.x;
    r.y = (py / (h > 0 ? double(h) : 1.0) - 0.5) * spanY + off.y;
    return r;
}

} // namespace Nacktmull
