///// Otter: Nacktmull engine header – perturbation/series API; replaces hybrid path; ASCII-only.
///// Schneefuchs: Header-only declarations; /WX-safe; no macro redefs beyond guarded LUCHS_LOG_HOST.
///// Maus: Minimal host+GPU structs; float2 orbit on device; clear lifecycle + stats.
///// Datei: src/nacktmull.hpp

#pragma once

#include <cuda_runtime.h>    // float2, double2, uchar4
#include <cstdint>
#include <cstddef>
#include <vector>

// Forward decl for host logger (project API)
namespace LuchsLogger { void logMessage(const char* file, int line, const char* fmt, ...); }
#ifndef LUCHS_LOG_HOST
#define LUCHS_LOG_HOST(...) ::LuchsLogger::logMessage(__FILE__, __LINE__, __VA_ARGS__)
#endif

namespace Nacktmull {

// ---------------------------------------------------------------------------
// Configuration (host)
// ---------------------------------------------------------------------------
struct Config {
    int     maxOrbitTerms        = 200000; // max reference-orbit length (CPU)
    double  recenterThreshold    = 1e-12;  // |delta c| threshold for re-centering
    double  maxDeltaError        = 1e-10;  // tolerated |delta z| error (heuristic)
    int     maxRecentersPerFrame = 2;      // guard against ping-pong
    double  refOrbitBudgetMs     = 0.0;    // CPU reference budget (0 = adaptive)
};

// Compact telemetry for PERF lines
struct Stats {
    double ref_ms  = 0.0;  // CPU reference time
    double kern_ms = 0.0;  // total GPU kernel time
    int    centers = 0;    // number of re-centerings in this frame
    int    orbitN  = 0;    // length of orbit used
    double err_max = 0.0;  // measured max |delta z|
};

// ---------------------------------------------------------------------------
// Host-side: reference orbit storage
// Current GPU pipeline uses float2 orbit (compact, fast). Can be lifted to
// double2 later if a double pipeline is introduced.
// ---------------------------------------------------------------------------
struct RefOrbitHost {
    std::vector<float2> z;  // length = orbitN (includes z0 = 0)
};

// ---------------------------------------------------------------------------
// Device-side: reference orbit buffer (RAII-light via methods)
// ---------------------------------------------------------------------------
struct RefOrbitDevice {
    float2* d_z   = nullptr; // device buffer for z^0_n
    int     count = 0;       // number of elements (orbit length)

    void free() noexcept {
        if (d_z) { cudaFree(d_z); d_z = nullptr; }
        count = 0;
    }
    // Allocate (or grow) orbit buffer to 'n' elements
    void ensure_capacity(int n) {
        if (n <= count) return;
        if (d_z) cudaFree(d_z);
        cudaMalloc(&d_z, static_cast<size_t>(n) * sizeof(float2));
        count = n;
    }
};

// ---------------------------------------------------------------------------
// Host API: engine lifecycle
// ---------------------------------------------------------------------------
struct View {
    int    width  = 0;
    int    height = 0;
    double zoom   = 1.0;          // view zoom (matches existing semantics)
    float2 offset {0.f, 0.f};     // camera center (image midpoint in fractal space)
};

struct Center {
    // Current reference center c0 (managed on host in double precision)
    double c0x = -0.5;
    double c0y =  0.0;
};

class Engine {
public:
    Engine() = default;
    ~Engine() { device_.free(); }

    // Initialize view/config; invalidates orbit if needed
    void initialize(const View& v, const Config& cfg);

    // Update view and mark rebuild when c0 is too far
    void update_view(const View& v);

    // Main entry: CPU reference orbit (if needed) + upload + GPU perturbation.
    // Writes per-pixel iterations (as before: d_out + d_iters).
    void render_frame(uchar4* d_out, int* d_iters, int maxIter, Stats& outStats);

    // Access current center (c0)
    const Center& center() const { return center_; }

private:
    // CPU: rebuild reference orbit (high-precision internally; float2 output)
    void build_ref_orbit_cpu(int maxIter, double budgetMs, Stats& stats);

    // GPU: send orbit to device
    void upload_ref_orbit_gpu();

    // GPU: launch perturbation render
    void launch_perturbation_kernel(uchar4* d_out, int* d_iters, int maxIter, Stats& stats);

    // Heuristic: is re-centering required?
    bool recenter_required() const;

private:
    Config         cfg_{};
    View           view_{};
    Center         center_{};     // current c0
    RefOrbitHost   hostOrbit_{};  // host buffer
    RefOrbitDevice device_{};     // device buffer
    bool           needRecenter_ = true; // first frame forces build
};

// ---------------------------------------------------------------------------
// GPU interface (definition in nacktmull.cu)
// Orbit is currently float2 (device side as above).
// ---------------------------------------------------------------------------
void launchPerturbationPass(
    uchar4*       d_out,
    int*          d_iters,
    int           w,
    int           h,
    double        zoom,
    float2        offset,
    int           maxIter,
    const float2* d_refOrbit,
    int           orbitCount,
    double        c0x,
    double        c0y,
    double&       outKernelMs  // filled kernel time (ms)
);

// Helper: pixel -> complex (double), Nacktmull coordinates
__host__ __device__ inline double2 pixelToComplexD(
    double px, double py, int w, int h, double spanX, double spanY, double2 off)
{
    double2 r;
    r.x = (px / (w > 0 ? double(w) : 1.0) - 0.5) * spanX + off.x;
    r.y = (py / (h > 0 ? double(h) : 1.0) - 0.5) * spanY + off.y;
    return r;
}

} // namespace Nacktmull
