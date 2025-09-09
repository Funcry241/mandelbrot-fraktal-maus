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

// ---------------------------------------------------
// Public config/state
// ---------------------------------------------------
struct Params {
    double centerX = -0.743643887037151;  // classic deep-zoom area
    double centerY =  0.131825904205330;
    double scale   =  1.0;
    int    maxIter = 1000;
};

struct Stats {
    int iterationsDone = 0;
    int pixelsProcessed = 0;
};

// Device orbit data (minimal)
struct Orbit {
    float2 z0;
    float2 c;
};

// Host-owned buffers (example layout)
struct Buffers {
    std::vector<int> iterations;
    int width = 0;
    int height = 0;
};

// API
void init(const Params& p);
void shutdown();
void render(const Params& p, Buffers& out);

} // namespace Nacktmull
