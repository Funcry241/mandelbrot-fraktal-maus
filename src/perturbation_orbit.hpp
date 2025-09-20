///// Otter: Deterministic host-side reference-orbit builder; double-precision only.
///// Schneefuchs: Header-only API; no hidden defaults; single source of truth is the caller's params.
// /// Maus: ASCII-only sanity logging in .cpp; this header introduces no logging.
// /// Datei: src/perturbation_orbit.hpp

#pragma once

#include <vector>          // std::vector
#include <vector_types.h>  // double2
#include <cstddef>         // std::size_t
#include <cstdint>         // fixed-width ints

// buildReferenceOrbit
// Purpose:
//   Build the Mandelbrot reference orbit z_{n+1} = z_n^2 + c in double precision,
//   starting at z_0 = 0, producing up to `maxLen` samples in `out`.
// Determinism:
//   Fully deterministic given identical (c, maxLen, segSize). No randomness.
// Logging:
//   Implementation may emit short ASCII sanity lines when enabled globally;
//   this declaration adds no logging side effects.
// Parameters:
//   c        : reference complex parameter (double2: x=Re(c), y=Im(c))
//   maxLen   : upper bound for the number of orbit samples to generate (n >= 1)
//   segSize  : segment size used for upload/streaming; echoed for telemetry
//   out      : output vector to be filled with z_n samples (z_1..z_len), capacity will be managed
//   len      : set to the final number of valid samples written to `out` (0 <= len <= maxLen)
// Contracts:
//   - `maxLen > 0` and `segSize > 0` expected by the implementation.
//   - `out` will be resized to `len` exactly on return.
//   - No device interaction; host-only computation in double precision.
// Exceptions:
//   - May throw std::bad_alloc if allocation/reserve fails.
// Notes:
//   - The implementation is provided in src/perturbation_orbit.cpp.
void buildReferenceOrbit(const double2 c,
                         int maxLen,
                         int segSize,
                         std::vector<double2>& out,
                         int& len);
