///// OtterDream — Replikatoren
///// File: src/perturb_orbit_mp.hpp
///// Purpose: CPU-MP Referenz-Orbit (Loader/Cache) - Stubs
///// Phase: 1 (Orbit-Replikatoren)
///// Hooks: F9 Orbit-Load ; frame_pipeline (optional Preload)
///// Depends: <string>
///// Build: /WX-safe
///// Log-Tags: [REPL/ORBIT]
///// Created: 2025-11-06 (Europe/Berlin)
///// Notes: Implementierung in src/perturb_orbit_mp.cpp; später MPFR/boost::multiprecision.

#pragma once
#include <string>

namespace Repl { namespace Orbit {

    // Lädt ein vorgebackenes Orbit aus assets/orbits/*.bin (Stub).
    // Rückgabe: true wenn erfolgreich (Stub: immer false).
    bool load_orbit_file(const std::string& path);

}} // namespace Repl::Orbit
