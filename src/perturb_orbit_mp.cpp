///// OtterDream — Replikatoren
///// File: src/perturb_orbit_mp.cpp
///// Purpose: CPU-MP Referenz-Orbit (Loader/Cache) - Stubs
///// Phase: 1 (Orbit-Replikatoren)
///// Hooks: F9 Orbit-Load ; frame_pipeline (optional Preload)
///// Depends: pch.hpp, luchs_log_host.hpp, perturb_orbit_mp.hpp
///// Build: /WX-safe
///// Log-Tags: [REPL/ORBIT]
///// Created: 2025-11-06 (Europe/Berlin)
///// Notes: Später: Parsing- und Versionstags der .bin-Dateien.

#include "pch.hpp"
#include "perturb_orbit_mp.hpp"
#include "luchs_log_host.hpp"

namespace Repl { namespace Orbit {

bool load_orbit_file(const std::string& path) {
    LUCHS_LOG_HOST("[REPL/ORBIT] load-orbit path=%s (stub)", path.c_str());
    return false; // Stub: kein echtes Laden
}

}} // namespace Repl::Orbit
