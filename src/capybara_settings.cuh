///// Otter: Central knobs for Capybara (enable, early iters, renorm ratio, logging cadence) — header-only, additive.
/// //// Schneefuchs: Safe defaults via #ifndef; optional bridge to project Settings via CAPY_USE_PROJECT_SETTINGS.
/// //// Maus: Include this before other Capybara headers to override defaults; ASCII-only; one path.
/// //// Datei: src/capybara_settings.cuh

#pragma once
#include <stdint.h>

// --------------------------------------------------------------------------------------
// HOW TO USE
// --------------------------------------------------------------------------------------
// 1) Quick start (no project Settings changes):
//    - Include this file *before* any other Capybara headers in your TU(s).
//    - Optionally set preprocessor defs in your build to override defaults, e.g.:
//        -DCAPY_ENABLED=1 -DCAPY_EARLY_ITERS=64 -DCAPY_DEBUG_LOGGING=0 -DCAPY_LOG_RATE=8192
//
// 2) Integrate with your project's Settings (preferred long-term):
//    - Define CAPY_USE_PROJECT_SETTINGS=1 at compile time, *and*
//    - Provide the following statics in your Settings header (names are suggestions):
//        namespace Settings {
//          inline constexpr bool  capybaraEnabled          = true;     // on/off
//          inline constexpr int   capybaraHiLoEarlyIters   = 64;       // budget for Hi/Lo early phase
//          inline constexpr double capybaraRenormRatio     = 3.5527136788005009e-15; // 2^-48
//          inline constexpr bool  capybaraMappingExactStep = true;     // use frexp/ldexp path
//          inline constexpr bool  capybaraDebugLogging     = false;    // device telemetry
//          inline constexpr int   capybaraLogRate          = 8192;     // rate limiter
//        }
//    - These values will override the macros below when CAPY_USE_PROJECT_SETTINGS is defined.
//
// 3) Scope & guarantees:
//    - Pure compile-time knobs; no ABI/API break.
//    - If you do nothing, safe defaults apply (Capybara on, 64 early iters, conservative renorm).
// --------------------------------------------------------------------------------------

// --------------------------- Base defaults (overridable) ----------------------
#ifndef CAPY_ENABLED
#define CAPY_ENABLED 1              // 1: enable Capybara early-phase; 0: classic only
#endif

#ifndef CAPY_EARLY_ITERS
#define CAPY_EARLY_ITERS 64         // number of Hi/Lo iterations before handing off to classic
#endif

#ifndef CAPY_RENORM_RATIO
#define CAPY_RENORM_RATIO 3.5527136788005009e-15 // 2^-48: fold lo->hi when |lo| > ratio*|hi|
#endif

#ifndef CAPY_MAPPING_EXACT_STEP
#define CAPY_MAPPING_EXACT_STEP 1   // 1: frexp/ldexp exact step scaling; 0: simple multiply
#endif

#ifndef CAPY_DEBUG_LOGGING
#define CAPY_DEBUG_LOGGING 0        // 1: enable device ASCII one-liners (rate-limited), 0: off
#endif

#ifndef CAPY_LOG_RATE
#define CAPY_LOG_RATE 8192          // log every Nth matching event (gid/iter keyed). 0 disables.
#endif

// --------------------------- Bridge to project Settings -----------------------
#if defined(CAPY_USE_PROJECT_SETTINGS) && (CAPY_USE_PROJECT_SETTINGS != 0)
  #include "settings.hpp"
  // Map macros to project-level constants if present. We intentionally do not
  // try to detect symbol existence; define these in Settings to opt-in.
  #undef  CAPY_ENABLED
  #define CAPY_ENABLED              (Settings::capybaraEnabled ? 1 : 0)

  #undef  CAPY_EARLY_ITERS
  #define CAPY_EARLY_ITERS          (Settings::capybaraHiLoEarlyIters)

  #undef  CAPY_RENORM_RATIO
  #define CAPY_RENORM_RATIO         (Settings::capybaraRenormRatio)

  #undef  CAPY_MAPPING_EXACT_STEP
  #define CAPY_MAPPING_EXACT_STEP   (Settings::capybaraMappingExactStep ? 1 : 0)

  #undef  CAPY_DEBUG_LOGGING
  #define CAPY_DEBUG_LOGGING        (Settings::capybaraDebugLogging ? 1 : 0)

  #undef  CAPY_LOG_RATE
  #define CAPY_LOG_RATE             (Settings::capybaraLogRate)
#endif

// --------------------------- Sanity clamps (compile-time) ---------------------
#if (CAPY_EARLY_ITERS < 0)
  #undef  CAPY_EARLY_ITERS
  #define CAPY_EARLY_ITERS 0
#endif

#if (CAPY_LOG_RATE < 0)
  #undef  CAPY_LOG_RATE
  #define CAPY_LOG_RATE 0
#endif

// Notes:
// - Escape radius remains fixed in the implementation (r^2 = 4.0) for parity.
// - If you enable CAPY_DEBUG_LOGGING=1, ensure device log buffer capacity is sufficient
//   and rate-limit (CAPY_LOG_RATE) to avoid flooding.
