///// Otter: One-stop include for Capybara — pulls settings first, then all headers; safe, additive, ASCII-only.
///// Schneefuchs: Does not modify global APIs; optional bridge to project Settings via CAPY_USE_PROJECT_SETTINGS.
///// Maus: Include this before any Capybara usage to centralize knobs (can flip CAPY_DEFAULT_ON here).
///// Datei: src/capybara_prelude.cuh

#pragma once

// --------------------------------------------------------------------------------------
// Purpose
// --------------------------------------------------------------------------------------
// This header ensures a consistent inclusion order for Capybara components:
//   1) Settings/knobs   (capybara_settings.cuh)
//   2) Math & mapping   (capybara_math.cuh, capybara_mapping.cuh)
//   3) Early iter       (capybara_ziter.cuh)
//   4) Integration API  (capybara_integration.cuh, capybara_pixel_iter.cuh)
//   5) Public launcher  (capybara_api.cuh) and selector (capybara_selector.cuh)
//
// Drop this include near the top of a TU (e.g., cuda_interop.cu or your render TU) to
// enable Capybara with one line and keep compile-time flags centralized.

// --------------------------------------------------------------------------------------
// Optional project-wide bridge
// --------------------------------------------------------------------------------------
// If your Settings.hpp defines the following constants, define CAPY_USE_PROJECT_SETTINGS=1
// in your build system or uncomment the define below to map macros to project Settings:
//
//   namespace Settings {
//     inline constexpr bool   capybaraEnabled          = true;   // on/off
//     inline constexpr int    capybaraHiLoEarlyIters   = 64;     // early-iter budget
//     inline constexpr double capybaraRenormRatio      = 3.5527136788005009e-15; // 2^-48
//     inline constexpr bool   capybaraMappingExactStep = true;   // frexp/ldexp mapping
//     inline constexpr bool   capybaraDebugLogging     = false;  // device telemetry
//     inline constexpr int    capybaraLogRate          = 8192;   // rate limiter
//   }
//
// #define CAPY_USE_PROJECT_SETTINGS 1

// --------------------------------------------------------------------------------------
// Optional default routing
// --------------------------------------------------------------------------------------
// To prefer Capybara globally without touching call sites that use the unified selector,
// define CAPY_DEFAULT_ON=1 here or in your build system.
// #define CAPY_DEFAULT_ON 1

// --------------------------------------------------------------------------------------
// Includes (order matters: settings first)
// --------------------------------------------------------------------------------------
#include "capybara_settings.cuh"
#include "capybara_math.cuh"
#include "capybara_mapping.cuh"
#include "capybara_ziter.cuh"
#include "capybara_integration.cuh"
#include "capybara_pixel_iter.cuh"
#include "capybara_api.cuh"
#include "capybara_selector.cuh"

// --------------------------------------------------------------------------------------
// Sanity checks (compile-time)
// --------------------------------------------------------------------------------------
#if (CAPY_ENABLED != 0)
  static_assert(CAPY_EARLY_ITERS >= 0, "CAPY_EARLY_ITERS must be >= 0");
  static_assert(CAPY_RENORM_RATIO > 0.0, "CAPY_RENORM_RATIO must be > 0");
#endif

// --------------------------------------------------------------------------------------
// Usage sketch (no code here, just reference):
// --------------------------------------------------------------------------------------
// In your render TU:
//   #include "capybara_prelude.cuh"
//   ...
//   // Prepare iterations buffer d_it, w, h, cx, cy, stepX, stepY, maxIter, stream
//   launch_mandelbrot_select(d_it, w, h, cx, cy, stepX, stepY, maxIter, stream, /*useCapybara=*/true);
//
// Or keep useCapybara=false and set CAPY_DEFAULT_ON=1 above to switch globally.
