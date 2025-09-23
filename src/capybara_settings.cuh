///// Otter: Capybara-Knobs (enable, early iters, renorm ratio, logging) – header-only, additive.
///// Schneefuchs: Sichere Defaults via #ifndef; optional Bridge zu Settings per CAPY_USE_PROJECT_SETTINGS.
///// Maus: Vor Capybara-Headern einbinden; ASCII-only; keine ABI-Änderung.
///// Datei: src/capybara_settings.cuh

#pragma once
#include <stdint.h>

// Hinweis:
// - Standard-Defaults unten (#ifndef). Mit -D… am Build umdefinierbar.
// - Wenn CAPY_USE_PROJECT_SETTINGS=1 gesetzt ist, werden Werte aus Settings.hpp
//   übernommen (siehe Mapping im unteren Block).

// --------------------------- Defaults (übersteuerbar) -------------------------
#ifndef CAPY_ENABLED
#define CAPY_ENABLED 1              // 1: Capybara aktiv; 0: klassischer Pfad
#endif

#ifndef CAPY_EARLY_ITERS
#define CAPY_EARLY_ITERS 64         // Hi/Lo-Iterationen vor Übergang auf classic
#endif

#ifndef CAPY_RENORM_RATIO
#define CAPY_RENORM_RATIO 3.5527136788005009e-15 // 2^-48: fold lo->hi bei |lo| > ratio*|hi|
#endif

#ifndef CAPY_MAPPING_EXACT_STEP
#define CAPY_MAPPING_EXACT_STEP 1   // 1: frexp/ldexp-Schritt; 0: einfache Multiplikation
#endif

#ifndef CAPY_DEBUG_LOGGING
#define CAPY_DEBUG_LOGGING 0        // 1: Device-Logs (rate-limitiert), 0: aus
#endif

#ifndef CAPY_LOG_RATE
#define CAPY_LOG_RATE 8192          // jede N-te passende Log-Zeile; 0: keine Logs
#endif

// --------------------------- Bridge zu Projekt-Settings -----------------------
#if defined(CAPY_USE_PROJECT_SETTINGS) && (CAPY_USE_PROJECT_SETTINGS != 0)
  #include "settings.hpp"
  #undef  CAPY_ENABLED
  #define CAPY_ENABLED              (Settings::capybaraEnabled ? 1 : 0)

  #undef  CAPY_EARLY_ITERS
  #define CAPY_EARLY_ITERS          (Settings::capybaraHiLoEarlyIters)

  #undef  CAPY_RENORM_RATIO
  #define CAPY_RENORM_RATIO         (Settings::capybaraRenormLoRatio)

  #undef  CAPY_MAPPING_EXACT_STEP
  #define CAPY_MAPPING_EXACT_STEP   (Settings::capybaraMappingExactStep ? 1 : 0)

  #undef  CAPY_DEBUG_LOGGING
  #define CAPY_DEBUG_LOGGING        (Settings::capybaraDebugLogging ? 1 : 0)

  #undef  CAPY_LOG_RATE
  #define CAPY_LOG_RATE             (Settings::capybaraLogRate)
#endif

// --------------------------- Sanity-Clamps (Compile-Time) ---------------------
#if (CAPY_EARLY_ITERS < 0)
  #undef  CAPY_EARLY_ITERS
  #define CAPY_EARLY_ITERS 0
#endif

#if (CAPY_LOG_RATE < 0)
  #undef  CAPY_LOG_RATE
  #define CAPY_LOG_RATE 0
#endif

// Notizen:
// - Escape-Radius bleibt in der Implementierung fix (r^2 = 4.0).
// - Bei CAPY_DEBUG_LOGGING=1 auf ausreichenden Device-Log-Puffer achten.
