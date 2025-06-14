// Datei: src/progressive.cu
// 🐭 Maus-Kommentar: Steuerung der progressiven Iterationen für Mandelbrot mit CUDA-managed Speicher

#include <cstdio>
#include <cmath>
#include "progressive.hpp"
#include "settings.hpp"

// -----------------------------------------------------------------------------
// 🔗 CUDA-managed globale Zustände – synchron für CPU & GPU sichtbar
// -----------------------------------------------------------------------------

// 🔢 Aktuelle Obergrenze der Iteration (wird dynamisch erhöht)
__device__ __managed__ int currentMaxIter = Settings::INITIAL_ITERATIONS;

// 🛑 Einmal-Flag, das gesetzt wird, wenn ein Reset durchgeführt wurde
__device__ __managed__ bool justResetFlag = false;

namespace Progressive {

/// 🔁 Setzt Iterationsanzahl zurück auf Initialwert und aktiviert Reset-Flag
void resetIterations() {
    ::currentMaxIter = Settings::INITIAL_ITERATIONS;
    ::justResetFlag = true;

    if (Settings::debugLogging) {
        std::fprintf(stdout, "[RESET] Iteration count reset to %d.\n", ::currentMaxIter);
    }
}

/// 🧾 Gibt aktuellen Iterationswert zurück (wird von Host genutzt)
int getCurrentIterations() {
    return ::currentMaxIter;
}

/// ⏫ Erhöht Iterationen schrittweise (z. B. pro Frame), aber nicht über MAX_ITERATIONS_CAP hinaus
void incrementIterations() {
    if (::currentMaxIter < Settings::MAX_ITERATIONS_CAP) {
        ::currentMaxIter += Settings::ITERATION_STEP;

        // Sicherheit: nicht über Limit hinausgehen
        if (::currentMaxIter > Settings::MAX_ITERATIONS_CAP) {
            ::currentMaxIter = Settings::MAX_ITERATIONS_CAP;
        }

        if (Settings::debugLogging) {
            std::fprintf(stdout, "[UPDATE] Iteration count increased to %d.\n", ::currentMaxIter);
        }
    }
}

/// 🕵️‍♂️ Liefert true, wenn ein Reset stattgefunden hat – danach automatisch wieder false
bool wasJustReset() {
    bool flag = ::justResetFlag;
    ::justResetFlag = false;
    return flag;
}

} // namespace Progressive
