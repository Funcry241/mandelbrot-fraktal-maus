#include <cstdio>
#include <cmath>        // 🐭 Für logf()
#include "progressive.hpp"
#include "settings.hpp" // 🐭 Damit Settings::debugLogging bekannt ist

// Definition — EINMAL
__device__ __managed__ int currentMaxIter = 100;
__device__ __managed__ bool justResetFlag = false;

constexpr int initialIterations = 100;
constexpr int maxIterationsCap  = 5000; // 🐭 Nicht unendlich wachsen lassen
constexpr int iterationStep     = 5;    // 🐭 Schrittweise Erhöhung

void resetIterations() {
    currentMaxIter = initialIterations;
    justResetFlag = true; // 🐭 Reset-Flag setzen
    if (Settings::debugLogging) {
        std::fprintf(stdout, "[RESET] Iterations reset to %d.\n", currentMaxIter);
    }
}

int getCurrentIterations() {
    // 🐭 Pro Frame leicht steigern für mehr Details bei tieferem Zoom
    if (currentMaxIter < maxIterationsCap) {
        currentMaxIter += iterationStep;
        if (currentMaxIter > maxIterationsCap) currentMaxIter = maxIterationsCap;
    }
    return currentMaxIter;
}

bool wasJustReset() {
    bool flag = justResetFlag;
    justResetFlag = false; // 🐭 Einmalig liefern
    return flag;
}
