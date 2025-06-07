#include <cstdio>
#include <cmath>        // 🐭 Für logf()
#include "progressive.hpp"
#include "settings.hpp" // 🐭 Settings enthält alles Nötige

// Definition — EINMAL
__device__ __managed__ int currentMaxIter = Settings::INITIAL_ITERATIONS;
__device__ __managed__ bool justResetFlag = false;

void resetIterations() {
    currentMaxIter = Settings::INITIAL_ITERATIONS;
    justResetFlag = true; // 🐭 Reset-Flag setzen
    if (Settings::debugLogging) {
        std::fprintf(stdout, "[RESET] Iterations reset to %d.\n", currentMaxIter);
    }
}

int getCurrentIterations() {
    return currentMaxIter;
}

void incrementIterations() {
    // 🐭 Iterationen explizit erhöhen — kontrollierter
    if (currentMaxIter < Settings::MAX_ITERATIONS_CAP) {
        currentMaxIter += Settings::ITERATION_STEP;
        if (currentMaxIter > Settings::MAX_ITERATIONS_CAP) {
            currentMaxIter = Settings::MAX_ITERATIONS_CAP;
        }
        if (Settings::debugLogging) {
            std::fprintf(stdout, "[UPDATE] Iterations increased to %d.\n", currentMaxIter);
        }
    }
}
