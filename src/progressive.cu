// Datei: src/progressive.cu

#include <cstdio>
#include <cmath>
#include "progressive.hpp"
#include "settings.hpp"

// 🐾 Device-Managed globale Variablen
__device__ __managed__ int currentMaxIter = Settings::INITIAL_ITERATIONS;
__device__ __managed__ bool justResetFlag = false;

namespace Progressive {

void resetIterations() {
    ::currentMaxIter = Settings::INITIAL_ITERATIONS;
    ::justResetFlag = true;
    if (Settings::debugLogging) {
        std::fprintf(stdout, "[RESET] Iterations reset to %d.\n", ::currentMaxIter);
    }
}

int getCurrentIterations() {
    return ::currentMaxIter;
}

void incrementIterations() {
    if (::currentMaxIter < Settings::MAX_ITERATIONS_CAP) {
        ::currentMaxIter += Settings::ITERATION_STEP;
        if (::currentMaxIter > Settings::MAX_ITERATIONS_CAP) {
            ::currentMaxIter = Settings::MAX_ITERATIONS_CAP;
        }
        if (Settings::debugLogging) {
            std::fprintf(stdout, "[UPDATE] Iterations increased to %d.\n", ::currentMaxIter);
        }
    }
}

bool wasJustReset() {
    bool flag = ::justResetFlag;
    ::justResetFlag = false;
    return flag;
}

} // namespace Progressive
