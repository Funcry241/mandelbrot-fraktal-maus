// Datei: src/progressive.cpp
#include <cstdio>
#include "progressive.hpp"

void resetIterations() {
    currentMaxIter = 100;     // Reset auf Anfang
    std::fprintf(stdout, "[RESET] Iterationen auf %d zurückgesetzt.\n", currentMaxIter);
}
