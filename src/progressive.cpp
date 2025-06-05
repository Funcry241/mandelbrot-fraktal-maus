// Datei: src/progressive.cpp
// 🐭 progressive.cpp – Implementierung für progressives Mandelbrot-Rendering

#include <cstdio>

// Initiale Iterationsanzahl
int currentMaxIter = 100;     // Startwert
const int iterStep = 50;      // Erhöhung pro Frame
const int iterMax  = 5000;    // Maximale Iterationsanzahl

void resetIterations() {
    currentMaxIter = 100;     // Reset auf Anfang
    std::fprintf(stdout, "[RESET] Iterationen auf %d zurückgesetzt.\n", currentMaxIter);
}
