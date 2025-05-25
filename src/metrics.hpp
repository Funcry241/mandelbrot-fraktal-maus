// src/metrics.hpp

#pragma once

// ----------------------------------------------------------------------------
// Metrics für Fraktal-Rendering
// ----------------------------------------------------------------------------
struct Metrics {
    int   fps          = 0;      // Frames pro Sekunde
    float avgNormIter  = 0.0f;   // Durchschnittliche (normierte) Iterationen
    float pctAtMaxIter = 0.0f;   // Prozentualer Anteil an maxIter-Abbrüchen
    float gradDensity  = 0.0f;   // Dichte der Gradientenpunkte
    float detailScore  = 0.0f;   // Score für die Detailqualität
    float detailPerMs  = 0.0f;   // DetailScore pro Millisekunde
};
