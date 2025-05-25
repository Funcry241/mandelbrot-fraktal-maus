#pragma once

#ifndef __CUDACC__
    #include <boost/multiprecision/cpp_dec_float.hpp>
    using BigFloat = boost::multiprecision::cpp_dec_float_50;
#else
    using BigFloat = double; // Fallback für CUDA (z. B. im mandelbrot.cu)
#endif

#include <iostream>

// -----------------------------------------------------------------------------
// Settings für Fraktal-Rendering
// -----------------------------------------------------------------------------
struct Settings {
    int     width           = 800;    // Fensterbreite
    int     height          = 600;    // Fensterhöhe
    BigFloat offsetX        = 0;      // X-Offset in der komplexen Ebene
    BigFloat offsetY        = 0;      // Y-Offset in der komplexen Ebene
    BigFloat zoom           = 1;      // Zoom-Faktor
    int     sampleStep      = 1;      // Sampling-Schrittweite
    int     maxIter         = 1000;   // Maximale Iterationen
    float   autoZoomPerSec  = 1.1f;   // Automatisches Zoom pro Sekunde (z.B. 10%)
    float   gradThreshold   = 0.5f;   // Grenzwert für adaptives Sampling
    float   panSpeed        = 1.0f;   // Geschwindigkeit für sanftes Panning
};

// -----------------------------------------------------------------------------
// CLI-Parsing & Logging
// -----------------------------------------------------------------------------

void init_cli(int argc, char** argv);
void init_logging();
void cleanup_logging();
