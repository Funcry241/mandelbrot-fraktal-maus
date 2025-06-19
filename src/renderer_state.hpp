#pragma once

#include "pch.hpp"  // 🧠 Enthält <cuda_runtime.h>, das float2 definiert – keine eigene Definition mehr nötig!

class RendererState {
public:
    // 🖼️ Fensterdimensionen
    int width;
    int height;
    GLFWwindow* window = nullptr;  // 🔲 OpenGL-Fensterhandle

    // 🔍 Aktueller Zoom & Bildverschiebung
    float zoom;
    float2 offset;

    // 🧮 Iterationsparameter (für progressive Darstellung)
    int baseIterations;
    int maxIterations;

    // 🎯 Zielwerte für Auto-Zoom (werden mit LERP angenähert)
    float2 targetOffset;
    float targetZoom;

    // 🧈 Zwischengespeicherte weichgeglättete Werte (smoothed Lerp)
    float2 smoothedOffset;
    float smoothedZoom;

    // 📈 FPS und Framezeit zur Anzeige im HUD
    float currentFPS = 0.0f;   // 🆕 explizit initialisiert
    float deltaTime = 0.0f;    // 🆕 explizit initialisiert

    // 🧩 Adaptive Tile-Größe + Entropie-Auswertung
    int lastTileSize;
    std::vector<float> h_entropy;

    // 🔗 CUDA-Puffer (Geräteseite)
    int* d_iterations = nullptr;
    float* d_entropy = nullptr;

    // 🎥 OpenGL-Puffer (neu: direkt statt über .resources)
    unsigned int pbo = 0;
    unsigned int tex = 0;

    // 🕒 Frame-Zählung und Zeit für FPS-Berechnung
    int frameCount = 0;
    double lastTime = 0.0;
    float lastFrameTime = 0.0f;

    // 🔁 Auto-Zoom Status
    bool shouldZoom = false;

    // 🔁 Konstruktor & Methoden zur Zustandspflege
    RendererState(int w, int h);
    void reset();
    void updateZoomTarget(float newZoom);
    void updateOffsetTarget(float2 newOffset);
    void applyLerpStep();
    void adaptIterationCount();
};

// 🧰 Initialisiert OpenGL-Textur, PBO, registriert bei CUDA und allokiert GPU-Buffer
void initResources(RendererState& state);
