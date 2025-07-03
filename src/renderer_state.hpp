// Datei: src/renderer_state.hpp
// Zeilen: 99
// 🐭 Maus-Kommentar: Der Renderer merkt sich nun Entropie, Kontrast, Index und Score (zoomResult) – für Analyse, Visualisierung oder Heatmap. 
// Flugente: offset zurück auf float2 für Performance. 
// Schneefuchs: „Wer messen will, muss erinnern.“

#pragma once

#include "pch.hpp"              // 🧠 Enthält <cuda_runtime.h>, float2 etc.
#include "zoom_logic.hpp"       // 📦 ZoomResult für Auto-Zoom-Auswertung

class RendererState {
public:
    // 🖼️ Fensterdimensionen
    int width;
    int height;
    GLFWwindow* window = nullptr;

    // 🔍 Zoom & Bildausschnitt
    double zoom;
    float2 offset;                    // 🦆 Flugente: war double2 → jetzt float2

    // 🧮 Iterationen
    int baseIterations;
    int maxIterations;

    // 🎯 Zielkoordinaten & Glättung
    double2 targetOffset;
    double2 filteredTargetOffset = { 0.0, 0.0 };
    float2 smoothedTargetOffset = { 0.0f, 0.0f };
    float smoothedTargetScore = -1.0f;

    // 📈 Anzeige
    float currentFPS = 0.0f;
    float deltaTime = 0.0f;

    // 🧩 Entropie & Kontrast
    int lastTileSize;
    std::vector<float> h_entropy;    // 🔢 Entropie pro Tile
    std::vector<float> h_contrast;   // 🐼 Kontrast pro Tile

    // 🔗 CUDA-Puffer (Geräteseite)
    int* d_iterations = nullptr;
    float* d_entropy   = nullptr;
    float* d_contrast  = nullptr;    // 🐼 Panda: device-Kontrastdaten

    // 🎥 OpenGL-Puffer
    unsigned int pbo = 0;
    unsigned int tex = 0;

    // 🕒 Zeit
    int frameCount = 0;
    double lastTime = 0.0;

    // 🔁 Auto-Zoom
    bool shouldZoom = false;

    // 🧠 Analyse
    ZoomLogic::ZoomResult zoomResult;
    float lastEntropy = 0.0f;
    float lastContrast = 0.0f;
    int   lastIndex = -1;
    bool justZoomed = false;

    // 📏 Supersampling
    int supersampling = 1;

    // 🔥 Overlay-Steuerung
    bool overlayEnabled = false;

    // 📌 Ziel-Tile-Index
    int lastTileIndex = -1;

    // 🧽 Verwaltung
    RendererState(int w, int h);
    void reset();
    void updateOffsetTarget(double2 newOffset);
    void adaptIterationCount();
    void setupCudaBuffers();
    void resize(int newWidth, int newHeight);
};
