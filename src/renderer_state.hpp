// Datei: src/renderer_state.hpp
// 🦦 Otter: Einheitlich in allen Forward-Deklarationen. Keine strukturelle Überraschung.
// 🦊 Schneefuchs: Speicher & Buffer exakt definiert, feingliedrig und logisch.

#pragma once

#include "pch.hpp"             // CUDA + float2/GLFW
#include "zoom_logic.hpp"      // ZoomResult für Auto-Zoom
#include <vector>

class RendererState {
public:
    // 🖼️ Fensterdimensionen (OpenGL-Viewport & Framebuffer-Größe)
    int width;
    int height;
    GLFWwindow* window = nullptr;

    // 🔍 Zoomfaktor & aktueller Fraktal-Ausschnitt (in Weltkoordinaten)
    double zoom = 1.0;
    float2 offset = { 0.0f, 0.0f };

    // 🧮 Iterationsparameter
    int baseIterations = 100;  // Ausgangswert
    int maxIterations  = 1000; // aktuell verwendeter Maximalwert

    // 🎯 Auto-Zoom Zielkoordinaten
    float2 targetOffset         = { 0.0f, 0.0f };   // analysiertes Ziel
    float2 filteredTargetOffset = { 0.0f, 0.0f };   // geglättetes Ziel
    float2 smoothedTargetOffset = { 0.0f, 0.0f };   // LERP-Interpoliertes Ziel
    float  smoothedTargetScore  = -1.0f;            // Entropie-Score des Zieltiles (wird geglättet)

    // 📈 Anzeige-Feedback
    float fps = 0.0f;
    float deltaTime  = 0.0f;

    // 🧩 Analysepuffer (Host)
    int lastTileSize = 0;
    std::vector<float> h_entropy;
    std::vector<float> h_contrast;

    // 🔗 Analysepuffer (Device)
    int*   d_iterations = nullptr;
    float* d_entropy    = nullptr;
    float* d_contrast   = nullptr;

    // 🎥 OpenGL-Zielpuffer (Interop via CUDA)
    unsigned int pbo = 0;  // Pixel Buffer Object
    unsigned int tex = 0;  // Texture (GL)

    // 🕒 Zeitsteuerung pro Frame
    int frameCount = 0;
    double lastTime = 0.0;

    // 🧠 Letztes Ergebnis der Zielanalyse (persistenter Zustand)
    ZoomLogic::ZoomResult zoomResult;
    float lastEntropy  = 0.0f;
    float lastContrast = 0.0f;
    bool justZoomed    = false;

    // 🔥 Heatmap-Overlay-Zustand
    bool heatmapOverlayEnabled = false;

    // HUD-Overlay-Zustand
    bool warzenschweinOverlayEnabled = false;

    // 🧽 Setup & Verwaltung
    RendererState(int w, int h);
    void reset();                             // stellt Initialzustand her
    void setupCudaBuffers();                  // allokiert Device-Buffer
    void resize(int newWidth, int newHeight); // Fenstergröße ändern
};
