// Datei: src/renderer_state.hpp
// Zeilen: 95
// 🐭 Maus-Kommentar: Der Renderer merkt sich nun Entropie, Kontrast, Index und Score (zoomResult) – für Analyse, Visualisierung oder Heatmap. Schneefuchs: „Wer messen will, muss erinnern.“

#pragma once

#include "pch.hpp"  // 🧠 Enthält <cuda_runtime.h>, das float2 definiert – keine eigene Definition mehr nötig!
#include "zoom_logic.hpp"  // 📦 Enthält ZoomResult für Auto-Zoom-Auswertung

class RendererState {
public:
    // 🖼️ Fensterdimensionen
    int width;
    int height;
    GLFWwindow* window = nullptr;  // 🔲 OpenGL-Fensterhandle

    // 🔍 Aktueller Zoom & Bildverschiebung (jetzt double für Präzision)
    double zoom;
    double2 offset;

    // 🧮 Iterationsparameter (für progressive Darstellung)
    int baseIterations;
    int maxIterations;

    // 🎯 Zielwert für Auto-Zoom (wird mit LERP angenähert)
    double2 targetOffset;
    double2 filteredTargetOffset = { 0.0, 0.0 };  // 🎯 Double-präzises geglättetes Ziel

    // 📌 Auto-Zoom-Ziel (geglättet über CUDA-Auswertung)
    float2 smoothedTargetOffset = { 0.0f, 0.0f };
    float smoothedTargetScore = -1.0f;

    // 📈 FPS und Framezeit zur Anzeige im HUD
    float currentFPS = 0.0f;
    float deltaTime = 0.0f;

    // 🧩 Adaptive Tile-Größe + Entropie-/Kontrastspeicher
    int lastTileSize;
    std::vector<float> h_entropy;   // 🔢 Entropie pro Tiles

    // 🔗 CUDA-Puffer (Geräteseite)
    int* d_iterations = nullptr;
    float* d_entropy = nullptr;

    // 🎥 OpenGL-Puffer (direkt im State enthalten)
    unsigned int pbo = 0;  // Pixel Buffer Object
    unsigned int tex = 0;  // Textur-ID für CUDA-Ausgabe

    // 🕒 Frame-Zählung und Zeit für FPS-Berechnung
    int frameCount = 0;
    double lastTime = 0.0;

    // 🔁 Auto-Zoom Status
    bool shouldZoom = false;

    // 🧠 Letzte Ziel-Auswertung als Struktur (für Kontrastanalyse etc.)
    ZoomLogic::ZoomResult zoomResult;

    // 🧠 Letzte gemerkte Werte für Entropieanalyse
    float lastEntropy = 0.0f;
    float lastContrast = 0.0f;
    int   lastIndex = -1;

    // 🆕 Merker für Analyse nach Zielwechsel
    bool justZoomed = false;

    // 📏 Supersampling-Faktor (z. B. 1 = aus, 2 = 2x2, 4 = 4x4 etc.)
    int supersampling = 1;

    // 🔥 Heatmap Overlay aktiv?
    bool overlayEnabled = false;

    // 🧭 Index des zuletzt gewählten Ziel-Tiles (für Zoom-Tracking)
    int lastTileIndex = -1;

    // 🔁 Konstruktor & Methoden zur Zustandspflege
    RendererState(int w, int h);
    void reset();
    void updateOffsetTarget(double2 newOffset);
    void adaptIterationCount();
    void setupCudaBuffers();

    // 🧽 Dynamischer Resize inkl. GPU-Ressourcen
    void resize(int newWidth, int newHeight);
};
