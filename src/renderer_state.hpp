// Datei: src/renderer_state.hpp
// Zeilen: 82
// 🐭 Maus-Kommentar: Der Renderer merkt sich nun den letzten Entropiewert, Kontrast und Index – nötig für relative Zielwahl im Auto-Zoom. Schneefuchs: „Ohne Erinnerung keine Richtung.“

#pragma once

#include "pch.hpp"  // 🧠 Enthält <cuda_runtime.h>, das float2 definiert – keine eigene Definition mehr nötig!

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

    // 🧩 Adaptive Tile-Größe + Entropie-Auswertung
    int lastTileSize;
    std::vector<float> h_entropy;

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

    // 🧠 Letzte Ziel-Auswertung (für relative Analyse)
    int lastIndex = -1;
    float lastEntropy = 0.0f;
    float lastContrast = 0.0f;

    // 🔁 Konstruktor & Methoden zur Zustandspflege
    RendererState(int w, int h);
    void reset();
    void updateOffsetTarget(double2 newOffset);
    void adaptIterationCount();
    void setupCudaBuffers();

    // 🧽 Dynamischer Resize inkl. GPU-Ressourcen
    void resize(int newWidth, int newHeight);
};
