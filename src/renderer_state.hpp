// 🦦 Otter: Einheitliche, klare Struktur – nur aktive Zustände. (Bezug zu Otter)
// 🦊 Schneefuchs: Speicher & Buffer exakt definiert, Host-Timings zentral – eine Quelle. (Bezug zu Schneefuchs)
// 🐜 Rote Ameise: tileSize explizit in Pipelines, hier nur Zustand & Ressourcen.

#pragma once

#include "pch.hpp"
#include "zoom_logic.hpp"        // Zoom V2: ZoomState
#include <vector>
#include <string>                // HUD-Text
#include "hermelin_buffer.hpp"   // RAII-Wrapper
#include <vector_types.h>        // float2

class RendererState {
public:
    // 🖼️ Fensterdimensionen (OpenGL-Viewport & Framebuffer-Größe)
    int width;
    int height;
    GLFWwindow* window = nullptr;

    // 🔍 Kamera (Fraktalraum)
    double zoom = 1.0;
    float2 offset = { 0.0f, 0.0f };

    // 🧮 Iterationsparameter
    int baseIterations = 100;   // Ausgangswert
    int maxIterations  = 1000;  // aktuell verwendeter Maximalwert

    // 📈 Anzeige-Feedback
    float fps = 0.0f;
    float deltaTime  = 0.0f;

    // 🧩 Analysepuffer (Host)
    int lastTileSize = 0;
    std::vector<float> h_entropy;
    std::vector<float> h_contrast;

    // 🔗 Analysepuffer (Device) mit RAII
    Hermelin::CudaDeviceBuffer d_iterations;
    Hermelin::CudaDeviceBuffer d_entropy;
    Hermelin::CudaDeviceBuffer d_contrast;

    // 🎥 OpenGL-Zielpuffer (Interop via CUDA) mit RAII
    Hermelin::GLBuffer pbo;
    Hermelin::GLBuffer tex;

    // 🕒 Zeitsteuerung pro Frame
    int frameCount = 0;
    double lastTime = 0.0;

    // 🌀 Zoom V2: Persistenter Zustand (keine Globals)
    ZoomLogic::ZoomState zoomV2State;

    // 🔥 Heatmap-/HUD-Overlay-Zustand
    bool heatmapOverlayEnabled = false;
    bool warzenschweinOverlayEnabled = false;

    // 📝 HUD-Text
    std::string warzenschweinText;

    // ⏱️ Timings – CUDA + HOST konsolidiert (eine Quelle)
    struct CudaPhaseTimings {
        // CUDA / Interop (gesetzt von renderCudaFrame)
        bool   valid = false;
        double mandelbrotTotal  = 0.0;
        double mandelbrotLaunch = 0.0;
        double mandelbrotSync   = 0.0;
        double entropy          = 0.0;
        double contrast         = 0.0;
        double deviceLogFlush   = 0.0;
        double pboMap           = 0.0;

        // HOST (gesetzt in frame_pipeline)
        double uploadMs         = 0.0; // PBO->Texture + FSQ
        double overlaysMs       = 0.0; // Heatmap + Warzenschwein
        double frameTotalMs     = 0.0; // beginFrame->Ende execute (ohne Swap)

        // 🐑 Schneefuchs: Pro-Frame-Reset nur für Host-Anteile. (Bezug zu Schneefuchs)
        void resetHostFrame() noexcept;
    };
    CudaPhaseTimings lastTimings;

    // 🧽 Setup & Verwaltung
    RendererState(int w, int h);
    void reset();                             // stellt Initialzustand her
    void setupCudaBuffers(int tileSize);      // allokiert/verifiziert Device-Buffer – tileSize explizit (🐜)
    void resize(int newWidth, int newHeight); // Fenstergröße ändern
};
