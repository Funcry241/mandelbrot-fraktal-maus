///// Otter: Einheitliche, klare Struktur – nur aktive Zustaende; Header schlank, keine PCH; Nacktmull-Pullover.
///// Schneefuchs: Speicher/Buffer exakt definiert; State entkoppelt; MSVC-Align-Warnung lokal gekapselt.
///  Maus: tileSize bleibt explizit; Progressive-State (z,it) + Cooldown-Mechanik; ASCII-only.
///// Datei: src/renderer_state.hpp
#pragma once

// Leichte Includes im Header (keine PCH)
#include <vector>
#include <string>
#include <array>
#include <vector_types.h>        // float2/double2 (__align__-Typen → MSVC C4324)
#include "hermelin_buffer.hpp"   // RAII-Wrapper fuer GL/CUDA-Buffer (by value erforderlich)
#include "zoom_logic.hpp"        // ZoomLogic::ZoomState (by-value Member → vollständiger Typ noetig)

// Vorwaertsdeklarationen statt schwerer Header
struct GLFWwindow;

// MSVC: float2/double2 sind __align__-Typen → C4324 (Padding). Lokal und gezielt unterdruecken.
#if defined(_MSC_VER)
  #pragma warning(push)
  #pragma warning(disable : 4324)
#endif

class RendererState {
public:
    // 🖼️ Fenster/Viewport
    int         width  = 0;
    int         height = 0;
    GLFWwindow* window = nullptr;

    // 🔍 Kamera (Komplexebene) — Nacktmull-Pullover: double-präzis
    //   Mapping: c = center + (ix - w/2)*pixelScale.x + i*(iy - h/2)*pixelScale.y
    double      zoom = 1.0;           // skalare Zoomgroesse (unitless)
    double2     center{0.0, 0.0};     // Weltzentrum c0 (double-Genauigkeit)
    double2     pixelScale{0.0, 0.0}; // Delta pro Pixel in Real/Imag (double)

    // 🧮 Iterationsparameter
    int baseIterations = 100;
    int maxIterations  = 1000;

    // 📈 Anzeige/Timing (Frame)
    float  fps       = 0.0f;
    float  deltaTime = 0.0f;

    // 🧩 Analysepuffer (Host)
    int                 lastTileSize = 0;
    std::vector<float>  h_entropy;
    std::vector<float>  h_contrast;

    // 🔗 Analyse/Iteration (Device) mit RAII
    Hermelin::CudaDeviceBuffer d_iterations; // int[width*height]
    Hermelin::CudaDeviceBuffer d_entropy;    // float[numTiles]
    Hermelin::CudaDeviceBuffer d_contrast;   // float[numTiles]

    // ➕ Progressive-State (Per-Pixel Resume) – Keks 4/5
    Hermelin::CudaDeviceBuffer d_stateZ;     // float2[width*height] – letzter z
    Hermelin::CudaDeviceBuffer d_stateIt;    // int   [width*height] – akk. Iterationen
    bool                       progressiveEnabled = true; // Host-Schalter (sanft)
    int                        progressiveCooldownFrames = 0; // 0=aktiv, >0=Pause

    // 🎥 OpenGL-Zielpuffer (Interop via CUDA) mit RAII
    static constexpr int kPboRingSize = 3;
    std::array<Hermelin::GLBuffer, kPboRingSize> pboRing;
    int pboIndex = 0;
    inline Hermelin::GLBuffer& currentPBO() { return pboRing[pboIndex]; }
    inline const Hermelin::GLBuffer& currentPBO() const { return pboRing[pboIndex]; }
    inline void advancePboRing() { pboIndex = (pboIndex + 1) % kPboRingSize; }
    Hermelin::GLBuffer tex;

    // 🕒 Zeitsteuerung pro Frame
    int    frameCount = 0;
    double lastTime   = 0.0;

    // 🌀 Zoom V3 Silk-Lite: Persistenter Zustand (keine Globals)
    ZoomLogic::ZoomState zoomV3State;

    // 🔥 Overlay-Zustaende
    bool        heatmapOverlayEnabled       = false;
    bool        warzenschweinOverlayEnabled = false;
    std::string warzenschweinText;

    // ⏱️ Timings – CUDA + HOST konsolidiert (eine Quelle)
    struct CudaPhaseTimings {
        // CUDA / Interop (gesetzt vom Renderpfad)
        bool   valid            = false;
        double mandelbrotTotal  = 0.0;
        double mandelbrotLaunch = 0.0;
        double mandelbrotSync   = 0.0;
        double entropy          = 0.0;
        double contrast         = 0.0;
        double deviceLogFlush   = 0.0;
        double pboMap           = 0.0;

        // HOST (gesetzt in frame_pipeline)
        double uploadMs         = 0.0;
        double overlaysMs       = 0.0;
        double frameTotalMs     = 0.0;

        void resetHostFrame() noexcept {
            uploadMs     = 0.0;
            overlaysMs   = 0.0;
            frameTotalMs = 0.0;
        }
    };
    CudaPhaseTimings lastTimings;

    // 🧽 Setup & Verwaltung
    RendererState(int w, int h);
    void reset();
    void setupCudaBuffers(int tileSize);
    void resize(int newWidth, int newHeight);

    // 🧯 Progressive-State vorsichtig invalidieren (1-Frame-Cooldown, optional Hard-Reset)
    void invalidateProgressiveState(bool hardReset) noexcept;
};

#if defined(_MSC_VER)
  #pragma warning(pop)
#endif
