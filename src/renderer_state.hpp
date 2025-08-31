///// Otter: Einheitliche, klare Struktur – nur aktive Zustaende; Header schlank, keine PCH; Nacktmull-Pullover.
///// Schneefuchs: Speicher/Buffer exakt definiert; Host-Timings zentral – eine Quelle; /WX-fest; ASCII-only.
///// Maus: tileSize bleibt in Pipelines explizit; hier nur Zustand & Ressourcen; keine versteckten Semantikwechsel.
// Datei: src/renderer_state.hpp

#pragma once

// Leichte Includes im Header (keine PCH)
#include <vector>
#include <string>
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

    // 📌 Referenz-Orbit-Basis (Perturbation)
    double2     orbitCenter{0.0, 0.0};     // Basis fuer Referenz-Orbit (Tile/Frame)
    bool        orbitRebuildRequested = false; // Host-Trigger fuer Orbit-Neuaufbau

    // 🛡️ Precision-Guards (Host-seitig ausgewertet)
    bool        precisionGuardTriggered = false; // (delta_cx/ulp) < Schwelle
    double      precisionGuardRatioLast = 0.0;   // letzte Ratio-Messung
    bool        rebaseRequested         = false; // Center-Rebase anfordern (Ausloeschungsschutz)

    // 🧮 Iterationsparameter
    int baseIterations = 100;   // Startbudget
    int maxIterations  = 1000;  // aktuell verwendete harte Obergrenze

    // 📈 Anzeige/Timing (Frame)
    float  fps       = 0.0f;
    float  deltaTime = 0.0f;

    // 🧩 Analysepuffer (Host)
    int                 lastTileSize = 0;
    std::vector<float>  h_entropy;
    std::vector<float>  h_contrast;

    // 🔗 Analyse/Iteration (Device) mit RAII
    Hermelin::CudaDeviceBuffer d_iterations;
    Hermelin::CudaDeviceBuffer d_entropy;
    Hermelin::CudaDeviceBuffer d_contrast;

    // 🎥 OpenGL-Zielpuffer (Interop via CUDA) mit RAII
    Hermelin::GLBuffer pbo;
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
        double uploadMs         = 0.0; // PBO->Texture + FSQ
        double overlaysMs       = 0.0; // Heatmap + Warzenschwein
        double frameTotalMs     = 0.0; // beginFrame->Ende execute (ohne Swap)

        // Pro-Frame-Reset nur fuer Host-Anteile.
        void resetHostFrame() noexcept {
            uploadMs     = 0.0;
            overlaysMs   = 0.0;
            frameTotalMs = 0.0;
            // CUDA-Messwerte bleiben vom Renderpfad gesetzt.
        }
    };
    CudaPhaseTimings lastTimings;

    // 🧽 Setup & Verwaltung
    RendererState(int w, int h);
    void reset();                             // stellt Initialzustand her
    void setupCudaBuffers(int tileSize);      // allokiert/verifiziert Device-Buffer – tileSize explizit
    void resize(int newWidth, int newHeight); // Fenstergroesse aendern
};

#if defined(_MSC_VER)
  #pragma warning(pop)
#endif
