///// Otter: Zaunkönig [ZK] – PBO-Fences & saubere Ring-Disziplin; Header schlank, keine PCH; Nacktmull-Pullover.
///// Schneefuchs: [ZK] GLsync vorwärts deklariert; Speicher/Buffer exakt; State entkoppelt; MSVC-Align-Warnung lokal gekapselt.
///// Maus: [ZK] Flags klar benannt (pboFence, skipUploadThisFrame); tileSize explizit; Progressive (z,it) mit Cooldown; ASCII-only.
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
struct __GLsync; using GLsync = __GLsync*; // [ZK] GLsync vorwaerts deklariert (keine GL-Header hier)

// CUDA-Stream schlank vorwaerts deklarieren (kein schwerer cuda_runtime*-Include im Header)
struct CUstream_st; using cudaStream_t = CUstream_st*; // Ownership liegt beim RendererState

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

    // 🔍 Kamera (Komplexebene)
    double      zoom = 1.0;
    double2     center{0.0, 0.0};
    double2     pixelScale{0.0, 0.0};

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

    // ➕ Progressive-State (Per-Pixel Resume)
    Hermelin::CudaDeviceBuffer d_stateZ;     // float2[width*height]
    Hermelin::CudaDeviceBuffer d_stateIt;    // int[width*height]
    bool                       progressiveEnabled = true;
    int                        progressiveCooldownFrames = 0;

    // 🎥 OpenGL-Zielpuffer (Interop via CUDA) mit RAII
    static constexpr int kPboRingSize = 3;
    std::array<Hermelin::GLBuffer, kPboRingSize> pboRing;
    int pboIndex = 0;
    inline Hermelin::GLBuffer& currentPBO() { return pboRing[pboIndex]; }
    inline const Hermelin::GLBuffer& currentPBO() const { return pboRing[pboIndex]; }
    inline void advancePboRing() { pboIndex = (pboIndex + 1) % kPboRingSize; }
    Hermelin::GLBuffer tex;

    // 🔒 [ZK] GL-Fences je Slot
    std::array<GLsync, kPboRingSize> pboFence{}; // nullptr = kein Fence gesetzt
    bool skipUploadThisFrame = false;

    // 🕒 Zeitsteuerung pro Frame
    int    frameCount = 0;
    double lastTime   = 0.0;

    // 🌀 Zoom V3 Silk-Lite
    ZoomLogic::ZoomState zoomV3State;

    // 🔥 Overlay-Zustaende
    bool        heatmapOverlayEnabled       = false;
    bool        warzenschweinOverlayEnabled = false;
    std::string warzenschweinText;

    // 🎬 CUDA Streams (Ownership im State) – 4e/4f
    cudaStream_t renderStream = nullptr; // non-blocking
    cudaStream_t copyStream   = nullptr; // non-blocking (Host->GL Copy / Staging)

    // ⏱️ Timings – CUDA + HOST konsolidiert
    struct CudaPhaseTimings {
        bool   valid            = false;
        double mandelbrotTotal  = 0.0;
        double mandelbrotLaunch = 0.0;
        double mandelbrotSync   = 0.0;
        double entropy          = 0.0;
        double contrast         = 0.0;
        double deviceLogFlush   = 0.0;
        double pboMap           = 0.0;
        double uploadMs         = 0.0;
        double overlaysMs       = 0.0;
        double frameTotalMs     = 0.0;
        void resetHostFrame() noexcept { uploadMs = overlaysMs = frameTotalMs = 0.0; }
    };
    CudaPhaseTimings lastTimings;

    // 🧽 Setup & Verwaltung
    RendererState(int w, int h);
    ~RendererState();
    void reset();
    void setupCudaBuffers(int tileSize);
    void resize(int newWidth, int newHeight);
    void invalidateProgressiveState(bool hardReset) noexcept;

private:
    // Stream-Lifecycle
    void createCudaStreamsIfNeeded();       // legt renderStream/copyStream non-blocking an
    void destroyCudaStreamsIfAny() noexcept; // zerstört beide, setzt auf nullptr
};

#if defined(_MSC_VER)
  #pragma warning(pop)
#endif
