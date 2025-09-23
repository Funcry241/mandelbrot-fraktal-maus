///// Otter: Zaunkönig [ZK] – PBO-Fences & saubere Ring-Disziplin; Header schlank, keine PCH; Capybara Single-Path.
///// Schneefuchs: EC/Wrapper entfernt – Felder als Legacy markiert; GLsync vorwärts deklariert; /WX-fest; State entkoppelt.
///// Maus: Klare Flags (pboFence, skipUploadThisFrame); tileSize explizit; Progressive (z,it) mit Cooldown; ASCII-only.
///// Datei: src/renderer_state.hpp

#pragma once

// Leichte Includes im Header (keine PCH)
#include <vector>
#include <string>
#include <array>
#include <vector_types.h>        // float2/double2 (__align__-Typen -> MSVC C4324)
#include "hermelin_buffer.hpp"   // RAII-Wrapper fuer GL/CUDA-Buffer (by value erforderlich)
#include "zoom_logic.hpp"        // ZoomLogic::ZoomState (by-value Member -> vollständiger Typ nötig)

// Vorwaertsdeklarationen statt schwerer Header
struct GLFWwindow;
struct __GLsync; using GLsync = __GLsync*; // [ZK] GLsync vorwärts deklariert (keine GL-Header hier)

// CUDA-Primitive schlank vorwärts deklarieren (kein cuda_runtime*-Include im Header)
struct CUstream_st; using cudaStream_t = CUstream_st*; // Ownership liegt beim RendererState
struct CUevent_st;  using cudaEvent_t  = CUevent_st*;  // Events für Render-Ketten

// MSVC: float2/double2 sind __align__-Typen -> C4324 (Padding). Lokal und gezielt unterdrücken.
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

    // 🧩 Analyse/Overlay (Host) — EC-Pfad aktuell deaktiviert.
    // Diese Felder bleiben als Legacy-Placeholder erhalten, damit optionale Overlays/HUD kompilieren.
    int                 lastTileSize = 0;
    std::vector<float>  h_entropy;         // legacy/overlay (leer im aktiven Pfad)
    std::vector<float>  h_contrast;        // legacy/overlay (leer im aktiven Pfad)
    bool                h_entropyPinned  = false; // legacy/no-op
    bool                h_contrastPinned = false; // legacy/no-op

    // 🔗 GPU-Puffer (RAII)
    // Iterationspuffer ist aktiv; EC-Puffer bleiben als Legacy-Platzhalter erhalten (No-Op im aktiven Pfad).
    Hermelin::CudaDeviceBuffer d_iterations; // uint16_t[width*height]
    Hermelin::CudaDeviceBuffer d_entropy;    // float[numTiles]   (legacy/overlay)
    Hermelin::CudaDeviceBuffer d_contrast;   // float[numTiles]   (legacy/overlay)

    // ➕ Progressive-State (Per-Pixel Resume)
    Hermelin::CudaDeviceBuffer d_stateZ;     // float2[width*height]
    Hermelin::CudaDeviceBuffer d_stateIt;    // uint16_t[width*height]
    bool                       progressiveEnabled = true;
    int                        progressiveCooldownFrames = 0;

    // 🎥 OpenGL-Zielpuffer (Interop via CUDA) mit RAII
    // Spiegel von Settings::pboRingSize (numerisch, um Header entkoppelt zu halten).
    // Konsistenz wird in TU(s) via static_assert geprüft.
    static constexpr int kPboRingSize = 8; // <— an Settings::pboRingSize angleichen

    std::array<Hermelin::GLBuffer, kPboRingSize> pboRing{};
    int pboIndex = 0;
    inline Hermelin::GLBuffer&       currentPBO()       { return pboRing[pboIndex]; }
    inline const Hermelin::GLBuffer& currentPBO() const { return pboRing[pboIndex]; }
    inline void advancePboRing() { pboIndex = (pboIndex + 1) % kPboRingSize; }
    Hermelin::GLBuffer tex;

    // 🔒 [ZK] GL-Fences je Slot
    std::array<GLsync, kPboRingSize> pboFence{}; // nullptr = kein Fence gesetzt
    bool skipUploadThisFrame = false;

    // 📊 Ring-Statistik (LOG-6)
    std::array<unsigned, kPboRingSize> ringUse{}; // pro Slot Nutzung
    unsigned ringSkip = 0;                        // Anzahl „skip upload this frame“

    // 🕒 Zeitsteuerung pro Frame
    int    frameCount = 0;
    double lastTime   = 0.0;

    // 🌀 Zoom V3 Silk-Lite
    ZoomLogic::ZoomState zoomV3State;

    // 🔥 Overlay-Zustaende
    bool        heatmapOverlayEnabled       = false;
    bool        warzenschweinOverlayEnabled = false;
    std::string warzenschweinText;

    // 🎬 CUDA Streams (Ownership im State) – non-blocking
    cudaStream_t renderStream = nullptr;
    cudaStream_t copyStream   = nullptr;

    // 🎯 CUDA Events zur asynchronen Verkettung
    // evEcDone wurde früher für EC benutzt; im Render-only Pfad wird es nach dem Rendern aufgezeichnet,
    // damit nachgelagerte Stufen (optional) warten können.
    cudaEvent_t  evEcDone   = nullptr;
    cudaEvent_t  evCopyDone = nullptr; // optional: D->H-Transfers fertig (derzeit kaum genutzt)

    // ⏱️ Timings – CUDA + HOST konsolidiert
    struct CudaPhaseTimings {
        bool   valid            = false;
        double mandelbrotTotal  = 0.0;
        double mandelbrotLaunch = 0.0;
        double mandelbrotSync   = 0.0;
        double entropy          = 0.0; // legacy
        double contrast         = 0.0; // legacy
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

private:
    // Stream-/Event-Lifecycle
    void createCudaStreamsIfNeeded();         // legt renderStream/copyStream non-blocking an
    void destroyCudaStreamsIfAny() noexcept;  // zerstört beide, setzt auf nullptr
    void createCudaEventsIfNeeded();          // legt evEcDone/evCopyDone (DisableTiming) an
    void destroyCudaEventsIfAny() noexcept;   // zerstört Events, setzt auf nullptr

    // Legacy-No-Op Hooks (EC deaktiviert)
    void ensureHostPinnedForAnalysis();       // no-op
    void unpinHostAnalysisIfAny() noexcept;   // no-op
};

#if defined(_MSC_VER)
  #pragma warning(pop)
#endif
