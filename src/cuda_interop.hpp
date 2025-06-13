#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h> // 🐭 Für Tasteneingaben (Space: Auto-Zoom, P: Pause)

// 🐭 Maus-Kommentar:
// Diese Header-Datei deklariert alle öffentlichen Schnittstellen zum CUDA-Teil:
// - `renderCudaFrame(...)` rendert ein Fraktalbild und analysiert die Komplexität.
// - `keyCallback(...)` verarbeitet Tastendrücke.
// - `setPauseZoom(...)` / `getPauseZoom()` kontrollieren Auto-Zoom-Logik.
// - `registerPBO(...)` / `unregisterPBO()` kümmern sich um die Registrierung des PBO bei CUDA.

namespace CudaInterop {

/// 🖼️ Rendert ein Frame in ein OpenGL-PBO (optional mit Auto-Zoom auf interessante Bereiche)
void renderCudaFrame(
    uchar4* pbo,                        // 🧵 OpenGL-PBO (mapped CUDA-Pointer)
    int* d_iterations,                 // 🔁 Iterationen je Pixel (CUDA-Buffer)
    float* d_stddev,                   // σ Tile-Komplexität (Standardabweichung je Tile)
    float* d_mean,                     // μ Durchschnittliche Iterationen je Tile
    int width,                         // 📐 Bildbreite
    int height,                        // 📐 Bildhöhe
    float zoom,                        // 🔍 Aktueller Zoomfaktor
    float2 offset,                     // 🎯 Bildmittelpunkt im Fraktalraum
    int maxIterations,                 // ⏳ Max Iterationen pro Pixel
    std::vector<float>& h_complexity,  // 📊 Host-Puffer für Komplexitätsanalyse
    float2& outNewOffset,              // ⛳ Ziel-Koordinate für nächsten Zoom
    bool& shouldZoom,                  // 🚦 Zoom auslösen?
    int tileSize                       // 📦 Tile-Größe für dynamische Analyse
);

/// ⌨️ Tasteneingaben (Space: Auto-Zoom an/aus, P: Pause)
void keyCallback(
    GLFWwindow* window,
    int key,
    int scancode,
    int action,
    int mods
);

/// ⏸️ Aktiviert oder deaktiviert Auto-Zoom-Pause
void setPauseZoom(bool pause);

/// ⏯️ Prüft, ob Auto-Zoom pausiert ist
bool getPauseZoom();

/// 🔌 PBO bei CUDA registrieren
void registerPBO(GLuint pbo);

/// 🧹 PBO von CUDA deregistrieren
void unregisterPBO();

} // namespace CudaInterop
