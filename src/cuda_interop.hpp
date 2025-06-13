#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h> // 🐭 Für Tasteneingaben (Space: Auto-Zoom, P: Pause)

// ----------------------------------------------------------------------
// 🎯 CUDA-Rendering- und Auto-Zoom-Controller (Namespace CudaInterop)
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
    int tileSize  // ⬅️ ❗️Dieser Parameter fehlte!
);

/// ⌨️ Key-Callback für Laufzeitsteuerung (Space: Auto-Zoom an/aus, P: Pause/Resume)
void keyCallback(
    GLFWwindow* window,
    int key,
    int scancode,
    int action,
    int mods
);

/// ⏸️ Setzt den Pause-Modus für den Auto-Zoom (true = Pause)
void setPauseZoom(bool pause);

/// ⏯️ Fragt ab, ob der Auto-Zoom aktuell pausiert ist
bool getPauseZoom();

} // namespace CudaInterop
