#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h> // 🐭 Für Tasteneingaben (Space: Auto-Zoom, P: Pause)

namespace CudaInterop {

/// 🖼️ Rendert ein Frame in ein OpenGL-PBO (optional mit Auto-Zoom auf interessante Bereiche)
void renderCudaFrame(
    uchar4* pbo,                        // 🧠 CUDA-gemapptes OpenGL-PBO
    int* d_iterations,                 // 🔁 Iterationen je Pixel (CUDA-Buffer)
    float* d_complexity,               // 📊 Komplexitätsdaten (pro Tile)
    float* d_stddev,                   // σ Tile-Komplexität (Standardabweichung je Tile)
    int width,                         // 📐 Bildbreite
    int height,                        // 📐 Bildhöhe
    float zoom,                        // 🔍 Aktueller Zoomfaktor
    float2 offset,                     // 🎯 Bildmittelpunkt im Fraktalraum
    int maxIterations,                 // ⏳ Max Iterationen pro Pixel
    const std::vector<float>& h_complexity,  // 📊 Host-Puffer für Komplexitätsanalyse
    float2& outNewOffset,              // ⛳ Ziel-Koordinate für nächsten Zoom
    bool& shouldZoom,                  // 🚦 Zoom auslösen?
    int tileSize                       // 📦 Tile-Größe für dynamische Analyse
);

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void setPauseZoom(bool pause);
bool getPauseZoom();
void registerPBO(GLuint pbo);
void unregisterPBO();

} // namespace CudaInterop
