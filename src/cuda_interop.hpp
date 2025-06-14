#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>  // 🐭 Für Tasteneingaben (Space: Auto-Zoom, P: Pause)

namespace CudaInterop {

/// 🖼️ Rendert ein Frame in ein OpenGL-PBO und analysiert die Komplexität für Auto-Zoom.
/// Der Host-Puffer `h_complexity` wird ggf. resized – daher NICHT const!
void renderCudaFrame(
    uchar4* pbo,                          // 🧠 CUDA-gemapptes OpenGL-PBO
    int* d_iterations,                   // 🔁 Iterationen je Pixel (CUDA-Buffer)
    float* d_complexity,                 // 📊 Komplexitätsdaten (pro Tile)
    float* d_stddev,                     // σ Tile-Komplexität (Standardabweichung je Tile)
    int width,                           // 📐 Bildbreite
    int height,                          // 📐 Bildhöhe
    float zoom,                          // 🔍 Aktueller Zoomfaktor
    float2 offset,                       // 🎯 Bildmittelpunkt im Fraktalraum
    int maxIterations,                   // ⏳ Max. Iterationen pro Pixel
    std::vector<float>& h_complexity,    // 📊 Host-Puffer für Komplexitätsanalyse (modifizierbar!)
    float2& outNewOffset,                // ⛳ Ziel-Koordinate für nächsten Zoom
    bool& shouldZoom,                    // 🚦 Auto-Zoom auslösen?
    int tileSize                         // 📦 Tile-Größe für Analyse
);

/// 🎹 Callback für Tastatureingaben (z. B. Space = Pause)
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

/// ⏸️ Setzt den Auto-Zoom pausiert/aktiv
void setPauseZoom(bool pause);

/// 🕹️ Fragt den aktuellen Zoom-Pausenstatus ab
bool getPauseZoom();

/// 🔗 Registriert ein OpenGL-PBO zur Verwendung mit CUDA
void registerPBO(GLuint pbo);

/// 🔌 Deregistriert das aktuell gebundene OpenGL-PBO aus CUDA
void unregisterPBO();

}  // namespace CudaInterop
