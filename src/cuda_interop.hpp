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
    cudaGraphicsResource_t cudaPboRes, // 🐭 OpenGL PBO Resource
    int width,
    int height,
    float& zoom,                       // 🔍 Aktueller Zoomfaktor (wird ggf. modifiziert)
    float2& offset,                    // 🎯 Aktueller Offset (Mitte des Bildes im Fraktalraum)
    int maxIter,                       // ⏳ Max Iterationen pro Pixel
    float* d_complexity,               // 🐭 CUDA-Buffer für Tile-Komplexitäten (Device)
    std::vector<float>& h_complexity,  // 🐭 Host-Speicher für Komplexitätsanalyse
    int* d_iterations,                 // 🐭 CUDA-Buffer für Iterationstiefe je Pixel
    bool autoZoomEnabled               // 🐭 Steuerung: Auto-Zoom aktivieren/deaktivieren
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
