// Datei: src/cuda_interop.hpp
// Zeilen: 42
// 🐭 Maus-Kommentar: Schnittstelle zur CUDA/OpenGL Interop – PBO-Registrierung, Rendering-Bridge, Auto-Zoom-Steuerung mit Entropieanalyse. Der `keyCallback` steuert via SPACE/P das Pausieren des Zooms – Schneefuchs war Fan von Tastenkombis mit Logik dahinter.

#ifndef CUDA_INTEROP_HPP
#define CUDA_INTEROP_HPP

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <GLFW/glfw3.h>

namespace CudaInterop {

void registerPBO(unsigned int pbo);
void unregisterPBO();

void renderCudaFrame(    
    int* d_iterations,
    float* d_entropy,    
    int width,
    int height,
    float zoom,
    float2 offset,
    int maxIterations,
    std::vector<float>& h_entropy,
    float2& newOffset,
    bool& shouldZoom,
    int tileSize
);

void setPauseZoom(bool pause);
bool getPauseZoom();

// 🧠 Tastatureingabe-Handler für Auto-Zoom Pause (Taste P oder SPACE)
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

} // namespace CudaInterop

#endif // CUDA_INTEROP_HPP
