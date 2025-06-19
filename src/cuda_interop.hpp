// Datei: src/cuda_interop.hpp
// Zeilen: 45
// 🐭 Maus-Kommentar: Schnittstelle zur CUDA/OpenGL Interop – PBO-Registrierung, CUDA-Render-Bridge, Auto-Zoom mit Entropieanalyse. Entfernt direkte CUDA-Includes zur Vermeidung von PCH-Fehlern. `keyCallback` erlaubt Zoom-Pause per Tastatur. Schneefuchs sagte: „Ein Interface soll nie stolpern.“

#ifndef CUDA_INTEROP_HPP
#define CUDA_INTEROP_HPP

#include <vector>
#include <GLFW/glfw3.h>
#include <vector_types.h>  // float2

// 🧠 Vorwärtsdeklaration – CUDA-Typen nicht direkt inkludieren
struct cudaGraphicsResource;

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
