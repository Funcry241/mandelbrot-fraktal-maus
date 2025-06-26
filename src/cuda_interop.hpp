// Datei: src/cuda_interop.hpp
// Zeilen: 48
// 🐭 Maus-Kommentar: Schnittstelle zur CUDA/OpenGL Interop – `globalRendererState` entfernt, `RendererState&` wird direkt übergeben. Schneefuchs: „Ein Zustand, der wandert, ist keiner, der lauert.“

#ifndef CUDA_INTEROP_HPP
#define CUDA_INTEROP_HPP

#include <vector>
#include <GLFW/glfw3.h>
#include <vector_types.h>  // float2, double2

// 🧠 Vorwärtsdeklaration – CUDA-Typen nicht direkt inkludieren
struct cudaGraphicsResource;

class RendererState;  // 🧠 Nur Vorwärtsdeklaration nötig (kein Include von renderer_state.hpp)

namespace CudaInterop {

void registerPBO(unsigned int pbo);
void unregisterPBO();

// 🔁 Haupt-Renderfunktion – mit double-Precision + direkter Übergabe des Zustands
void renderCudaFrame(    
    int* d_iterations,
    float* d_entropy,
    int width,
    int height,
    double zoom,
    double2 offset,
    int maxIterations,
    std::vector<float>& h_entropy,
    float2& newOffset,
    bool& shouldZoom,
    int tileSize,
    RendererState& state // ✅ neu: Zustand explizit übergeben
);

void setPauseZoom(bool pause);
bool getPauseZoom();

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

} // namespace CudaInterop

#endif // CUDA_INTEROP_HPP
