// Datei: src/cuda_interop.hpp
// Zeilen: 48
// 🐭 Maus-Kommentar: Schnittstelle zur CUDA/OpenGL Interop – PBO-Registrierung, CUDA-Render-Bridge, Auto-Zoom mit Entropieanalyse. Entfernt direkte CUDA-Includes zur Vermeidung von PCH-Fehlern. `keyCallback` erlaubt Zoom-Pause per Tastatur. Schneefuchs sagte: „Ein Interface soll nie stolpern.“

#ifndef CUDA_INTEROP_HPP
#define CUDA_INTEROP_HPP

#include <vector>
#include <GLFW/glfw3.h>
#include <vector_types.h>  // float2

// 🧠 Vorwärtsdeklaration – CUDA-Typen nicht direkt inkludieren
struct cudaGraphicsResource;

class RendererState;  // 🧠 Nur Vorwärtsdeklaration nötig (kein Include von renderer_state.hpp)

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

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

// 🧭 Globale Referenz auf den aktuellen Renderer-State für CUDA ↔ Zoom-Steuerung
extern RendererState* globalRendererState;

} // namespace CudaInterop

#endif // CUDA_INTEROP_HPP
