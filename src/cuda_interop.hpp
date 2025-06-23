// Datei: src/cuda_interop.hpp
// Zeilen: 49
// 🐭 Maus-Kommentar: Schnittstelle zur CUDA/OpenGL Interop – jetzt mit double-präzisem Zoom & Offset für hohe Vergrößerungstiefe. Keine CUDA-Header in der PCH – sauber gekapselt. Schneefuchs sagte: „Wer weit sehen will, braucht scharfe Koordinaten.“

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

// 🔁 Haupt-Renderfunktion – jetzt mit double-Parameter für Präzision
void renderCudaFrame(    
    int* d_iterations,
    float* d_entropy,
    int width,
    int height,
    double zoom,           // ✅ jetzt double
    double2 offset,        // ✅ jetzt double2
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
