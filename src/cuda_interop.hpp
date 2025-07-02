// Datei: src/cuda_interop.hpp
// Zeilen: 53
// 🐭 Maus-Kommentar: Schnittstelle zur CUDA/OpenGL Interop – Signatur für `renderCudaFrame` auf 13 Argumente aktualisiert (inkl. `supersampling`, `RendererState&`) – Schneefuchs: „Ein Zustand, der wandert, ist keiner, der lauert.“

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
    double2& newOffset,     // ✅ FIXED: war fälschlich float2 – jetzt korrekt
    bool& shouldZoom,
    int tileSize,
    int supersampling,      // ✅ NEU: Supersampling-Faktor
    RendererState& state    // ✅ explizit übergeben
);

void setPauseZoom(bool pause);
bool getPauseZoom();

// 🧪 Evaluation nach Zielwechsel – Frame-Analyse direkt aus GPU-Buffer
void logZoomEvaluation(const int* d_iterations, int width, int height, int maxIterations, double zoom);

} // namespace CudaInterop

#endif // CUDA_INTEROP_HPP
