// Datei: src/cuda_interop.hpp
// Zeilen: 54
// 🐭 Maus-Kommentar: Schnittstelle zur CUDA/OpenGL Interop – Signatur für `renderCudaFrame` auf 14 Argumente aktualisiert (inkl. Kontrastpuffer und `RendererState&`) – Schneefuchs: „Ein Zustand, der wandert, ist keiner, der lauert.“

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

// 🔁 Haupt-Renderfunktion – mit double-Precision, Supersampling, Kontrastanalyse und Zustand
void renderCudaFrame(    
    int* d_iterations,
    float* d_entropy,
    float* d_contrast,                   // ✅ NEU: GPU-Puffer für Kontrast
    int width,
    int height,
    double zoom,
    double2 offset,
    int maxIterations,
    std::vector<float>& h_entropy,
    std::vector<float>& h_contrast,     // ✅ NEU: Host-Kontrastdaten
    double2& newOffset,
    bool& shouldZoom,
    int tileSize,
    int supersampling,
    RendererState& state
);

void setPauseZoom(bool pause);
bool getPauseZoom();

// 🧪 Evaluation nach Zielwechsel – Frame-Analyse direkt aus GPU-Buffer
void logZoomEvaluation(const int* d_iterations, int width, int height, int tileSize, double zoom);

} // namespace CudaInterop

#endif // CUDA_INTEROP_HPP
