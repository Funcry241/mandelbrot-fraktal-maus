// Datei: src/cuda_interop.hpp
// Zeilen: 54
// 🐭 Maus-Kommentar: Schnittstelle zur CUDA/OpenGL Interop – Flugente-konform auf float2 reduziert, um FPS-Verlust zu vermeiden. Schneefuchs: „Präzision darf rasten, wenn Performance hastet.“

#ifndef CUDA_INTEROP_HPP
#define CUDA_INTEROP_HPP

#include <vector>
#include <GLFW/glfw3.h>
#include <vector_types.h> // float2

// 🧠 Vorwärtsdeklaration – CUDA-Typen nicht direkt inkludieren
struct cudaGraphicsResource;

class RendererState; // 🧠 Nur Vorwärtsdeklaration nötig (kein Include von renderer_state.hpp)

namespace CudaInterop {

void registerPBO(unsigned int pbo);
void unregisterPBO();

// 🔁 Haupt-Renderfunktion – Flugente-konform mit float2, Supersampling, Kontrastanalyse und Zustand
void renderCudaFrame(
int* d_iterations,
float* d_entropy,
float* d_contrast,
int width,
int height,
float zoom,
float2 offset,
int maxIterations,
std::vector<float>& h_entropy,
std::vector<float>& h_contrast,
float2& newOffset,
bool& shouldZoom,
int tileSize,
int supersampling,
RendererState& state
);

void setPauseZoom(bool pause);
bool getPauseZoom();

// 🧪 Evaluation nach Zielwechsel – Frame-Analyse direkt aus GPU-Buffer
void logZoomEvaluation(const int* d_iterations, int width, int height, int tileSize, float zoom);

} // namespace CudaInterop

#endif // CUDA_INTEROP_HPP
