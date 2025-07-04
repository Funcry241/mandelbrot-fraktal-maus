// Datei: src/cuda_interop.hpp
// Zeilen: 65
// 🐭 Maus-Kommentar: Schnittstelle zur CUDA/OpenGL-Interop – Kolibri+Panda integriert, Flugente-konform mit float2.
// Schneefuchs: „Präzision darf rasten, wenn Performance eilt.“

#ifndef CUDA_INTEROP_HPP
#define CUDA_INTEROP_HPP

#include <vector>
#include <GLFW/glfw3.h>
#include <vector_types.h> // float2

// 🧠 Vorwärtsdeklaration für RendererState
class RendererState;

namespace CudaInterop {

void registerPBO(unsigned int pbo);
void unregisterPBO();

// 🔁 Haupt-Renderfunktion:
// - d_iterations: Device-Puffer Iterationen
// - d_entropy, d_contrast: Device-Puffer Entropie/Kontrast
// - width, height: Bildgröße
// - zoom, offset: Fraktal-Transformation
// - maxIterations: Iterationslimit
// - h_entropy, h_contrast: Host-Puffer zum Empfangen der Analysedaten
// - newOffset, shouldZoom: Ausgabewerte für Auto-Zoom
// - tileSize: aktuelle Kachelgröße
// - supersampling: globaler Fallback-Wert (nicht mehr genutzt intern)
// - state: RendererState für Zoom-Ergebnisse und Einstellungen
// - d_tileSupersampling, h_tileSupersampling: Puffer für adaptives Supersampling pro Tile
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
    RendererState& state,
    int* d_tileSupersampling,
    std::vector<int>& h_tileSupersampling
);

void setPauseZoom(bool pause);
bool getPauseZoom();

// 🧪 CSV-Ausgabe für Zielanalyse
void logZoomEvaluation(const int* d_iterations, int width, int height, int tileSize, float zoom);

} // namespace CudaInterop

#endif // CUDA_INTEROP_HPP
