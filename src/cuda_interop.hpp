// Datei: src/cuda_interop.hpp
// 🐭 Maus-Kommentar: Schnittstelle zur CUDA/OpenGL Interop – inkl. PBO-Handling und Auto-Zoom-Logik

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
    uchar4* output,
    int* d_iterations,
    float* d_complexity,
    float* d_stddev,
    int width,
    int height,
    float zoom,
    float2 offset,
    int maxIterations,
    std::vector<float>& h_complexity,
    float2& newOffset,
    bool& shouldZoom,
    int tileSize
);

void setPauseZoom(bool pause);
bool getPauseZoom();

// 🧠 Neu: Tastatureingabe-Handler für Auto-Zoom Pause (Taste P)
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

} // namespace CudaInterop

#endif // CUDA_INTEROP_HPP
