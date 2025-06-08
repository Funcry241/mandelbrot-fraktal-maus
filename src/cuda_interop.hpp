#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h> // 🐭 Für Tasteneingaben

// ----------------------------------------------------------------------
// 🐭 Kernel-Wrappers

extern "C" void launch_debugGradient(uchar4* img, int width, int height);

// 🐭 Mandelbrot-Hybrid-Renderer (Iteration Buffer wird mitgeführt)
// Kein extern "C", da C++-Signatur!
void launch_mandelbrotHybrid(
    uchar4* img,
    int* iterations,   // 🐭 Iteration Buffer
    int width,
    int height,
    float zoom,
    float2 offset,
    int maxIter
);

// ----------------------------------------------------------------------
// 🐭 Gesamte CUDA-Rendering-Pipeline (Namespace CudaInterop)
namespace CudaInterop {

/// 🐭 Rendert einen CUDA-Frame in ein OpenGL-PBO mit optionalem Auto-Zoom.
void renderCudaFrame(
    cudaGraphicsResource_t cudaPboRes,
    int width,
    int height,
    float& zoom,
    float2& offset,
    int maxIter,
    float* d_complexity,
    std::vector<float>& h_complexity,
    int* d_iterations,
    bool autoZoomEnabled    // 🐭 Auto-Zoom jetzt gesteuert über Parameter
);

/// 🐭 Key-Callback zur Laufzeit-Steuerung (z.B. Leertaste für Pause/Resume)
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

/// 🐭 Setzt den Pause-Zustand für Zoom (true = pausiert)
void setPauseZoom(bool pause);

/// 🐭 Holt den aktuellen Pause-Zustand für Zoom
bool getPauseZoom();

} // namespace CudaInterop
