// Datei: src/main.cu
// Maus-Kommentar: Host-Code erstellt Bildpuffer, startet für jede Tile einen Block und rechnet
// das Fraktal hochparallel; anschließend wird kompaktes Complexity-Feedback gesammelt
// und auf den nächsten Zoom-Schritt angewendet.

#include "core_kernel.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <thrust/fill.h>
#include <thrust/execution_policy.h>  // für thrust::device

#include <iostream>
#include <vector>
#include <limits>

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void glCheck(const char* loc) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "GL error at " << loc << ": 0x"
                  << std::hex << err << std::dec << std::endl;
    }
}

// Komplexitäts-Kernel: zählt nicht-schwarze Pixel pro Tile
__global__ void computeComplexity(const uchar4* img,
                                  int width, int height,
                                  float* complexity);

int main() {
    const int width  = 1024;
    const int height = 768;
    size_t imgBytes  = width * height * sizeof(uchar4);

    float zoom    = 300.0f;
    float2 offset = make_float2(0.0f, 0.0f);
    int   maxIter = 500;

    std::cout << "=== Programm gestartet ===" << std::endl;

    // GLFW + OpenGL init
    if (!glfwInit()) exit(EXIT_FAILURE);
    std::cout << "GLFW initialisiert" << std::endl;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow(width, height, "Auto-Zoom Mandelbrot", nullptr, nullptr);
    if (!window) exit(EXIT_FAILURE);
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW init failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "GLEW initialisiert, OpenGL-Kontext ready" << std::endl;

    // PBO + CUDA-GL Interop
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, imgBytes, nullptr, GL_DYNAMIC_DRAW);
    glCheck("glBufferData");

    cudaGraphicsResource* cudaPbo;
    checkCuda(cudaGraphicsGLRegisterBuffer(&cudaPbo, pbo, cudaGraphicsMapFlagsWriteDiscard),
              "cudaGraphicsGLRegisterBuffer");

    // Texture
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glCheck("Texture Setup");

    // Complexity-Buffer
    int tilesX     = (width  + TILE_W - 1) / TILE_W;
    int tilesY     = (height + TILE_H - 1) / TILE_H;
    int totalTiles = tilesX * tilesY;

    float* d_complexity = nullptr;
    checkCuda(cudaMalloc(&d_complexity, totalTiles * sizeof(float)), "cudaMalloc complexity");
    std::vector<float> h_complexity(totalTiles);

    std::cout << "Setup abgeschlossen, betrete Haupt-Loop" << std::endl;

    int frame = 0;
    while (!glfwWindowShouldClose(window)) {
        std::cout << "Frame " << frame++
                  << ": Zoom=" << zoom
                  << " Offset=(" << offset.x << "," << offset.y << ")"
                  << std::endl;

        // Debug: fülle PBO erst mal rot, um Pipeline zu testen
        {
            uchar4* d_img = nullptr;
            size_t sz = 0;
            checkCuda(cudaGraphicsMapResources(1, &cudaPbo, 0), "MapResources (debug fill)");
            checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_img, &sz, cudaPbo),
                      "GetMappedPointer (debug fill)");
            thrust::fill_n(thrust::device, d_img, width*height, make_uchar4(255,0,0,255));
            checkCuda(cudaGraphicsUnmapResources(1, &cudaPbo, 0), "UnmapResources (debug fill)");
        }

        // 1) Reset Complexity
        checkCuda(cudaMemset(d_complexity, 0, totalTiles * sizeof(float)), "memset complexity");

        // 2) PBO → CUDA
        uchar4* d_img = nullptr;
        size_t   sz   = 0;
        checkCuda(cudaGraphicsMapResources(1, &cudaPbo, 0), "MapResources");
        checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_img, &sz, cudaPbo),
                  "GetMappedPointer");

        // 3) Fraktal-Kernel tile-parallel: ein Block pro Tile
        dim3 blockDim(TILE_W, TILE_H);
        dim3 gridDim (tilesX,   tilesY);
        launch_mandelbrotHybrid(d_img,
                                width, height,
                                zoom, offset,
                                maxIter);
        checkCuda(cudaDeviceSynchronize(), "mandelbrotHybrid");

        // 4) Complexity-Kernel
        computeComplexity<<<gridDim, blockDim>>>(d_img, width, height, d_complexity);
        checkCuda(cudaDeviceSynchronize(), "computeComplexity");

        // 5) Unmap PBO
        checkCuda(cudaGraphicsUnmapResources(1, &cudaPbo, 0), "UnmapResources");

        // 6) Best Tile finden
        checkCuda(cudaMemcpy(h_complexity.data(), d_complexity,
                             totalTiles * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "Memcpy complexity");

        int   bestIdx   = 0;
        float bestScore = -1.0f;
        for (int i = 0; i < totalTiles; ++i) {
            if (h_complexity[i] > bestScore) {
                bestScore = h_complexity[i];
                bestIdx   = i;
            }
        }
        std::cout << "  bestScore=" << bestScore << std::endl;

        int bestX = bestIdx % tilesX;
        int bestY = bestIdx / tilesX;
        offset.x += ((bestX + 0.5f) * TILE_W - width  * 0.5f) / zoom;
        offset.y += ((bestY + 0.5f) * TILE_H - height * 0.5f) / zoom;
        zoom *= 1.2f;

        // 7) Rendern aus PBO
        glClear(GL_COLOR_BUFFER_BIT);
        glCheck("glClear");

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glCheck("glTexSubImage2D");

        glBegin(GL_QUADS);
          glTexCoord2f(0,0); glVertex2f(-1,-1);
          glTexCoord2f(1,0); glVertex2f( 1,-1);
          glTexCoord2f(1,1); glVertex2f( 1, 1);
          glTexCoord2f(0,1); glVertex2f(-1, 1);
        glEnd();
        glCheck("glEnd");

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    cudaFree(d_complexity);
    cudaGraphicsUnregisterResource(cudaPbo);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
