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

#include <iostream>
#include <vector>
#include <limits>

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Komplexitäts-Kernel: zählt nicht-schwarze Pixel pro Tile
__global__ void computeComplexity(const uchar4* img,
                                  int width, int height,
                                  float* complexity);

int main() {
    const int W = 1024;
    const int H = 768;
    size_t imgBytes = W * H * sizeof(uchar4);

    float zoom    = 300.0f;
    float2 offset = make_float2(0.0f, 0.0f);
    int   maxIter = 500;

    // GLFW + OpenGL init
    if (!glfwInit()) exit(EXIT_FAILURE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow(W, H, "Auto-Zoom Mandelbrot", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    // Viewport und Pixel-Alignment
    int fbW, fbH;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    glViewport(0, 0, fbW, fbH);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW init failed" << std::endl;
        return EXIT_FAILURE;
    }

    // PBO + CUDA-GL Interop
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, imgBytes, nullptr, GL_DYNAMIC_DRAW);

    cudaGraphicsResource* cudaPbo;
    checkCuda(cudaGraphicsGLRegisterBuffer(&cudaPbo, pbo, cudaGraphicsMapFlagsWriteDiscard),
              "cudaGraphicsGLRegisterBuffer");

    // Texture
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Complexity-Buffer
    int tilesX     = (W + TILE_W - 1) / TILE_W;
    int tilesY     = (H + TILE_H - 1) / TILE_H;
    int totalTiles = tilesX * tilesY;

    float* d_complexity = nullptr;
    checkCuda(cudaMalloc(&d_complexity, totalTiles * sizeof(float)), "cudaMalloc complexity");
    std::vector<float> h_complexity(totalTiles);

    // Haupt-Loop
    while (!glfwWindowShouldClose(window)) {
        // 1) Reset Complexity
        checkCuda(cudaMemset(d_complexity, 0, totalTiles * sizeof(float)), "memset complexity");

        // 2) PBO → CUDA
        uchar4* d_img = nullptr;
        size_t   sz   = 0;
        checkCuda(cudaGraphicsMapResources(1, &cudaPbo, 0), "MapResources");
        checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_img, &sz, cudaPbo),
                  "GetMappedPointer");

        // --- Debug: Test-Fill (mittleres Grau) ---
        // cudaMemset(d_img, 128, imgBytes);

        // 3) Fraktal-Kernel: ein Block pro Tile
        dim3 bd(TILE_W, TILE_H);
        dim3 gd(tilesX,   tilesY);
        mandelbrotHybrid<<<gd, bd>>>(d_img, W, H, zoom, offset, maxIter);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel mandelbrotHybrid launch error: "
                      << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        checkCuda(cudaDeviceSynchronize(), "mandelbrotHybrid");

        // 4) Complexity-Kernel
        computeComplexity<<<gd, bd>>>(d_img, W, H, d_complexity);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel computeComplexity launch error: "
                      << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        checkCuda(cudaDeviceSynchronize(), "computeComplexity");

        // --- Debug: Sample Pixel ---
        /*
        uchar4 sample;
        cudaMemcpy(&sample, d_img, sizeof(uchar4), cudaMemcpyDeviceToHost);
        std::cout << "Sample RGBA: (" << int(sample.x) << ","
                  << int(sample.y) << "," << int(sample.z) << ")\n";
        */

        // 5) Unmap PBO
        checkCuda(cudaGraphicsUnmapResources(1, &cudaPbo, 0), "UnmapResources");

        // 6) Best Tile finden und Zoom anpassen
        checkCuda(cudaMemcpy(h_complexity.data(), d_complexity,
                             totalTiles * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "Memcpy complexity");

        int bestIdx = 0;
        float bestScore = -1.0f;
        for (int i = 0; i < totalTiles; ++i) {
            if (h_complexity[i] > bestScore) {
                bestScore = h_complexity[i];
                bestIdx   = i;
            }
        }
        int bestX = bestIdx % tilesX;
        int bestY = bestIdx / tilesX;
        offset.x += ((bestX + 0.5f) * TILE_W -  W * 0.5f) / zoom;
        offset.y += ((bestY + 0.5f) * TILE_H -  H * 0.5f) / zoom;
        zoom *= 1.2f;

        // 7) Rendern aus PBO
        glClear(GL_COLOR_BUFFER_BIT);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H,
                        GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        glBegin(GL_QUADS);
          glTexCoord2f(0,0); glVertex2f(-1,-1);
          glTexCoord2f(1,0); glVertex2f( 1,-1);
          glTexCoord2f(1,1); glVertex2f( 1, 1);
          glTexCoord2f(0,1); glVertex2f(-1, 1);
        glEnd();

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
