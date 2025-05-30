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
    std::cout << "=== Programm gestartet ===" << std::endl;

    const int width  = 1024;
    const int height = 768;
    size_t imgBytes  = width * height * sizeof(uchar4);

    float zoom    = 300.0f;
    float2 offset = make_float2(0.0f, 0.0f);
    int   maxIter = 500;

    if (!glfwInit()) {
        std::cerr << "glfwInit() failed" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "GLFW initialisiert" << std::endl;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow(width, height, "Auto-Zoom Mandelbrot", nullptr, nullptr);
    if (!window) {
        std::cerr << "glfwCreateWindow() failed" << std::endl;
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW init failed" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "GLEW initialisiert, OpenGL-Kontext ready" << std::endl;

    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, imgBytes, nullptr, GL_DYNAMIC_DRAW);

    cudaGraphicsResource* cudaPbo;
    checkCuda(cudaGraphicsGLRegisterBuffer(&cudaPbo, pbo, cudaGraphicsMapFlagsWriteDiscard),
              "cudaGraphicsGLRegisterBuffer");

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    int tilesX     = (width  + TILE_W - 1) / TILE_W;
    int tilesY     = (height + TILE_H - 1) / TILE_H;
    int totalTiles = tilesX * tilesY;

    float* d_complexity = nullptr;
    checkCuda(cudaMalloc(&d_complexity, totalTiles * sizeof(float)), "cudaMalloc complexity");
    std::vector<float> h_complexity(totalTiles);

    std::cout << "Setup abgeschlossen, betrete Haupt-Loop" << std::endl;

    int frameCount = 0;
    while (!glfwWindowShouldClose(window)) {
        // Debug-Ausgabe pro Frame
        std::cout << "Frame " << frameCount++
                  << ": Zoom=" << zoom
                  << " Offset=(" << offset.x << "," << offset.y << ")" << std::endl;
        std::cout.flush();

        checkCuda(cudaMemset(d_complexity, 0, totalTiles * sizeof(float)), "memset complexity");

        uchar4* d_img = nullptr;
        size_t sz = 0;
        checkCuda(cudaGraphicsMapResources(1, &cudaPbo, 0), "MapResources");
        checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_img, &sz, cudaPbo), "GetMappedPointer");

        dim3 blockDim(TILE_W, TILE_H);
        dim3 gridDim (tilesX,   tilesY);
        launch_mandelbrotHybrid(d_img, width, height, zoom, offset, maxIter);
        checkCuda(cudaDeviceSynchronize(), "mandelbrotHybrid");

        computeComplexity<<<gridDim, blockDim>>>(d_img, width, height, d_complexity);
        checkCuda(cudaDeviceSynchronize(), "computeComplexity");

        checkCuda(cudaGraphicsUnmapResources(1, &cudaPbo, 0), "UnmapResources");

        checkCuda(cudaMemcpy(h_complexity.data(), d_complexity, totalTiles * sizeof(float), cudaMemcpyDeviceToHost),
                  "Memcpy complexity");

        int bestIdx = 0; float bestScore = -1.0f;
        for (int i = 0; i < totalTiles; ++i) {
            if (h_complexity[i] > bestScore) {
                bestScore = h_complexity[i];
                bestIdx = i;
            }
        }
        std::cout << "  bestScore=" << bestScore << std::endl;

        int bestX = bestIdx % tilesX;
        int bestY = bestIdx / tilesX;
        offset.x += ((bestX + 0.5f) * TILE_W - width  * 0.5f) / zoom;
        offset.y += ((bestY + 0.5f) * TILE_H - height * 0.5f) / zoom;
        zoom *= 1.2f;

        glClear(GL_COLOR_BUFFER_BIT);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        glBegin(GL_QUADS);
          glTexCoord2f(0,0); glVertex2f(-1,-1);
          glTexCoord2f(1,0); glVertex2f( 1,-1);
          glTexCoord2f(1,1); glVertex2f( 1, 1);
          glTexCoord2f(0,1); glVertex2f(-1, 1);
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaFree(d_complexity);
    cudaGraphicsUnregisterResource(cudaPbo);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
