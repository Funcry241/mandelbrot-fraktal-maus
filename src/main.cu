// Datei: src/main.cu
// Maus-Kommentar: Host-Code erstellt Bildpuffer, startet für jede Tile einen Block und rechnet
// das Fraktal hochparallel; anschließend wird kompaktes Complexity-Feedback gesammelt
// und auf den nächsten Zoom-Schritt angewendet.

#include "core_kernel.h"

#ifdef _WIN32
#  include <windows.h>
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
        std::cerr << "[CUDA ERROR] " << msg << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Komplexitäts-Kernel: zählt die Gesamt-Iteration pro Tile
__global__ void computeComplexity(const uchar4* img,
                                  int width, int height,
                                  float* complexity)
{
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int startX = tileX * TILE_W;
    int startY = tileY * TILE_H;
    int endX   = min(startX + TILE_W, width);
    int endY   = min(startY + TILE_H, height);

    // Thread‐strided Loop innerhalb der Kachel
    float sum = 0.0f;
    for (int y = startY + threadIdx.y; y < endY; y += blockDim.y) {
        for (int x = startX + threadIdx.x; x < endX; x += blockDim.x) {
            uchar4 c = img[y * width + x];
            // Nicht‐schwarz addieren
            if (c.x || c.y || c.z) sum += 1.0f;
        }
    }
    // Reduktion im Block
    __shared__ float buf[TILE_W * TILE_H / 32];  // genug für 32 Threads
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    buf[tid] = sum;
    __syncthreads();

    // einfache Warp‐Reduktion
    if (tid < 32) {
        for (int offset = 32; offset < blockDim.x * blockDim.y; offset += 32)
            buf[tid] += buf[tid + offset];
        if (tid == 0)
            complexity[blockIdx.y * gridDim.x + blockIdx.x] = buf[0];
    }
}

int main() {
    const int width  = 1024;
    const int height = 768;
    const size_t imgBytes = size_t(width) * height * sizeof(uchar4);

    float zoom    = 300.0f;
    float2 offset = make_float2(0.0f, 0.0f);
    int   maxIter = 500;

    // Sanity‐Check maxIter
    if (maxIter < 1) {
        std::cerr << "[WARN] maxIter<1, setze auf 1\n";
        maxIter = 1;
    }

    // GLFW + OpenGL init
    if (!glfwInit()) {
        std::cerr << "[ERROR] GLFW init failed\n";
        return EXIT_FAILURE;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow(width, height, "Auto-Zoom Mandelbrot", nullptr, nullptr);
    if (!window) return EXIT_FAILURE;
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "[ERROR] GLEW init failed\n";
        return EXIT_FAILURE;
    }

    // PBO + CUDA-GL Interop
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, imgBytes, nullptr, GL_DYNAMIC_DRAW);

    cudaGraphicsResource* cudaPbo = nullptr;
    checkCuda(cudaGraphicsGLRegisterBuffer(&cudaPbo, pbo, cudaGraphicsMapFlagsWriteDiscard),
              "cudaGraphicsGLRegisterBuffer");

    // Textur anlegen
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Complexity‐Buffer
    int tilesX = (width  + TILE_W - 1) / TILE_W;
    int tilesY = (height + TILE_H - 1) / TILE_H;
    int totalTiles = tilesX * tilesY;

    float* d_complexity = nullptr;
    checkCuda(cudaMalloc(&d_complexity, totalTiles * sizeof(float)),
              "cudaMalloc complexity");
    std::vector<float> h_complexity(totalTiles);

    // Haupt-Loop
    while (!glfwWindowShouldClose(window)) {
        // 1) Complexity zurücksetzen
        checkCuda(cudaMemset(d_complexity, 0, totalTiles * sizeof(float)),
                  "cudaMemset complexity");

        // 2) Map PBO
        uchar4* d_img = nullptr;
        size_t sz = 0;
        checkCuda(cudaGraphicsMapResources(1, &cudaPbo, 0), "cudaGraphicsMapResources");
        checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_img, &sz, cudaPbo),
                  "cudaGraphicsResourceGetMappedPointer");

        // 3) Mandelbrot-Kernel
        dim3 blockDim(TILE_W, TILE_H);
        dim3 gridDim (tilesX,   tilesY);
        mandelbrotHybrid<<<gridDim, blockDim>>>(d_img,
                                                width, height,
                                                zoom, offset,
                                                maxIter);
        checkCuda(cudaGetLastError(), "mandelbrotHybrid launch");
        checkCuda(cudaDeviceSynchronize(), "mandelbrotHybrid sync");

        // 4) Complexity-Kernel
        computeComplexity<<<gridDim, blockDim>>>(d_img, width, height, d_complexity);
        checkCuda(cudaGetLastError(), "computeComplexity launch");
        checkCuda(cudaDeviceSynchronize(), "computeComplexity sync");

        // 5) Unmap PBO
        checkCuda(cudaGraphicsUnmapResources(1, &cudaPbo, 0), "cudaGraphicsUnmapResources");

        // 6) Best Tile ermitteln
        checkCuda(cudaMemcpy(h_complexity.data(), d_complexity,
                             totalTiles * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy complexity");

        int   bestIdx   = 0;
        float bestScore = -1.0f;
        for (int i = 0; i < totalTiles; ++i) {
            if (h_complexity[i] > bestScore) {
                bestScore = h_complexity[i];
                bestIdx   = i;
            }
        }

        int bestX = bestIdx % tilesX;
        int bestY = bestIdx / tilesX;
        offset.x += ((bestX + 0.5f) * TILE_W - width  * 0.5f) / zoom;
        offset.y += ((bestY + 0.5f) * TILE_H - height * 0.5f) / zoom;
        zoom *= 1.2f;

        // 7) Render aus PBO
        glClear(GL_COLOR_BUFFER_BIT);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        width, height,
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
