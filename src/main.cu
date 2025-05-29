// Datei: src/main.cu
// Maus-Kommentar: Host-Code initialisiert Bildpuffer, setzt den globalen Tile-Zähler zurück,
// startet den mandelbrotPersistent-Kernel und rendert via CUDA-OpenGL-Interop das automatische Fly-Through.

#include "core_kernel.h"

#ifdef _WIN32
#include <windows.h>           // für OpenGL unter Windows
#endif

// Wir verlinken gegen glew32.dll, daher kein GLEW_STATIC!
// GLEW muss vor GLFW und anderen GL-Headern inkludiert werden:
#include <GL/glew.h>            
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>    // für CUDA-OpenGL Interop
#include <vector_types.h>       // für uchar4, float2
#include <vector_functions.h>   // für make_uchar4, make_float2

#include <iostream>
#include <vector>
#include <limits>

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Kernel: Zähle in jeder Kachel, wie viele Pixel nicht schwarz sind → Komplexität
__global__ void computeComplexity(const uchar4* img,
                                  int width, int height,
                                  float* complexity);

int main() {
    // Bildgrößen
    const int width  = 1024;
    const int height = 768;
    size_t imgBytes = width * height * sizeof(uchar4);

    // Fraktal-Parameter
    float zoom    = 300.0f;
    float2 offset = make_float2(0.0f, 0.0f);
    int maxIter   = 500;

    // GLFW + OpenGL init
    if (!glfwInit()) exit(EXIT_FAILURE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
    GLFWwindow* window = glfwCreateWindow(width, height, "Auto-Zoom Mandelbrot", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        std::cerr << "GLEW init failed: " << glewGetErrorString(glewErr) << std::endl;
        return EXIT_FAILURE;
    }

    // PBO erstellen und mit CUDA registrieren
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, imgBytes, nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsResource* cudaPbo;
    checkCuda(cudaGraphicsGLRegisterBuffer(&cudaPbo, pbo, cudaGraphicsMapFlagsWriteDiscard),
              "cudaGraphicsGLRegisterBuffer");

    // Texture für Anzeige
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Komplexitätsbuffer
    int tilesX = (width + TILE_W -1)/TILE_W;
    int tilesY = (height + TILE_H -1)/TILE_H;
    int totalTiles = tilesX * tilesY;
    float* d_complexity = nullptr;
    checkCuda(cudaMalloc(&d_complexity, totalTiles * sizeof(float)), "cudaMalloc complexity");
    std::vector<float> h_complexity(totalTiles);

    // Haupt-Loop
    while (!glfwWindowShouldClose(window)) {
        // Reset
        checkCuda(cudaMemset(d_complexity, 0, totalTiles*sizeof(float)), "memset complexity");
        checkCuda(cudaMemcpyToSymbol(tileIdxGlobal, 0, sizeof(int)),       "reset tileIdxGlobal");

        // PBO zu CUDA mapen
        uchar4* d_img = nullptr;
        size_t   size = 0;
        checkCuda(cudaGraphicsMapResources(1, &cudaPbo, 0), "MapResources");
        checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_img, &size, cudaPbo),
                  "GetMappedPointer");

        // Mandelbrot persistent Kernel
        dim3 blockDim(TILE_W, TILE_H);
        dim3 gridDim(1, 1);
        mandelbrotPersistent<<<gridDim, blockDim>>>(d_img,
                                                    width, height,
                                                    zoom, offset,
                                                    maxIter);
        checkCuda(cudaDeviceSynchronize(), "Kernel execution");

        // Komplexitäts-Kernel
        dim3 bc(TILE_W, TILE_H);
        dim3 gc(tilesX, tilesY);
        computeComplexity<<<gc, bc>>>(d_img, width, height, d_complexity);
        checkCuda(cudaDeviceSynchronize(), "Complexity kernel");

        // PBO unmapen
        checkCuda(cudaGraphicsUnmapResources(1, &cudaPbo, 0), "UnmapResources");

        // Komplexitätswerte auslesen und besten Tile finden
        checkCuda(cudaMemcpy(h_complexity.data(), d_complexity,
                             totalTiles*sizeof(float), cudaMemcpyDeviceToHost),
                  "Memcpy complexity");
        int bestIdx = 0;
        float bestScore = -1.0f;
        for (int i = 0; i < totalTiles; ++i) {
            if (h_complexity[i] > bestScore) {
                bestScore = h_complexity[i];
                bestIdx = i;
            }
        }
        int bestX = bestIdx % tilesX;
        int bestY = bestIdx / tilesX;
        // Zoom/Offset aktualisieren
        offset.x += ((bestX + 0.5f)*TILE_W - width*0.5f)/zoom;
        offset.y += ((bestY + 0.5f)*TILE_H - height*0.5f)/zoom;
        zoom *= 1.2f;

        // Rendern aus PBO
        glClear(GL_COLOR_BUFFER_BIT);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
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
