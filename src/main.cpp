// Datei: src/main.cpp

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include "settings.hpp"
#include "cuda_interop.hpp"

// Debug-Utilities
#define CHECK_CUDA_ERROR(msg) { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl; \
    } \
}
#define CHECK_GL_ERROR(msg) { \
    GLenum glErr; \
    while ((glErr = glGetError()) != GL_NO_ERROR) { \
        std::cerr << "[GL ERROR] " << msg << ": " << std::hex << glErr << std::endl; \
    } \
}

// Fenstergröße
constexpr int WIDTH  = 800;
constexpr int HEIGHT = 600;

static cudaGraphicsResource* cudaPboRes = nullptr;
static GLuint pbo = 0;
static GLuint tex = 0;

std::vector<float> h_complexity;
float* d_complexity = nullptr;

float zoom   = 1.0f;
float2 offset = {0.0f, 0.0f};
int maxIter = 500;

void initGL() {
    std::cout << "[INFO] Initialisiere GLEW...\n";
    glewInit();
    CHECK_GL_ERROR("nach glewInit");

    std::cout << "[INFO] Erstelle PBO...\n";
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    CHECK_GL_ERROR("nach PBO Setup");

    std::cout << "[INFO] Registriere PBO bei CUDA...\n";
    cudaGraphicsGLRegisterBuffer(&cudaPboRes, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
    CHECK_CUDA_ERROR("nach cudaGraphicsGLRegisterBuffer");

    std::cout << "[INFO] Erstelle Textur...\n";
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    CHECK_GL_ERROR("nach Textur Setup");

    std::cout << "[INFO] Alloziere Complexity Buffer...\n";
    int tilesX = (WIDTH  + Settings::TILE_W - 1) / Settings::TILE_W;
    int tilesY = (HEIGHT + Settings::TILE_H - 1) / Settings::TILE_H;
    int totalTiles = tilesX * tilesY;
    h_complexity.resize(totalTiles, 0.0f);
    cudaMalloc(&d_complexity, totalTiles * sizeof(float));
    CHECK_CUDA_ERROR("nach cudaMalloc Complexity Buffer");

    std::cout << "[INFO] OpenGL und CUDA initialisiert.\n";
}

void render() {
    CudaInterop::renderCudaFrame(
        cudaPboRes,
        WIDTH,
        HEIGHT,
        zoom,
        offset,
        maxIter,
        d_complexity,
        h_complexity
    );
    CHECK_CUDA_ERROR("nach renderCudaFrame");

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    CHECK_GL_ERROR("nach glTexSubImage2D");

    glClear(GL_COLOR_BUFFER_BIT);
    CHECK_GL_ERROR("nach glClear");

    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(-1, -1);
    glTexCoord2f(1, 0); glVertex2f( 1, -1);
    glTexCoord2f(1, 1); glVertex2f( 1,  1);
    glTexCoord2f(0, 1); glVertex2f(-1,  1);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    CHECK_GL_ERROR("nach Frame Rendering");
}

void cleanup() {
    std::cout << "[INFO] Cleaning up...\n";
    cudaGraphicsUnregisterResource(cudaPboRes);
    cudaFree(d_complexity);

    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &pbo);
    std::cout << "[INFO] Cleanup done.\n";
}

int main() {
    std::cout << "[INFO] Starte GLFW...\n";
    if (!glfwInit()) {
        std::cerr << "[FATAL] GLFW-Initialisierung fehlgeschlagen!\n";
        return -1;
    }

    std::cout << "[INFO] Erstelle Fenster...\n";
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Mandelbrot OtterDream", nullptr, nullptr);
    if (!window) {
        std::cerr << "[FATAL] Fenstererstellung fehlgeschlagen!\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    initGL();

    std::cout << "[INFO] Starte Render-Loop...\n";
    while (!glfwWindowShouldClose(window)) {
        render();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "[INFO] Programm beendet.\n";
    return 0;
}
