// Datei: src/renderer_core.cu

// 1) Zuerst glew.h einbinden (verhindert "gl.h included before glew.h"-Fehler)
#include <GL/glew.h>

// 2) Direkt danach GLFW, ohne dass vorher gl.h geladen wird
#include <GLFW/glfw3.h>

// 3) Projekt‐Header
#include "settings.hpp"        // für Settings::TILE_W, Settings::TILE_H, initialZoom, maxIterations, width, height
#include "core_kernel.h"       // Deklaration von launch_mandelbrotHybrid, launch_debugGradient, computeComplexity (Kernel-Wrappers)
#include "cuda_interop.hpp"    // Deklaration von CudaInterop::renderCudaFrame(...)
#include "opengl_utils.hpp"    // Deklaration von createProgramFromSource, createFullscreenQuad, drawFullscreenQuad, deleteFullscreenQuad
#include "renderer_core.hpp"   // Enthält die Klassendeklaration von Renderer
#include "hud.hpp"             // Enthält Hud::draw(...)
#include "memory_utils.hpp"   // <---- FEHLT BISHER!

#include <iostream>
#include <vector>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// -------------------------------------------------------------
// STB Easy Font für schnelles Text-Rendering
#include "stb_easy_font.h"

// -------------------------------------------------------------
// Definiere WINDOW-BREITE und -HÖHE als Konstanten, basierend auf Settings:
static const int WIDTH  = Settings::width;
static const int HEIGHT = Settings::height;

// -------------------------------------------------------------
// Globale statische Variablen für OpenGL/CUDA-Ressourcen

static GLuint pbo            = 0;
static GLuint tex            = 0;
static cudaGraphicsResource* cudaPboRes = nullptr;
static GLuint program        = 0;
static GLuint VAO            = 0;
static GLuint VBO            = 0;
static GLuint EBO            = 0;

// FPS-Berechnung
static double lastTime       = 0.0;
static int    frameCount     = 0;
static float  currentFPS     = 0.0f;

// Zoom / Offset / Iterationen
static float   zoom          = Settings::initialZoom;
static float2  offset        = {0.0f, 0.0f};
static int     maxIter       = Settings::maxIterations;

// Complexity-Buffers
static float*            d_complexity = nullptr;
static std::vector<float> h_complexity;

// -------------------------------------------------------------
// Makros für Fehlerprüfung

#define CUDA_CHECK(call)                                                     \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA-Fehler in " << __FILE__ << ":" << __LINE__     \
                      << " -> " << cudaGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while(0)

#define CUDA_SYNC_CHECK()                                                     \
    do {                                                                      \
        cudaError_t err_sync = cudaDeviceSynchronize();                       \
        if (err_sync != cudaSuccess) {                                        \
            std::cerr << "CUDA-Sync-Fehler in " << __FILE__ << ":" << __LINE__\
                      << " -> " << cudaGetErrorString(err_sync) << std::endl; \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while(0)

#define GL_CHECK()                                                            \
    do {                                                                      \
        GLenum err = glGetError();                                            \
        if (err != GL_NO_ERROR) {                                             \
            std::cerr << "OpenGL-Fehler in " << __FILE__ << ":" << __LINE__   \
                      << " -> 0x" << std::hex << err << std::dec << std::endl;\
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while(0)

// -------------------------------------------------------------
// Shader-Quellen (Vertex + Fragment)

static const char* vertexShaderSrc = R"GLSL(
#version 430 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aTex;
out vec2 vTex;
void main(){
    vTex = aTex;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)GLSL";

static const char* fragmentShaderSrc = R"GLSL(
#version 430 core
in vec2 vTex;
out vec4 FragColor;
uniform sampler2D uTex;
void main(){
    FragColor = texture(uTex, vTex);
}
)GLSL";

// -------------------------------------------------------------
// Implementierung der Renderer-Klassenmethoden

Renderer::Renderer(int width, int height)
    : windowWidth(width), windowHeight(height), window(nullptr)
{
    // Leerer Konstruktor; Fenster wird in initGL() erzeugt
}

Renderer::~Renderer() {
    // Destruktor – Cleanup erfolgt manuell über cleanup()
}

void Renderer::initGL() {
    // GLFW initialisieren
    if (!glfwInit()) {
        std::cerr << "GLFW-Init fehlgeschlagen\n";
        std::exit(EXIT_FAILURE);
    }

    // OpenGL Version & Profil
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Fenster erzeugen
    window = glfwCreateWindow(windowWidth, windowHeight, "OtterDream Mandelbrot", nullptr, nullptr);
    if (!window) {
        std::cerr << "Fenster-Erstellung fehlgeschlagen\n";
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // V-Sync aktivieren

    // Hilfsfunktion aufrufen
    initGL_impl(window);
}

void Renderer::renderFrame() {
    renderFrame_impl(window);
}

void Renderer::cleanup() {
    cleanup_impl();
    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

bool Renderer::shouldClose() const {
    return (window && glfwWindowShouldClose(window));
}

// -------------------------------------------------------------
// Statische Helfer-Funktionen (mit korrekter Qualifikation)

void Renderer::initGL_impl(GLFWwindow* window) {
    // **GLEW** initialisieren (muss nach Kontext-Erzeugung passieren)
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW-Init fehlgeschlagen\n";
        std::exit(EXIT_FAILURE);
    }

    // --- 1) PBO erstellen + CUDA-GL-Interop ---
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPboRes, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // --- 2) Textur erstellen ---
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // --- 3) Shader-Programm kompilieren + linken ---
    program = createProgramFromSource(vertexShaderSrc, fragmentShaderSrc);

    // --- 4) Vollbild-Quad (VAO, VBO, EBO) initialisieren ---
    createFullscreenQuad(&VAO, &VBO, &EBO);
    GL_CHECK();

    // --- 5) Complexity-Buffer auf dem Device anlegen ---
    int tilesX     = (WIDTH  + Settings::TILE_W - 1) / Settings::TILE_W;
    int tilesY     = (HEIGHT + Settings::TILE_H - 1) / Settings::TILE_H;
    int totalTiles = tilesX * tilesY;
    d_complexity   = allocComplexityBuffer(totalTiles);
    h_complexity.resize(totalTiles);

    // FPS-Initialisierung
    lastTime   = glfwGetTime();
    frameCount = 0;
    currentFPS = 0.0f;
}

void Renderer::renderFrame_impl(GLFWwindow* window) {
    // 1) FPS berechnen
    double currentTime = glfwGetTime();
    frameCount++;
    if (currentTime - lastTime >= 1.0) {
        currentFPS  = float(frameCount / (currentTime - lastTime));
        frameCount  = 0;
        lastTime    = currentTime;
    }

    // 2) Gesamte CUDA-Pipeline auslagern
    CudaInterop::renderCudaFrame(
        cudaPboRes,
        WIDTH, HEIGHT,
        zoom, offset,
        maxIter,
        d_complexity, h_complexity
    );

    // 3) HUD (FPS + Zoom/Offset) zeichnen
    Hud::draw(currentFPS, zoom, offset.x, offset.y, WIDTH, HEIGHT);

    // 4) Buffers tauschen & Events abfragen
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Renderer::cleanup_impl() {
#if !DEBUG_GRADIENT
    CUDA_CHECK(cudaFree(d_complexity));
#endif
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaPboRes));
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    glDeleteProgram(program);
    deleteFullscreenQuad(&VAO, &VBO, &EBO);
}

// ------------------------------------------
// In diesem File dürfen **keine** __global__-Kernels mehr
// implementiert sein – diese liegen in core_kernel.cu.
// Hier rufen wir nur die externen Wrapper auf (bzw. die vollständig ausgelagerte CUDA-Pipeline):
//   - CudaInterop::renderCudaFrame(...)
//   - Hud::draw(...)
// ------------------------------------------
