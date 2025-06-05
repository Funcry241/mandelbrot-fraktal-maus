// Datei: src/renderer_core.cu

// 🐭 Maus-Kommentar: Jetzt mit dynamischem glViewport für Fenster-Resize und fixiertem Texture-Binding und CUDA-Device-Set!

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "settings.hpp"
#include "core_kernel.h"
#include "cuda_interop.hpp"
#include "opengl_utils.hpp"
#include "renderer_core.hpp"
#include "hud.hpp"                // 🐭 HUD
#include "memory_utils.hpp"
#include "progressive.hpp"

#include <iostream>
#include <vector>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "stb_easy_font.h"

// -------------------------------------------------------------
// Definiere Fenstergröße
static int WIDTH  = Settings::width;
static int HEIGHT = Settings::height;

// -------------------------------------------------------------
// Globale Ressourcen

static GLuint pbo            = 0;
static GLuint tex            = 0;
static cudaGraphicsResource* cudaPboRes = nullptr;
static GLuint program        = 0;
static GLuint VAO            = 0;
static GLuint VBO            = 0;
static GLuint EBO            = 0;

// FPS
static double lastTime       = 0.0;
static int    frameCount     = 0;
static float  currentFPS     = 0.0f;

// Zoom / Offset
static float   zoom          = Settings::initialZoom;
static float2  offset        = {Settings::initialOffsetX, Settings::initialOffsetY};

// Complexity-Buffer
static float*            d_complexity = nullptr;
static std::vector<float> h_complexity;

// -------------------------------------------------------------
// Fehlerprüfungs-Makros

#define CUDA_CHECK(call) \
    do { cudaError_t err = call; if (err != cudaSuccess) { \
        std::cerr << "CUDA-Fehler in " << __FILE__ << ":" << __LINE__ \
                  << " -> " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); }} while(0)

#define GL_CHECK() \
    do { GLenum err = glGetError(); if (err != GL_NO_ERROR) { \
        std::cerr << "OpenGL-Fehler in " << __FILE__ << ":" << __LINE__ \
                  << " -> 0x" << std::hex << err << std::dec << std::endl; std::exit(EXIT_FAILURE); }} while(0)

// -------------------------------------------------------------
// Shader

static const char* vertexShaderSrc = R"GLSL(
#version 430 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aTex;
out vec2 vTex;
void main() {
    vTex = aTex;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)GLSL";

static const char* fragmentShaderSrc = R"GLSL(
#version 430 core
in vec2 vTex;
out vec4 FragColor;
uniform sampler2D uTex;
void main() {
    FragColor = texture(uTex, vTex);
}
)GLSL";

// -------------------------------------------------------------
// Renderer-Methoden

Renderer::Renderer(int width, int height)
    : windowWidth(width), windowHeight(height), window(nullptr)
{
}

Renderer::~Renderer() {
}

void Renderer::initGL() {
    if (!glfwInit()) {
        std::cerr << "GLFW-Init fehlgeschlagen\n";
        std::exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(windowWidth, windowHeight, "OtterDream Mandelbrot", nullptr, nullptr);
    if (!window) {
        std::cerr << "Fenster-Erstellung fehlgeschlagen\n";
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    initGL_impl(window);

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int w, int h) {
        glViewport(0, 0, w, h);
    });
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

void Renderer::initGL_impl(GLFWwindow* window) {
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW-Init fehlgeschlagen\n";
        std::exit(EXIT_FAILURE);
    }

    // 🐭 Setze CUDA-Device für OpenGL-Interop
    cudaSetDevice(0);
    cudaGLSetGLDevice(0);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPboRes, pbo, cudaGraphicsMapFlagsWriteDiscard));

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    program = createProgramFromSource(vertexShaderSrc, fragmentShaderSrc);

    // 🐭 Binde Sampler-Uniform 'uTex' auf Texture Unit 0
    glUseProgram(program);
    GLint texLoc = glGetUniformLocation(program, "uTex");
    glUniform1i(texLoc, 0);  // uTex -> GL_TEXTURE0
    glUseProgram(0);

    createFullscreenQuad(&VAO, &VBO, &EBO);
    GL_CHECK();

    int tilesX     = (WIDTH  + Settings::TILE_W - 1) / Settings::TILE_W;
    int tilesY     = (HEIGHT + Settings::TILE_H - 1) / Settings::TILE_H;
    int totalTiles = tilesX * tilesY;
    d_complexity   = allocComplexityBuffer(totalTiles);
    h_complexity.resize(totalTiles);

    lastTime   = glfwGetTime();
    frameCount = 0;
    currentFPS = 0.0f;

    glViewport(0, 0, WIDTH, HEIGHT);

    // 🐭 HUD initialisieren
    Hud::init();
}

void Renderer::renderFrame_impl(GLFWwindow* window) {
    double currentTime = glfwGetTime();
    frameCount++;
    if (currentTime - lastTime >= 1.0) {
        currentFPS  = float(frameCount / (currentTime - lastTime));
        frameCount  = 0;
        lastTime    = currentTime;
    }

    CudaInterop::renderCudaFrame(
        cudaPboRes,
        WIDTH, HEIGHT,
        zoom, offset,
        currentMaxIter,               // Progressive Iterations
        d_complexity, h_complexity
    );

    // Progressive Iteration erhöhen
    currentMaxIter = std::min(currentMaxIter + iterStep, iterMax);

    // PBO → Texture kopieren
    glBindTexture(GL_TEXTURE_2D, tex);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(program);
    glBindVertexArray(VAO);
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    glUseProgram(0);

    Hud::draw(currentFPS, zoom, offset.x, offset.y, WIDTH, HEIGHT);

    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Renderer::cleanup_impl() {
    CUDA_CHECK(cudaFree(d_complexity));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaPboRes));
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    glDeleteProgram(program);
    deleteFullscreenQuad(&VAO, &VBO, &EBO);

    // 🐭 HUD Ressourcen freigeben
    Hud::cleanup();
}
