#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "settings.hpp"
#include "core_kernel.h"
#include "cuda_interop.hpp"
#include "opengl_utils.hpp"
#include "renderer_core.hpp"
#include "hud.hpp"
#include "memory_utils.hpp"
#include "progressive.hpp"
#include <iostream>
#include <vector>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include "stb_easy_font.h"

// Globals
static GLuint pbo = 0, tex = 0, program = 0, VAO = 0, VBO = 0, EBO = 0;
static cudaGraphicsResource* cudaPboRes = nullptr;
static double lastTime = 0.0;
static int frameCount = 0;
static float currentFPS = 0.0f, lastFrameTime = 0.0f;
static float zoom = Settings::initialZoom;
static float2 offset = {Settings::initialOffsetX, Settings::initialOffsetY};
static float* d_complexity = nullptr;
static int* d_iterations = nullptr;
static std::vector<float> h_complexity;

inline void CUDA_CHECK(cudaError_t err) { if (err != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; std::exit(EXIT_FAILURE); } }
inline void GL_CHECK() { if (GLenum err = glGetError(); err != GL_NO_ERROR) { std::cerr << "OpenGL error: 0x" << std::hex << err << std::dec << std::endl; std::exit(EXIT_FAILURE); } }

static constexpr char vertexShaderSrc[] = R"GLSL(
#version 430 core
layout(location=0) in vec2 aPos; layout(location=1) in vec2 aTex;
out vec2 vTex; void main() { vTex = aTex; gl_Position = vec4(aPos, 0.0, 1.0); }
)GLSL";

static constexpr char fragmentShaderSrc[] = R"GLSL(
#version 430 core
in vec2 vTex; out vec4 FragColor; uniform sampler2D uTex;
void main() { FragColor = texture(uTex, vTex); }
)GLSL";

Renderer::Renderer(int width, int height) : windowWidth(width), windowHeight(height), window(nullptr) {}
Renderer::~Renderer() {}

void Renderer::initGL() {
    CudaInterop::checkDynamicParallelismSupport();
    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; std::exit(EXIT_FAILURE); }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (!(window = glfwCreateWindow(windowWidth, windowHeight, "OtterDream Mandelbrot", nullptr, nullptr))) {
        std::cerr << "Window creation failed\n"; glfwTerminate(); std::exit(EXIT_FAILURE); }
    glfwMakeContextCurrent(window); glfwSwapInterval(1);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* w, int newW, int newH) {
        if (auto* self = static_cast<Renderer*>(glfwGetWindowUserPointer(w))) self->resize(newW, newH);
    });
    initGL_impl(window);
}

void Renderer::renderFrame() { renderFrame_impl(window); }
void Renderer::cleanup() { cleanup_impl(); if (window) { glfwDestroyWindow(window); glfwTerminate(); } }
bool Renderer::shouldClose() const { return (window && glfwWindowShouldClose(window)); }

void Renderer::resize(int newWidth, int newHeight) {
    if (newWidth <= 0 || newHeight <= 0) return;
    windowWidth = newWidth; windowHeight = newHeight;
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaPboRes));
    glDeleteBuffers(1, &pbo); glDeleteTextures(1, &tex);
    glGenBuffers(1, &pbo); glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, windowWidth * windowHeight * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPboRes, pbo, cudaGraphicsMapFlagsWriteDiscard));
    glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowWidth, windowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    CUDA_CHECK(cudaFree(d_complexity)); CUDA_CHECK(cudaFree(d_iterations));
    int totalTiles = ((windowWidth + Settings::TILE_W - 1) / Settings::TILE_W) * ((windowHeight + Settings::TILE_H - 1) / Settings::TILE_H);
    d_complexity = allocComplexityBuffer(totalTiles); h_complexity.resize(totalTiles);
    CUDA_CHECK(cudaMalloc(&d_iterations, windowWidth * windowHeight * sizeof(int)));
    glViewport(0, 0, windowWidth, windowHeight);
}

void Renderer::initGL_impl(GLFWwindow*) {
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::cerr << "GLEW init failed\n"; std::exit(EXIT_FAILURE); }
    cudaSetDevice(0); cudaGLSetGLDevice(0);
    glGenBuffers(1, &pbo); glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, windowWidth * windowHeight * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPboRes, pbo, cudaGraphicsMapFlagsWriteDiscard));
    glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowWidth, windowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
    program = createProgramFromSource(vertexShaderSrc, fragmentShaderSrc);
    glUseProgram(program); glUniform1i(glGetUniformLocation(program, "uTex"), 0); glUseProgram(0);
    createFullscreenQuad(&VAO, &VBO, &EBO); GL_CHECK();
    int totalTiles = ((windowWidth + Settings::TILE_W - 1) / Settings::TILE_W) * ((windowHeight + Settings::TILE_H - 1) / Settings::TILE_H);
    d_complexity = allocComplexityBuffer(totalTiles); h_complexity.resize(totalTiles);
    CUDA_CHECK(cudaMalloc(&d_iterations, windowWidth * windowHeight * sizeof(int)));
    lastTime = glfwGetTime(); frameCount = 0; currentFPS = 0.0f;
    glViewport(0, 0, windowWidth, windowHeight);
    Hud::init();
}

void Renderer::renderFrame_impl(GLFWwindow*) {
    double frameStart = glfwGetTime(); frameCount++;
    if (frameStart - lastTime >= 1.0) {
        currentFPS = float(frameCount / (frameStart - lastTime));
        frameCount = 0; lastTime = frameStart;
    }
    CudaInterop::renderCudaFrame(cudaPboRes, windowWidth, windowHeight, zoom, offset,
        getCurrentIterations(), d_complexity, h_complexity, d_iterations);
    if (!wasJustReset()) currentMaxIter = std::min(currentMaxIter + iterStep, iterMax);
    glBindTexture(GL_TEXTURE_2D, tex); glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, windowWidth, windowHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(program); glBindVertexArray(VAO); glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0); glUseProgram(0);
    lastFrameTime = float((glfwGetTime() - frameStart) * 1000.0f);
    Hud::draw(currentFPS, lastFrameTime, zoom, offset.x, offset.y, windowWidth, windowHeight);
    glfwSwapBuffers(window); glfwPollEvents();
}

void Renderer::cleanup_impl() {
    CUDA_CHECK(cudaFree(d_complexity)); CUDA_CHECK(cudaFree(d_iterations));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaPboRes));
    glDeleteBuffers(1, &pbo); glDeleteTextures(1, &tex); glDeleteProgram(program);
    deleteFullscreenQuad(&VAO, &VBO, &EBO); Hud::cleanup();
}
