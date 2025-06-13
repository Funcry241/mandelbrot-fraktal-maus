// 🐭 Maus-Kommentar: Zentrale Steuerung für OpenGL-Rendering, CUDA-Pipeline, HUD und Auto-Zoom.

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include "settings.hpp"
#include "core_kernel.h"
#include "cuda_interop.hpp"
#include "opengl_utils.hpp"
#include "renderer_core.hpp"
#include "hud.hpp"
#include "memory_utils.hpp"
#include "progressive.hpp"
#include "stb_easy_font.h"
#include <iostream>
#include <vector>

// 🔐 CUDA-Fehlerprüfung
inline void CUDA_CHECK(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

// 🔐 OpenGL-Fehlerprüfung
inline void GL_CHECK() {
    if (GLenum err = glGetError(); err != GL_NO_ERROR) {
        std::cerr << "OpenGL error: 0x" << std::hex << err << std::dec << '\n';
        std::exit(EXIT_FAILURE);
    }
}

// 🎨 Vertex-Shader: einfache Fullscreen-Quad-Pipeline
static constexpr const char* vertexShaderSrc = R"GLSL(
#version 430 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aTex;
out vec2 vTex;
void main() {
    vTex = aTex;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)GLSL";

// 🎨 Fragment-Shader: Textur direkt anzeigen
static constexpr const char* fragmentShaderSrc = R"GLSL(
#version 430 core
in vec2 vTex;
out vec4 FragColor;
uniform sampler2D uTex;
void main() {
    FragColor = texture(uTex, vTex);
}
)GLSL";

Renderer::Renderer(int width, int height)
    : windowWidth(width), windowHeight(height), window(nullptr),
      pbo(0), tex(0), program(0), VAO(0), VBO(0), EBO(0),
      d_complexity(nullptr), d_iterations(nullptr),
      zoom(Settings::initialZoom),
      offset{Settings::initialOffsetX, Settings::initialOffsetY},
      lastTime(0.0), frameCount(0), currentFPS(0.0f), lastFrameTime(0.0f) {}

Renderer::~Renderer() {
    if (d_complexity) CUDA_CHECK(cudaFree(d_complexity));
    if (d_iterations) CUDA_CHECK(cudaFree(d_iterations));
    CudaInterop::unregisterPBO();
    if (pbo) glDeleteBuffers(1, &pbo);
    if (tex) glDeleteTextures(1, &tex);
    if (program) glDeleteProgram(program);
    OpenGLUtils::deleteFullscreenQuad(&VAO, &VBO, &EBO);
    Hud::cleanup();
    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

void Renderer::initGL() {
    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        std::exit(EXIT_FAILURE);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(windowWidth, windowHeight, "OtterDream Mandelbrot", nullptr, nullptr);
    if (!window) {
        std::cerr << "Window creation failed\n";
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* w, int newW, int newH) {
        if (auto* self = static_cast<Renderer*>(glfwGetWindowUserPointer(w)))
            self->resize(newW, newH);
    });
    initGL_impl();
}

bool Renderer::shouldClose() const {
    return (window && glfwWindowShouldClose(window));
}

void Renderer::resize(int newWidth, int newHeight) {
    if (newWidth <= 0 || newHeight <= 0) return;
    windowWidth = newWidth;
    windowHeight = newHeight;

    if (pbo) glDeleteBuffers(1, &pbo);
    if (tex) glDeleteTextures(1, &tex);

    setupPBOAndTexture();
    setupBuffers();
    glViewport(0, 0, windowWidth, windowHeight);
}

GLFWwindow* Renderer::getWindow() const {
    return window;
}

void Renderer::initGL_impl() {
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW init failed\n";
        std::exit(EXIT_FAILURE);
    }
    cudaSetDevice(0);
    cudaGLSetGLDevice(0);

    setupPBOAndTexture();
    program = OpenGLUtils::createProgramFromSource(vertexShaderSrc, fragmentShaderSrc);
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "uTex"), 0);
    glUseProgram(0);
    OpenGLUtils::createFullscreenQuad(&VAO, &VBO, &EBO);
    GL_CHECK();
    setupBuffers();
    lastTime = glfwGetTime();
    frameCount = 0;
    currentFPS = 0.0f;
    glViewport(0, 0, windowWidth, windowHeight);
    Hud::init();
}

void Renderer::renderFrame_impl(bool autoZoomEnabled) {
    static int lastTileSize = -1;

    double frameStart = glfwGetTime();
    frameCount++;

    if (frameStart - lastTime >= 1.0) {
        currentFPS = static_cast<float>(frameCount) / static_cast<float>(frameStart - lastTime);
        frameCount = 0;
        lastTime = frameStart;
    }

    int currentTileSize = Settings::dynamicTileSize(zoom);
    if (currentTileSize != lastTileSize && Settings::debugLogging)
        std::printf("[DEBUG] TileSize changed to %d\n", currentTileSize);
    lastTileSize = currentTileSize;

    float2 newOffset;
    bool shouldZoom;

    CudaInterop::renderCudaFrame(
        nullptr,  // pointer wird innerhalb von renderCudaFrame() gemappt
        d_iterations,
        d_complexity,
        nullptr,
        windowWidth,
        windowHeight,
        zoom,
        offset,
        Progressive::getCurrentIterations(),
        h_complexity,
        newOffset,
        shouldZoom,
        currentTileSize
    );

    if (autoZoomEnabled && shouldZoom) {
        offset = newOffset;
        zoom *= 1.05f;
    }

    glBindTexture(GL_TEXTURE_2D, tex);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, windowWidth, windowHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(program);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    glUseProgram(0);

    lastFrameTime = static_cast<float>((glfwGetTime() - frameStart) * 1000.0);
    Hud::draw(currentFPS, lastFrameTime, zoom, offset.x, offset.y, windowWidth, windowHeight);
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Renderer::setupPBOAndTexture() {
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, windowWidth * windowHeight * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CudaInterop::registerPBO(pbo); // 🟢 Neue Registrierung

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowWidth, windowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::setupBuffers() {
    constexpr int minTileSize = Settings::MIN_TILE_SIZE;
    int totalTiles = ((windowWidth + minTileSize - 1) / minTileSize) *
                     ((windowHeight + minTileSize - 1) / minTileSize);
    d_complexity = MemoryUtils::allocComplexityBuffer(totalTiles);
    h_complexity.resize(totalTiles);
    CUDA_CHECK(cudaMalloc(&d_iterations, windowWidth * windowHeight * sizeof(int)));
}

void Renderer::renderFrame(bool autoZoomEnabled) {
    renderFrame_impl(autoZoomEnabled);
}
