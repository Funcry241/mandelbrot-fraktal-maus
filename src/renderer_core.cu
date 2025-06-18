// Datei: src/renderer_core.cu
// Zeilen: 274
// 🐭 Maus-Kommentar: Alpha 4.2 – Auto-Zoom korrigiert: Kamera zoomt und bewegt sich zu interessanter Region. Offset & Zoomfaktor werden übernommen. Kommentarzeile aktualisiert. Schneefuchs würde sagen: „Endlich bewegt sich das Universum, nicht nur der Blick.“

#include "pch.hpp"

#include "settings.hpp"
#include "core_kernel.h"
#include "cuda_interop.hpp"
#include "opengl_utils.hpp"
#include "renderer_core.hpp"
#include "hud.hpp"
#include "progressive.hpp"
#include "stb_easy_font.h"
#include "common.hpp"

namespace {
const auto GL_CHECK = [] {
    if (GLenum err = glGetError(); err != GL_NO_ERROR) {
        std::cerr << "OpenGL error: 0x" << std::hex << err << std::dec << '\n';
        std::exit(EXIT_FAILURE);
    }
};
}

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
      d_entropy(nullptr), d_iterations(nullptr),
      zoom(Settings::initialZoom),
      offset{Settings::initialOffsetX, Settings::initialOffsetY},
      lastTime(0.0), frameCount(0), currentFPS(0.0f), lastFrameTime(0.0f),
      lastTileSize(-1) {
    CudaInterop::setPauseZoom(false);
    std::printf("[DEBUG] Auto-Zoom ist aktuell: %s\n", CudaInterop::getPauseZoom() ? "PAUSIERT" : "AKTIV");
}

Renderer::~Renderer() {
    freeDeviceBuffers();
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

void Renderer::freeDeviceBuffers() {
    if (d_entropy) {
        CUDA_CHECK(cudaFree(d_entropy));
        d_entropy = nullptr;
    }
    if (d_iterations) {
        CUDA_CHECK(cudaFree(d_iterations));
        d_iterations = nullptr;
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
    glfwSetKeyCallback(window, CudaInterop::keyCallback);

    initGL_impl();
}

bool Renderer::shouldClose() const {
    return window && glfwWindowShouldClose(window);
}

void Renderer::resize(int newWidth, int newHeight) {
    if (newWidth <= 0 || newHeight <= 0) return;
    windowWidth = newWidth;
    windowHeight = newHeight;

    if (pbo) {
        CudaInterop::unregisterPBO();
        glDeleteBuffers(1, &pbo);
        pbo = 0;
    }
    if (tex) {
        glDeleteTextures(1, &tex);
        tex = 0;
    }
    freeDeviceBuffers();
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
    double frameStart = glfwGetTime();
    frameCount++;

    if (frameStart - lastTime >= 1.0) {
        currentFPS = frameCount / static_cast<float>(frameStart - lastTime);
        frameCount = 0;
        lastTime = frameStart;
    }

    int currentTileSize = Settings::dynamicTileSize(zoom);
    if (currentTileSize != lastTileSize) {
        if (Settings::debugLogging)
            std::printf("[DEBUG] TileSize changed to %d\n", currentTileSize);
        freeDeviceBuffers();
        setupBuffers();
        lastTileSize = currentTileSize;
    }

    float2 newOffset;
    bool shouldZoom;

    CudaInterop::renderCudaFrame(
        d_iterations,
        d_entropy,
        windowWidth,
        windowHeight,
        zoom,
        offset,
        Progressive::getCurrentIterations(),
        h_entropy,
        newOffset,
        shouldZoom,
        currentTileSize
    );

    // 🚜 Zielpunkt glätten, damit Zoombewegung ruhig bleibt
    static float2 smoothedTarget = newOffset;
    smoothedTarget.x += (newOffset.x - smoothedTarget.x) * 0.1f;
    smoothedTarget.y += (newOffset.y - smoothedTarget.y) * 0.1f;

    if (!CudaInterop::getPauseZoom()) {
        zoom *= Settings::AUTOZOOM_SPEED;
        offset.x += (smoothedTarget.x - offset.x) * Settings::LERP_FACTOR;
        offset.y += (smoothedTarget.y - offset.y) * Settings::LERP_FACTOR;
        Progressive::incrementIterations();
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
    if (pbo) {
        CudaInterop::unregisterPBO();
        glDeleteBuffers(1, &pbo);
    }
    if (tex) {
        glDeleteTextures(1, &tex);
    }

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, windowWidth * windowHeight * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CudaInterop::registerPBO(pbo);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowWidth, windowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::setupBuffers() {
    int currentTileSize = Settings::dynamicTileSize(zoom);
    int totalTiles = ((windowWidth + currentTileSize - 1) / currentTileSize)
                     * ((windowHeight + currentTileSize - 1) / currentTileSize);

    std::printf("[DEBUG] setupBuffers: TileSize=%d → totalTiles=%d\n", currentTileSize, totalTiles);

    CUDA_CHECK(cudaMalloc(&d_entropy, totalTiles * sizeof(float)));
    h_entropy.resize(totalTiles);

    CUDA_CHECK(cudaMalloc(&d_iterations, windowWidth * windowHeight * sizeof(int)));
}

void Renderer::renderFrame(bool autoZoomEnabled) {
    renderFrame_impl(autoZoomEnabled);
}
