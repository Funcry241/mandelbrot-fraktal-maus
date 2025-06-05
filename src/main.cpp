// Datei: src/main.cpp

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "settings.hpp"
#include "cuda_interop.hpp"
#include "hud.hpp"

// Shader Sources
const char* vertexShaderSrc = R"(
#version 430 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTex;
out vec2 vTex;
void main() {
    vTex = aTex;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

const char* fragmentShaderSrc = R"(
#version 430 core
in vec2 vTex;
out vec4 FragColor;
uniform sampler2D uTex;
void main() {
    FragColor = texture(uTex, vTex);
}
)";

// 🐭 Debug-Utilities
inline void debugLog(const char* message) {
    if (Settings::debugLogging) {
        std::cout << message << std::endl;
    }
}

inline void debugError(const char* message) {
    if (Settings::debugLogging) {
        std::cerr << message << std::endl;
    }
}

#define CHECK_CUDA_ERROR(msg) { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess && Settings::debugLogging) { \
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl; \
    } \
}

#define CHECK_GL_ERROR(msg) { \
    GLenum glErr; \
    while ((glErr = glGetError()) != GL_NO_ERROR) { \
        if (Settings::debugLogging) { \
            std::cerr << "[GL ERROR] " << msg << ": 0x" << std::hex << glErr << std::dec << std::endl; \
        } \
    } \
}

// OpenGL / CUDA Ressourcen
static cudaGraphicsResource* cudaPboRes = nullptr;
static GLuint pbo = 0;
static GLuint tex = 0;
static GLuint VAO = 0, VBO = 0, EBO = 0;
static GLuint shaderProgram = 0;

std::vector<float> h_complexity;
float* d_complexity = nullptr;

float zoom   = 1.0f;
float2 offset = {0.0f, 0.0f};
int maxIter = 500;

// 🐭 FPS Tracking
static double lastTime = 0.0;
static int    frameCount = 0;
static float  currentFPS = 0.0f;

// Shader-Hilfsfunktionen
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        debugError(infoLog);
    }
    return shader;
}

GLuint createShaderProgram(const char* vertexSrc, const char* fragmentSrc) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        debugError(infoLog);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

void initGL() {
    debugLog("[INFO] Initialisiere GLEW...");
    glewExperimental = GL_TRUE;
    glewInit();
    CHECK_GL_ERROR("nach glewInit");

    Hud::init();

    debugLog("[INFO] Shader erstellen...");
    shaderProgram = createShaderProgram(vertexShaderSrc, fragmentShaderSrc);

    debugLog("[INFO] Erstelle Fullscreen Quad...");
    float quadVertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f
    };
    unsigned int quadIndices[] = { 0, 1, 2, 2, 3, 0 };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), quadIndices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    glUseProgram(shaderProgram);
    glUniform1i(glGetUniformLocation(shaderProgram, "uTex"), 0);
    glUseProgram(0);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, Settings::width * Settings::height * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&cudaPboRes, pbo, cudaGraphicsMapFlagsWriteDiscard);
    CHECK_CUDA_ERROR("cudaGraphicsGLRegisterBuffer");

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, Settings::width, Settings::height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    int tilesX = (Settings::width  + Settings::TILE_W - 1) / Settings::TILE_W;
    int tilesY = (Settings::height + Settings::TILE_H - 1) / Settings::TILE_H;
    int totalTiles = tilesX * tilesY;
    h_complexity.resize(totalTiles, 0.0f);
    cudaMalloc(&d_complexity, totalTiles * sizeof(float));
    CHECK_CUDA_ERROR("cudaMalloc d_complexity");

    glViewport(0, 0, Settings::width, Settings::height);
}

void render() {
    double currentTime = glfwGetTime();
    frameCount++;
    if (currentTime - lastTime >= 1.0) {
        currentFPS = float(frameCount / (currentTime - lastTime));
        frameCount = 0;
        lastTime = currentTime;
    }

    CudaInterop::renderCudaFrame(
        cudaPboRes, Settings::width, Settings::height, zoom, offset, maxIter, d_complexity, h_complexity
    );
    CHECK_CUDA_ERROR("nach renderCudaFrame");

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, Settings::width, Settings::height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    CHECK_GL_ERROR("nach glTexSubImage2D");

    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    glUseProgram(0);

    Hud::draw(currentFPS, zoom, offset.x, offset.y, Settings::width, Settings::height);

    CHECK_GL_ERROR("nach Frame Rendering");
}

void cleanup() {
    debugLog("[INFO] Cleaning up...");
    cudaGraphicsUnregisterResource(cudaPboRes);
    cudaFree(d_complexity);

    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &pbo);

    glDeleteProgram(shaderProgram);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteVertexArrays(1, &VAO);

    Hud::cleanup();
    debugLog("[INFO] Cleanup done.");
}

int main() {
    debugLog("[INFO] Starte GLFW...");
    if (!glfwInit()) {
        debugError("[FATAL] GLFW-Initialisierung fehlgeschlagen!");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    debugLog("[INFO] Erstelle Fenster...");
    GLFWwindow* window = glfwCreateWindow(Settings::width, Settings::height, "Mandelbrot OtterDream", nullptr, nullptr);
    if (!window) {
        debugError("[FATAL] Fenstererstellung fehlgeschlagen!");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    int device = 0;
    cudaSetDevice(device);
    cudaGLSetGLDevice(device);

    initGL();

    debugLog("[INFO] Starte Render-Loop...");
    while (!glfwWindowShouldClose(window)) {
        render();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();

    debugLog("[INFO] Programm beendet.");
    return 0;
}
