// Datei: src/main.cpp

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "settings.hpp"
#include "cuda_interop.hpp"

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
        std::cerr << "[GL ERROR] " << msg << ": 0x" << std::hex << glErr << std::dec << std::endl; \
    } \
}

// Fenstergröße
constexpr int WIDTH  = 800;
constexpr int HEIGHT = 600;

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

// Shader-Hilfsfunktionen
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "[SHADER ERROR] " << infoLog << std::endl;
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

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "[PROGRAM LINK ERROR] " << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

void initGL() {
    std::cout << "[INFO] Initialisiere GLEW...\n";
    glewExperimental = GL_TRUE;
    glewInit();
    CHECK_GL_ERROR("nach glewInit");

    std::cout << "[INFO] Shader erstellen...\n";
    shaderProgram = createShaderProgram(vertexShaderSrc, fragmentShaderSrc);

    std::cout << "[INFO] Erstelle Fullscreen Quad...\n";
    float quadVertices[] = {
        // Position   // TexCoord
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
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
    glUniform1i(glGetUniformLocation(shaderProgram, "uTex"), 0); // Texture unit 0
    glUseProgram(0);

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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
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

    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    glUseProgram(0);

    CHECK_GL_ERROR("nach Frame Rendering");
}

void cleanup() {
    std::cout << "[INFO] Cleaning up...\n";
    cudaGraphicsUnregisterResource(cudaPboRes);
    cudaFree(d_complexity);

    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &pbo);

    glDeleteProgram(shaderProgram);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteVertexArrays(1, &VAO);
    std::cout << "[INFO] Cleanup done.\n";
}

int main() {
    std::cout << "[INFO] Starte GLFW...\n";
    if (!glfwInit()) {
        std::cerr << "[FATAL] GLFW-Initialisierung fehlgeschlagen!\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    std::cout << "[INFO] Erstelle Fenster...\n";
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Mandelbrot OtterDream", nullptr, nullptr);
    if (!window) {
        std::cerr << "[FATAL] Fenstererstellung fehlgeschlagen!\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;

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
