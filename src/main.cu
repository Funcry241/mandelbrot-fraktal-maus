// File: src/main.cu
// Variante 2: Modern OpenGL-Pipeline mit Shadern und VAO/VBO für Fullscreen-Quad

#include "core_kernel.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <vector_types.h>
#include <vector_functions.h>

#include <iostream>
#include <vector>
#include <cstdlib>

// CUDA-Error-Check
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Komplexitäts-Kernel (unverändert)
__global__ void computeComplexity(const uchar4* img,
                                  int width, int height,
                                  float* complexity);

// Vertex-Shader (Fullscreen-Quad)
static const char* vertexShaderSrc = R"glsl(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)glsl";

// Fragment-Shader (Textur-Sampling)
static const char* fragmentShaderSrc = R"glsl(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTex;
void main() {
    FragColor = texture(uTex, TexCoord);
}
)glsl";

int main() {
    const int width  = 1024;
    const int height = 768;
    size_t imgBytes  = width * height * sizeof(uchar4);

    float zoom    = 300.0f;
    float2 offset = make_float2(0.0f, 0.0f);
    int   maxIter = 500;

    // GLFW + OpenGL Core-Profile
    if (!glfwInit()) std::exit(EXIT_FAILURE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(width, height, "Auto-Zoom Mandelbrot", nullptr, nullptr);
    if (!window) std::exit(EXIT_FAILURE);
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW init failed\n";
        return EXIT_FAILURE;
    }

    // --- PBO + CUDA-GL Interop ---
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, imgBytes, nullptr, GL_DYNAMIC_DRAW);

    cudaGraphicsResource* cudaPbo = nullptr;
    checkCuda(cudaGraphicsGLRegisterBuffer(&cudaPbo, pbo, cudaGraphicsMapFlagsWriteDiscard),
              "cudaGraphicsGLRegisterBuffer");

    // --- Texture ---
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // --- Shader-Programm erstellen ---
    auto compileShader = [&](GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char buf[512];
            glGetShaderInfoLog(s, 512, nullptr, buf);
            std::cerr << "Shader-Compile-Error:\n" << buf << std::endl;
            std::exit(EXIT_FAILURE);
        }
        return s;
    };

    GLuint vs = compileShader(GL_VERTEX_SHADER,   vertexShaderSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    {
        GLint ok;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) {
            char buf[512];
            glGetProgramInfoLog(program, 512, nullptr, buf);
            std::cerr << "Program-Link-Error:\n" << buf << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    glDeleteShader(vs);
    glDeleteShader(fs);

    // --- Fullscreen-Quad (VAO/VBO) ---
    float quadVerts[] = {
        // positions    // texcoords
        -1.0f, -1.0f,   0.0f, 0.0f,
         1.0f, -1.0f,   1.0f, 0.0f,
        -1.0f,  1.0f,   0.0f, 1.0f,
         1.0f,  1.0f,   1.0f, 1.0f,
    };
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
      glBindBuffer(GL_ARRAY_BUFFER, VBO);
      glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
      glEnableVertexAttribArray(1);
      glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
      glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // --- Complexity-Buffer ---
    int tilesX     = (width  + TILE_W - 1) / TILE_W;
    int tilesY     = (height + TILE_H - 1) / TILE_H;
    int totalTiles = tilesX * tilesY;
    float* d_complexity = nullptr;
    checkCuda(cudaMalloc(&d_complexity, totalTiles * sizeof(float)), "cudaMalloc complexity");
    std::vector<float> h_complexity(totalTiles);

    // Haupt-Loop
    while (!glfwWindowShouldClose(window)) {
        // Reset Complexity
        checkCuda(cudaMemset(d_complexity, 0, totalTiles * sizeof(float)), "memset complexity");

        // PBO → CUDA
        uchar4* d_img = nullptr;
        size_t sz = 0;
        checkCuda(cudaGraphicsMapResources(1, &cudaPbo, 0), "MapResources");
        checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_img, &sz, cudaPbo),
                  "GetMappedPointer");

        // ** Hier: mandelbrotHybrid statt mandelbrotPersistent **
        dim3 blockDim(TILE_W, TILE_H), gridDim(tilesX, tilesY);
        mandelbrotHybrid<<<gridDim, blockDim>>>(d_img, width, height, zoom, offset, maxIter);
        checkCuda(cudaDeviceSynchronize(), "mandelbrotHybrid");

        // Complexity-Kernel
        computeComplexity<<<gridDim, blockDim>>>(d_img, width, height, d_complexity);
        checkCuda(cudaDeviceSynchronize(), "computeComplexity");

        // Unmap PBO
        checkCuda(cudaGraphicsUnmapResources(1, &cudaPbo, 0), "UnmapResources");

        // Best Tile finden
        checkCuda(cudaMemcpy(h_complexity.data(), d_complexity,
                             totalTiles * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "Memcpy complexity");
        int bestIdx = 0; float bestScore = -1.0f;
        for (int i = 0; i < totalTiles; ++i) {
            if (h_complexity[i] > bestScore) {
                bestScore = h_complexity[i];
                bestIdx = i;
            }
        }
        int bestX = bestIdx % tilesX, bestY = bestIdx / tilesX;
        offset.x += ((bestX + 0.5f) * TILE_W - width  * 0.5f) / zoom;
        offset.y += ((bestY + 0.5f) * TILE_H - height * 0.5f) / zoom;
        zoom *= 1.2f;

        // --- Rendern via Shader ---
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program);
        glBindVertexArray(VAO);

        // Textur aus PBO
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glUniform1i(glGetUniformLocation(program, "uTex"), 0);

        // Draw Quad
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    cudaFree(d_complexity);
    cudaGraphicsUnregisterResource(cudaPbo);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    glDeleteProgram(program);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
