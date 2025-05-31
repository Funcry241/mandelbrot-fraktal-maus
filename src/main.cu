// Datei: src/main.cu
// Maus-Kommentar:
// 1) Debug-Modus: Einfache Gradient-Ausgabe, um PBO→Texture→Quad-Pipeline zu prüfen.
// 2) Sobald der Gradient klappt, können wir wieder zu Mandelbrot™ zurückschalten.
//    Dazu am Anfang DEBUG_GRADIENT auf 0 setzen.
//
// Der untenstehende Code kennt zwei Modi:
//   - DEBUG_GRADIENT == 1 → testKernel() füllt das Bild mit einem X/Y-Gradient.
//   - DEBUG_GRADIENT == 0 → launch_mandelbrotHybrid() füllt das Bild wie gewohnt.

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
#include <cmath>
#include <limits>

// -------------------------------------------------------------
// Fehlerprüfung Makros
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA Fehler in " << __FILE__ << ":" << __LINE__ \
                      << " -> " << cudaGetErrorString(err) << std::endl;   \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while(0)

#define GL_CHECK()                                                         \
    do {                                                                   \
        GLenum err = glGetError();                                         \
        if (err != GL_NO_ERROR) {                                          \
            std::cerr << "OpenGL Fehler in " << __FILE__ << ":"           \
                      << __LINE__ << " -> 0x" << std::hex << err << std::endl; \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while(0)

// -------------------------------------------------------------
// Umschalter für Debug-Gradient
#define DEBUG_GRADIENT 1

// -------------------------------------------------------------
// Shader-Quellen
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
// Shader-Helpers
GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[512]; glGetShaderInfoLog(s, 512, nullptr, buf);
        std::cerr << "Shader-Compile-Error: " << buf << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return s;
}

GLuint createProgram() {
    GLuint v = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[512]; glGetProgramInfoLog(p, 512, nullptr, buf);
        std::cerr << "Program-Link-Error: " << buf << std::endl;
        std::exit(EXIT_FAILURE);
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

// -------------------------------------------------------------
// Prototyp für Complexity-Kernel (wird im echten Mandelbrot-Modus gebraucht)
__global__ void computeComplexity(const uchar4* img,
                                  int width, int height,
                                  float* complexity);

// -------------------------------------------------------------
// Debug / Test-Kernel: füllt IMG mit einem einfachen X/Y-Gradient
#if DEBUG_GRADIENT
__global__ void testKernel(uchar4* img, int width, int height) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= width || ty >= height) return;
    // einfacher Farbverlauf: R = (tx % 256), G = (ty % 256), B = 0
    unsigned char r = static_cast<unsigned char>(tx & 0xFF);
    unsigned char g = static_cast<unsigned char>(ty & 0xFF);
    img[ty * width + tx] = make_uchar4(r, g, 0, 255);
}
#endif

// -------------------------------------------------------------
// Echt-Kernel-Prototyp für Mandelbrot (im Nicht-Debug-Modus)
#if !DEBUG_GRADIENT
extern "C" void launch_mandelbrotHybrid(uchar4* img,
                                        int w, int h,
                                        float zoom, float2 offset,
                                        int maxIter);
#endif

// -------------------------------------------------------------
int main() {
    std::cout << "=== Programm gestartet ===\n";

    // --- Bild-Settings ---
    const int W = 1024, H = 768;
    size_t imgBytes = size_t(W) * H * sizeof(uchar4);

    // --- Mandelbrot-Parameter (im echten Modus) ---
    float zoom = 300.0f;
    float2 offset = make_float2(0.0f, 0.0f);
    int maxIter = 500;

    // --- GLFW + GL Context ---
    if (!glfwInit()) {
        std::cerr << "GLFW-Init fehlgeschlagen!\n";
        std::exit(EXIT_FAILURE);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(W, H, "OtterDream Mandelbrot", nullptr, nullptr);
    if (!win) {
        std::cerr << "Fenster-Erstellung fehlgeschlagen!\n";
        std::exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(win);

    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW-Init fehlgeschlagen\n";
        std::exit(EXIT_FAILURE);
    }
    std::cout << "GLFW + GLEW init OK\n";

    // --- PBO + CUDA-GL Interop ---
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, imgBytes, nullptr, GL_DYNAMIC_DRAW);
    GL_CHECK();

    cudaGraphicsResource* cudaPbo;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &cudaPbo, pbo, cudaGraphicsMapFlagsWriteDiscard
    ));

    // --- Texture Setup ---
    GLuint tex;
    glGenTextures(1, &tex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    GL_CHECK();

    // --- Shader + Quad Setup ---
    GLuint program = createProgram();

    // Sampler-Uniform "uTex" auf Texture Unit 0 setzen
    glUseProgram(program);
    GLint loc = glGetUniformLocation(program, "uTex");
    if (loc >= 0) {
        glUniform1i(loc, 0);
    } else {
        std::cerr << "Warnung: Uniform 'uTex' nicht gefunden!\n";
    }

    GLuint VAO, VBO, EBO;
    float quad[] = {
        // Pos    // Tex
        -1.0f,-1.0f,   0.0f,0.0f,
         1.0f,-1.0f,   1.0f,0.0f,
         1.0f, 1.0f,   1.0f,1.0f,
        -1.0f, 1.0f,   0.0f,1.0f
    };
    unsigned idx[] = {0,1,2, 2,3,0};

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
      glBindBuffer(GL_ARRAY_BUFFER, VBO);
      glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

      // Position (location=0)
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
      glEnableVertexAttribArray(0);
      // TexCoord (location=1)
      glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                            (void*)(2 * sizeof(float)));
      glEnableVertexAttribArray(1);
    glBindVertexArray(0);
    GL_CHECK();

    // --- Complexity Buffer (wird im echten Modus benutzt) ---
    int tilesX = (W + TILE_W - 1) / TILE_W;
    int tilesY = (H + TILE_H - 1) / TILE_H;
    int totalTiles = tilesX * tilesY;

    float* d_complexity = nullptr;
    if (!DEBUG_GRADIENT) {
        CUDA_CHECK(cudaMalloc(&d_complexity,
                              totalTiles * sizeof(float)));
    }
    std::vector<float> h_complexity(totalTiles);

    std::cout << "Setup abgeschlossen, betrete Haupt-Loop\n";

    int frame = 0;
    while (!glfwWindowShouldClose(win)) {
        // 1) PBO → CUDA-Device (mappen)
        uchar4* d_img = nullptr;
        size_t sz_ptr = 0;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPbo, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
                       (void**)&d_img, &sz_ptr, cudaPbo));

#if DEBUG_GRADIENT
        // *** Debug-Modus: einfacher Gradient ***
        dim3 bd_g(16, 16);
        dim3 gd_g((W + bd_g.x - 1) / bd_g.x,
                  (H + bd_g.y - 1) / bd_g.y);
        testKernel<<<gd_g, bd_g>>>(d_img, W, H);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
#else
        // *** Echte Mandelbrot-Ausgabe ***
        launch_mandelbrotHybrid(d_img, W, H, zoom, offset, maxIter);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Complexity-Kernel
        CUDA_CHECK(cudaMemset(d_complexity, 0,
                              totalTiles * sizeof(float)));
        dim3 bd(TILE_W, TILE_H);
        dim3 gd(tilesX, tilesY);
        computeComplexity<<<gd, bd>>>(d_img, W, H, d_complexity);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
#endif

        // 2) PBO unmappen
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPbo, 0));

        // 3) Upload PBO → Texture
        glActiveTexture(GL_TEXTURE0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H,
                        GL_RGBA, GL_UNSIGNED_BYTE, 0);
        GL_CHECK();

        // 4) Rendern mit Vollbild-Quad + Shader
        glViewport(0, 0, W, H);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        GL_CHECK();

        glfwSwapBuffers(win);
        glfwPollEvents();

        // 5) Debug-Ausgabe (NUR im echten Modus sinnvoll)
#if DEBUG_GRADIENT
        if (frame == 0) {
            std::cout << "(DEBUG_GRADIENT-Modus: Farbverlauf angezeigt)\n";
        }
#else
        // read back complexity & finde beste Kachel
        CUDA_CHECK(cudaMemcpy(h_complexity.data(), d_complexity,
                              totalTiles * sizeof(float),
                              cudaMemcpyDeviceToHost));

        int bestIdx = 0;
        float bestScore = -1.0f;
        for (int i = 0; i < totalTiles; ++i) {
            if (h_complexity[i] > bestScore) {
                bestScore = h_complexity[i];
                bestIdx = i;
            }
        }

        std::cout << "Frame " << frame
                  << ": zoom=" << zoom
                  << " offset=(" << offset.x << "," << offset.y << ")"
                  << " bestScore=" << bestScore
                  << std::endl;
#endif

        frame++;

#if !DEBUG_GRADIENT
        // 6) Zoom + Offset NACH dem Rendern (nur im echten Modus)
        if (bestScore > 0.0f) {
            int bx = bestIdx % tilesX;
            int by = bestIdx / tilesX;
            offset.x += ((bx + 0.5f)*TILE_W - W*0.5f)/zoom;
            offset.y += ((by + 0.5f)*TILE_H - H*0.5f)/zoom;
        }
        zoom *= 1.01f;
#endif
    }

    // --- Cleanup ---
    if (!DEBUG_GRADIENT) {
        CUDA_CHECK(cudaFree(d_complexity));
    }
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaPbo));
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    glDeleteProgram(program);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteVertexArrays(1, &VAO);
    glfwDestroyWindow(win);
    glfwTerminate();

    return 0;
}
