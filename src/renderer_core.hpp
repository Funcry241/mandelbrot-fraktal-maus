// Datei: src/renderer_core.hpp
#ifndef RENDERER_CORE_HPP
#define RENDERER_CORE_HPP

#include <vector>
#include <cuda_gl_interop.h>   // für CUDA/OpenGL Interop
#include <GL/glew.h>           // für GLuint

struct GLFWwindow;             // 🐭 Forward Declaration spart Include-Zeit

// 🐭 Renderer-Klasse: Steuert Fenster, OpenGL-Setup, CUDA-Pipeline & Auto-Zoom
class Renderer {
public:
    __host__ Renderer(int width, int height);                // 🏗️ Konstruktor
    __host__ ~Renderer();                                    // 🧹 Destruktor

    __host__ void initGL();                                  // 🌐 OpenGL-Init (VAO, PBO, Texture)
    __host__ void renderFrame(bool autoZoomEnabled);         // 🎥 Frame Rendern mit optionalem Auto-Zoom
    __host__ bool shouldClose() const;                       // 🚪 Fenster schließen?
    __host__ void resize(int newWidth, int newHeight);       // ↔️ Resize behandeln
    __host__ GLFWwindow* getWindow() const;                  // 🪟 Zugriff auf GLFW-Handle

private:
    __host__ void initGL_impl();                             // 🔧 OpenGL Context Setup
    __host__ void renderFrame_impl(bool autoZoomEnabled);    // 🌀 Rendering Loop
    __host__ void setupPBOAndTexture();                      // 📦 PBO + Texture konfigurieren
    __host__ void setupBuffers();                            // 📊 CUDA-Buffer anlegen
    __host__ void freeDeviceBuffers();                       // 🧽 CUDA-Buffer freigeben (d_stddev, d_iterations, ...)

    int windowWidth;     // 📐 Fensterbreite
    int windowHeight;    // 📐 Fensterhöhe
    GLFWwindow* window;  // 🪟 GLFW-Fenster

    // OpenGL Objekte
    GLuint pbo;          // 🎞️ Pixel Buffer Object
    GLuint tex;          // 🖼️ Texture
    GLuint program;      // 🧠 Shader-Programm
    GLuint VAO, VBO, EBO; // 🔩 Geometrieobjekte

    // CUDA-Buffer
    float* d_complexity = nullptr;       // σ Komplexität je Tile (Device)
    float* d_stddev = nullptr;           // σ Standardabweichung je Tile (Device)
    int* d_iterations = nullptr;         // 🔁 Iterationen je Pixel (Device)

    // Host-Buffer
    std::vector<float> h_complexity;     // σ Kopie für Auswertung (Host)

    // Frame-Metadaten
    float zoom;           // 🔍 Zoom-Level
    float2 offset;        // 🎯 Mittelpunkt im Fraktalraum
    double lastTime;      // ⏲️ Zeittracking
    int frameCount;
    float currentFPS;
    float lastFrameTime;

    int lastTileSize = -1; // 🧠 Letzter verwendeter Tile-Size (zur Auto-Zoom-Erkennung)
};

#endif // RENDERER_CORE_HPP
