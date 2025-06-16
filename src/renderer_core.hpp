// Datei: src/renderer_core.hpp
// Zeilen: 60
// 🐭 Maus-Kommentar: Öffentliche Steuerzentrale für Rendering, Fenster und Auto-Zoom. Die Klasse `Renderer` kapselt die OpenGL-Initialisierung, CUDA-Interop-Setup, PBO/Texture-Bindung und das adaptive Zoomverhalten. Diese Header-Datei ist vollständig unabhängig von Implementierungsdetails und trennt klar die API von der Logik. Schneefuchs hätte hier auf das klare Interface mit `initGL()` und `renderFrame()` bestanden.

#pragma once

#include <vector>
#include <cuda_gl_interop.h>  // Für CUDA/OpenGL Interop

struct GLFWwindow;  // 🪟 Forward Declaration spart Header-Ballast

class Renderer {
public:
    __host__ Renderer(int width, int height);                // 🏗️ Konstruktor
    __host__ ~Renderer();                                    // 🧹 Destruktor

    __host__ void initGL();                                  // 🌐 OpenGL initialisieren
    __host__ void renderFrame(bool autoZoomEnabled);         // 🎥 Bild rendern (mit/ohne Auto-Zoom)
    __host__ bool shouldClose() const;                       // 🚪 Fensterstatus
    __host__ void resize(int newWidth, int newHeight);       // ↔️ Resize
    __host__ GLFWwindow* getWindow() const;                  // 🪟 Zugriff auf GLFW-Fenster

private:
    void initGL_impl();                                      // 🔧 GL Setup
    void renderFrame_impl(bool autoZoomEnabled);             // 🌀 Frame zeichnen
    void setupPBOAndTexture();                               // 📦 GL PBO + Texture konfigurieren
    void setupBuffers();                                     // 📊 CUDA-Buffer anlegen
    void freeDeviceBuffers();                                // 🧽 Buffer freigeben

    int windowWidth;
    int windowHeight;
    GLFWwindow* window = nullptr;

    // OpenGL-Objekte
    GLuint pbo = 0;
    GLuint tex = 0;
    GLuint program = 0;
    GLuint VAO = 0, VBO = 0, EBO = 0;

    // CUDA Device Buffer
    float* d_entropy = nullptr;      // 🧠 Entropie pro Tile (statt „Komplexität“)
    int*   d_iterations = nullptr;   // 🔁 Iterationswerte pro Pixel

    // Host-Side Auswertung
    std::vector<float> h_entropy;    // 🖥️ Entropie-Ergebnisse

    // Rendering-Zustand
    float zoom = 1.0f;
    float2 offset = {0.0f, 0.0f};
    double lastTime = 0.0;
    int frameCount = 0;
    float currentFPS = 0.0f;
    float lastFrameTime = 0.0f;
    int lastTileSize = -1;
};
