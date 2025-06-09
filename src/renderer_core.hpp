#ifndef RENDERER_CORE_HPP
#define RENDERER_CORE_HPP

#include <vector>
#include <cuda_gl_interop.h>   // cudaGraphicsResource_t
#ifndef __CUDACC__
#include <GL/glew.h>           // GLuint (nur für CPU-Seite relevant)
#endif

struct GLFWwindow;             // 🐭 Forward Declaration spart Include-Zeit

class Renderer {
public:
    __host__ Renderer(int width, int height);                 // 🐭 Konstruktor für CPU
    __host__ ~Renderer();                                      // 🐭 Destruktor für CPU

    __host__ void initGL();                                    // OpenGL & CUDA initialisieren
    __host__ void renderFrame(bool autoZoomEnabled);           // Frame Rendern (Auto-Zoom optional)
    __host__ bool shouldClose() const;                         // Prüfen, ob Fenster geschlossen werden soll
    __host__ void resize(int newWidth, int newHeight);         // Fenstergrößenänderung behandeln
    __host__ GLFWwindow* getWindow() const;                    // Zugriff auf GLFW Fenster (z.B. für Callbacks)

private:
    __host__ void initGL_impl();                               // OpenGL Context Setup intern
    __host__ void renderFrame_impl(bool autoZoomEnabled);      // 🐭 Internes Frame Rendering
    __host__ void setupPBOAndTexture();                        // PBO + Textur Setup
    __host__ void setupBuffers();                              // CUDA-Buffer Setup

    int windowWidth;
    int windowHeight;
    GLFWwindow* window;
    GLuint pbo;
    GLuint tex;
    GLuint program;
    GLuint VAO;
    GLuint VBO;
    GLuint EBO;
    cudaGraphicsResource_t cudaPboRes;
    float* d_complexity;
    int* d_iterations;
    std::vector<float> h_complexity;
    float zoom;
    float2 offset;
    double lastTime;
    int frameCount;
    float currentFPS;
    float lastFrameTime;
};

#endif // RENDERER_CORE_HPP
