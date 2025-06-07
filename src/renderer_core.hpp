#ifndef RENDERER_CORE_HPP
#define RENDERER_CORE_HPP

#include <GLFW/glfw3.h>
#include <vector>
#include <cuda_gl_interop.h>

// Minimal forward declaration
typedef struct cudaGraphicsResource* cudaGraphicsResource_t;

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();                               // 🐭 Automatisches Cleanup

    void initGL();                             // Initialisiert OpenGL & CUDA
    void renderFrame();                        // Rendert einen Frame
    bool shouldClose() const;                  // Prüft, ob Fenster geschlossen werden soll
    void resize(int newWidth, int newHeight);  // Behandelt Fenstergrößenänderung

private:
    void initGL_impl();                        // OpenGL Context Setup intern
    void renderFrame_impl();                   // Frame Render intern
    void setupPBOAndTexture();                 // 🆕 PBO + Texture initialisieren
    void setupBuffers();                       // 🆕 CUDA-Buffer initialisieren

    int windowWidth, windowHeight;
    GLFWwindow* window;
    GLuint pbo, tex, program, VAO, VBO, EBO;
    cudaGraphicsResource_t cudaPboRes;
    float* d_complexity;
    int* d_iterations;
    std::vector<float> h_complexity;
    float zoom;
    float2 offset;
    double lastTime;
    int frameCount;
    float currentFPS, lastFrameTime;
};

#endif // RENDERER_CORE_HPP
