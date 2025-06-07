#ifndef RENDERER_CORE_HPP
#define RENDERER_CORE_HPP

#include <GLFW/glfw3.h>
#include <vector>
#include <cuda_gl_interop.h>

typedef struct cudaGraphicsResource* cudaGraphicsResource_t;

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    void initGL();
    void renderFrame();
    void cleanup();               // 🐭 ✅ HINZUGEFÜGT
    bool shouldClose() const;
    void resize(int newWidth, int newHeight);

private:
    void initGL_impl();
    void renderFrame_impl();
    void cleanup_impl();

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
