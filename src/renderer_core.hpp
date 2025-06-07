#pragma once

#include <GLFW/glfw3.h>
#include <vector>
#include <cuda_gl_interop.h>

using cudaGraphicsResource_t = struct cudaGraphicsResource*;

class Renderer {
public:
    Renderer(int w, int h);
    ~Renderer();
    void initGL(), renderFrame();
    bool shouldClose() const;
    void resize(int newW, int newH);

private:
    void initGL_impl(), renderFrame_impl(), setupPBOAndTexture(), setupBuffers();
    int windowWidth, windowHeight;
    GLFWwindow* window;
    GLuint pbo, tex, program, VAO, VBO, EBO;
    cudaGraphicsResource_t cudaPboRes;
    float* d_complexity;               // ✅ Device Buffer
    int* d_iterations;                 // ✅ Device Buffer
    std::vector<float> h_complexity;    // ✅ Host Buffer (std::vector!)
    float zoom;
    float2 offset;
    double lastTime;
    int frameCount;
    float currentFPS, lastFrameTime;
};
