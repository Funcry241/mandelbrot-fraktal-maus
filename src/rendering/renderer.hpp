// src/rendering/renderer.hpp

#pragma once

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>
#include "settings.hpp"
#include "metrics.hpp"

/// @brief Renderer für Mandelbrot-Fraktal mit OpenGL/CUDA-Interop.
class Renderer {
public:
    explicit Renderer(GLFWwindow* window);
    ~Renderer();

    /// Poll GLFW-Events.
    void pollEvents();

    /// Zeichnet den aktuellen Frame basierend auf S.offsetX/Y.
    void renderFrame();

    const Settings& getSettings() const;

private:
    void initGL();
    void setupShaders();
    void setupBuffers();

    GLFWwindow*           window;
    GLuint                pbo     = 0;
    GLuint                tex     = 0;
    cudaGraphicsResource* cuda_pbo_resource = nullptr;
    GLuint                quadVAO = 0, quadVBO = 0, quadProg = 0;
};
