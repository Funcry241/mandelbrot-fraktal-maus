// Datei: src/renderer_core.hpp

#ifndef RENDERER_CORE_HPP
#define RENDERER_CORE_HPP

#include <GLFW/glfw3.h>

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    void initGL();
    void renderFrame();
    void cleanup();
    bool shouldClose() const;   // <-- nur Deklaration, keine Definition hier

private:
    void initGL_impl(GLFWwindow* window);
    void renderFrame_impl(GLFWwindow* window);
    void cleanup_impl();

    int windowWidth;
    int windowHeight;
    GLFWwindow* window;
};

#endif // RENDERER_CORE_HPP
