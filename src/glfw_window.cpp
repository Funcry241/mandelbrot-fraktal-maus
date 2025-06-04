#include "glfw_window.hpp"
#include <iostream>

GLFWwindow* createGLFWWindow(int width, int height, const char* title) {
    if (!glfwInit()) {
        std::cerr << "GLFW-Init fehlgeschlagen\n";
        return nullptr;
    }
    // Core-Profile 4.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        std::cerr << "Fenster-Erstellung fehlgeschlagen\n";
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // V-Sync einschalten
    return window;
}

bool windowShouldClose(GLFWwindow* window) {
    return (window && glfwWindowShouldClose(window));
}

void swapAndPoll(GLFWwindow* window) {
    glfwSwapBuffers(window);
    glfwPollEvents();
}
