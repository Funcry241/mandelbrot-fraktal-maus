///// Otter: Implements GLFW init order; swapInterval only after context is current.
///// Schneefuchs: Explicit error logs and exceptions; avoids undefined calls prior to glfwInit.
///// Maus: No printf/fprintf; all logs via LUCHS_LOG_HOST; ASCII only.
///// Datei: src/glfw_bootstrap.cpp

#include "pch.hpp"
#include "glfw_bootstrap.hpp"
#include <GLFW/glfw3.h>

namespace otterdream {

GlfwApp::GlfwApp() = default;

GlfwApp::~GlfwApp() {
    if (window_) {
        LUCHS_LOG_HOST("[GLFW] destroying window");
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    if (initialized_) {
        LUCHS_LOG_HOST("[GLFW] terminating");
        glfwTerminate();
        initialized_ = false;
    }
}

void GlfwApp::initGlfw_() {
    if (initialized_) return;
    if (!glfwInit()) {
        LUCHS_LOG_HOST("[GLFW] glfwInit failed");
        throw std::runtime_error("glfwInit failed");
    }
    initialized_ = true;
    LUCHS_LOG_HOST("[GLFW] glfwInit ok");
}

void GlfwApp::setHints_() {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef GLFW_CONTEXT_ROBUSTNESS
    glfwWindowHint(GLFW_CONTEXT_ROBUSTNESS, GLFW_LOSE_CONTEXT_ON_RESET);
#endif
#ifdef GLFW_CONTEXT_RELEASE_BEHAVIOR
    glfwWindowHint(GLFW_CONTEXT_RELEASE_BEHAVIOR, GLFW_RELEASE_BEHAVIOR_FLUSH);
#endif
}

GLFWwindow* GlfwApp::createWindow_(int w, int h, const char* title) {
    GLFWwindow* win = glfwCreateWindow(w, h, title, nullptr, nullptr);
    if (!win) {
        LUCHS_LOG_HOST("[GLFW] glfwCreateWindow failed w=%d h=%d title=%s", w, h, title);
        throw std::runtime_error("glfwCreateWindow failed");
    }
    LUCHS_LOG_HOST("[GLFW] window created w=%d h=%d title=%s", w, h, title);
    return win;
}

void GlfwApp::makeCurrentAndSetSwapInterval_(GLFWwindow* win, bool vsync) {
    glfwMakeContextCurrent(win);
    LUCHS_LOG_HOST("[GLFW] context current");
    glfwSwapInterval(vsync ? 1 : 0);
    LUCHS_LOG_HOST("[GLFW] swapInterval=%d", vsync ? 1 : 0);
}

GLFWwindow* GlfwApp::initAndCreate(const GlfwAppConfig& cfg) {
    initGlfw_();
    setHints_();
    window_ = createWindow_(cfg.width, cfg.height, cfg.title.c_str());
    makeCurrentAndSetSwapInterval_(window_, cfg.vsync);
    return window_;
}

} // namespace otterdream
