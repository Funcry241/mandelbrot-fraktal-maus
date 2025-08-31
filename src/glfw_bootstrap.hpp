///// Otter: GLFW bootstrap – correct init order and context binding; swapInterval after make current.
///// Schneefuchs: OpenGL 4.3 core profile hints; no GLEW calls here; ASCII logs only.
///// Maus: Deterministic behavior; no hidden side-effects; logs via LUCHS_LOG_HOST.
///// Datei: src/glfw_bootstrap.hpp

#pragma once
#include <stdexcept>
#include <string>
#include <cstdint>
#include "luchs_log_host.hpp"

struct GLFWwindow; // forward decl

namespace otterdream {

struct GlfwAppConfig {
    int width  = 1280;
    int height = 720;
    bool vsync = true;
    std::string title = "Fruehstuecksbaer";
};

class GlfwApp {
public:
    GlfwApp();
    ~GlfwApp();

    // Initializes GLFW, creates window, makes context current, sets swap interval.
    GLFWwindow* initAndCreate(const GlfwAppConfig& cfg);

    // Access window (may be nullptr before initAndCreate).
    GLFWwindow* window() const noexcept { return window_; }

private:
    GLFWwindow* window_ = nullptr;
    bool initialized_ = false;

    void initGlfw_();
    void setHints_();
    GLFWwindow* createWindow_(int w, int h, const char* title);
    void makeCurrentAndSetSwapInterval_(GLFWwindow* win, bool vsync);
};

} // namespace otterdream
