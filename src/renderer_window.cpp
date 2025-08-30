///// Otter: Moderner GL-Kontext (4.3 Core), deterministischer Setup-Flow; frühes Fehler-Logging.
///// Schneefuchs: ASCII-Logs; Debug-Kontext nur bei Debug/Perf; compile-time Fensterposition; keine C4127.
///// Maus: State clean; zentrieren/Position deterministisch; Header/Source synchron.

#include "pch.hpp"
#include "renderer_window.hpp"
#include "renderer_core.hpp"
#include "settings.hpp"
#include "renderer_loop.hpp"   // RendererLoop::keyCallback
#include "luchs_log_host.hpp"

namespace RendererInternals {

// GLFW-Error-Callback (ASCII only)
static void glfwErrorCallback(int code, const char* description) {
    LUCHS_LOG_HOST("[GLFW-ERROR] code=%d desc=%s", code, description ? description : "(null)");
}

// Fenster zentrieren ODER feste Position setzen.
// Hinweis: Die Entscheidung, ob feste Position genutzt wird, wird hier bewusst
// als compile-time Branch geschrieben, um C4127 zu vermeiden.
static void centerWindowIfRequested(GLFWwindow* window, int w, int h) {
    if (!window) return;

    // compile-time Flag für feste Position
    constexpr bool kHasFixedPos =
        (Settings::windowPosX >= 0) && (Settings::windowPosY >= 0);

    if constexpr (kHasFixedPos) {
        glfwSetWindowPos(window, Settings::windowPosX, Settings::windowPosY);
        return;
    } else {
        GLFWmonitor* mon = glfwGetPrimaryMonitor();
        const GLFWvidmode* vm = mon ? glfwGetVideoMode(mon) : nullptr;
        if (vm) {
            const int x = (vm->width  - w) / 2;
            const int y = (vm->height - h) / 2;
            glfwSetWindowPos(window, x, y);
        }
    }
}

// Erstellt Fenster, setzt GL-Kontext & VSync
GLFWwindow* createGLFWWindow(int width, int height) {
    // Error-Callback VOR glfwInit()
    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit()) {
        LUCHS_LOG_HOST("[ERROR] GLFW init failed");
        return nullptr;
    }

    glfwDefaultWindowHints();

    // OpenGL 4.3 Core (Forward-Compatible für macOS/striktes Core)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

    // Debug-Kontext nur, wenn wir tatsächlich debuggen/profilen (compile-time)
    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
    } else {
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_FALSE);
    }

#ifdef GLFW_SRGB_CAPABLE
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(width, height, "OtterDream Mandelbrot", nullptr, nullptr);
    if (!window) {
        LUCHS_LOG_HOST("[ERROR] Window creation failed");
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);

    // VSync: im Perf-Logging-Modus aus, sonst an (kein C4127, da ?:)
    glfwSwapInterval(Settings::performanceLogging ? 0 : 1);

    centerWindowIfRequested(window, width, height);
    return window;
}

// Callbacks registrieren (Größe, Tasten)
void configureWindowCallbacks(GLFWwindow* window, void* userPointer) {
    glfwSetWindowUserPointer(window, userPointer);

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* win, int newW, int newH) {
        if (auto* self = static_cast<Renderer*>(glfwGetWindowUserPointer(win))) {
            self->resize(newW, newH);
        }
    });

    glfwSetKeyCallback(window, RendererLoop::keyCallback);
}

} // namespace RendererInternals

namespace RendererWindow {

GLFWwindow* createWindow(int width, int height, Renderer* instance) {
    GLFWwindow* window = RendererInternals::createGLFWWindow(width, height);
    if (!window) return nullptr;
    RendererInternals::configureWindowCallbacks(window, instance);
    return window;
}

bool shouldClose(GLFWwindow* window) {
    return glfwWindowShouldClose(window);
}

void destroyWindow(GLFWwindow* window) {
    if (window) {
        glfwDestroyWindow(window);
    }
    // glfwTerminate() erfolgt zentral im Renderer/Programmende.
}

} // namespace RendererWindow
