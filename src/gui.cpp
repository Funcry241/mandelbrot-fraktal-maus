// src/gui.cpp

#include <GL/glew.h>              // Muss vor glfw/gl.h kommen
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "gui.hpp"
#include "settings.hpp"
#include "metrics.hpp"
#include <sstream>
#include <iomanip>

GLFWwindow* init_window() {
    // Hints für OpenGL 3.3 Core
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(
        Settings().width,
        Settings().height,
        "OtterDream Mandelbrot",
        nullptr, nullptr
    );
    if (window) {
        glfwMakeContextCurrent(window);
        // VSync ausschalten, um echte FPS zu messen
        glfwSwapInterval(0);
    }
    return window;
}

void init_gui(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 1.2f;  // etwas größere HUD-Schrift

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding   = 5.0f;
    style.WindowBorderSize = 1.0f;
    style.Colors[ImGuiCol_WindowBg].w = 0.7f;  // Transparenz

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

void render_gui(const Settings& S, const Metrics& M) {
    // neuen Frame beginnen
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Flags für ein einfaches, automatisches HUD-Fenster
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration
                           | ImGuiWindowFlags_AlwaysAutoResize
                           | ImGuiWindowFlags_NoSavedSettings;
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);

    ImGui::Begin("HUD", nullptr, flags);

    // 1) FPS direkt aus ImGuiIO
    ImGuiIO& io = ImGui::GetIO();
    ImGui::Text("FPS: %.1f", io.Framerate);

    // 2) Zoom (BigFloat → double) mit 2 Nachkommastellen
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2)
           << "Zoom: " << S.zoom.convert_to<double>();
        ImGui::Text("%s", ss.str().c_str());
    }

    // 3) Normierte Iterationen & Prozent bei maxIter
    ImGui::Text("Durchschn. Iter: %.2f", M.avgNormIter * S.maxIter);
    ImGui::Text("%% @ maxIter: %.1f%%", M.pctAtMaxIter * 100.0f);

    ImGui::End();

    // Rendern
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void shutdown_gui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}
