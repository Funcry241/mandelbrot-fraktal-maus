#include <utility>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <iostream>

#include "settings.hpp"
#include "input/callbacks.hpp"
#include "rendering/renderer.hpp"
#include "utils/cuda_utils.hpp"
#include "gui.hpp"
#include "metrics.hpp"
#include "compute/boundary.hpp"

Settings S;
Metrics   M;

int main(int argc, char** argv) {
    init_cli(argc, argv);
    init_logging();

    if (!glfwInit()) { std::cerr<<"GLFW failed\n"; return -1; }
    GLFWwindow* window = init_window();
    if (!window) { glfwTerminate(); return -1; }
    if (glewInit() != GLEW_OK) { std::cerr<<"GLEW failed\n"; return -1; }

    init_gui(window);
    init_callbacks(window);

    Renderer renderer(window);
    BoundaryService boundary;
    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        double dt  = now - lastTime;
        lastTime   = now;

        // Auto-Zoom
        double zoomFac = std::pow(static_cast<double>(S.autoZoomPerSec), dt);
        S.zoom *= BigFloat(zoomFac);

        boundary.update(S);

        // Deadzone + Panning
        auto bfTarget = boundary.getNextTarget();
        BigFloat tx = bfTarget.first, ty = bfTarget.second;
        BigFloat dxFull = tx - S.offsetX, dyFull = ty - S.offsetY;
        BigFloat eps = BigFloat("1e-3") / S.zoom;
        if (abs(dxFull) >= eps || abs(dyFull) >= eps) {
            BigFloat maxShift = BigFloat("0.1") / S.zoom;
            BigFloat dx = dxFull, dy = dyFull;
            BigFloat d2 = dx*dx + dy*dy;
            if (d2 > maxShift*maxShift) {
                double d = std::sqrt(d2.convert_to<double>());
                dx = dx / BigFloat(d) * maxShift;
                dy = dy / BigFloat(d) * maxShift;
            }
            BigFloat blend = BigFloat(std::min(1.0, double(S.panSpeed) * dt));
            S.offsetX += dx * blend;
            S.offsetY += dy * blend;
        }

        renderer.pollEvents();
        renderer.renderFrame();
    }

    shutdown_gui();
    cleanup_logging();
    return 0;
}
