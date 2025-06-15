// Datei: src/main.cpp
// 🐭 Maus-Kommentar: Hauptprogramm – Einstiegspunkt, nutzt jetzt pch.hpp zur Konfliktvermeidung und sauberen Init

#include "pch.hpp"

#include "renderer_core.hpp"
#include "settings.hpp"

int main() {
    Renderer renderer(Settings::width, Settings::height);
    renderer.initGL();

    while (!renderer.shouldClose()) {
        renderer.renderFrame(true); // Auto-Zoom aktiviert
    }

    return 0;
}
