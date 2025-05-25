// src/settings.cpp

#include "settings.hpp"
#include <fstream>
#include <iostream>
#include <cstring>

static bool debug_mode = false;
static std::ofstream debug_log;

// Liest das „-d“-Flag aus den Kommandozeilenparametern
void init_cli(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-d") == 0) {
            debug_mode = true;
        }
    }
}

// Öffnet die Logdatei, wenn Debug-Modus aktiv
void init_logging() {
    if (debug_mode) {
        debug_log.open("mandelbrot_otterdream_log.txt", std::ios::out);
        if (!debug_log) {
            std::cerr << "Konnte Logdatei nicht öffnen\n";
        } else {
            debug_log << "[DEBUG] Logging gestartet\n";
        }
    }
}

// Schliesst die Logdatei, wenn sie geöffnet wurde
void cleanup_logging() {
    if (debug_mode && debug_log.is_open()) {
        debug_log << "[DEBUG] Logging beendet\n";
        debug_log.close();
    }
}
