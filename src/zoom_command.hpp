// Datei: src/zoom_command.hpp
// Zeilen: 63
// 🐭 Maus-Kommentar: Struktur für jede Auto-Zoom-Entscheidung – deterministisch, replayfähig, testbar.
// 🦦 Otter: Reproduzierbares Verhalten durch CommandBus, jeder Frame dokumentiert.
// 🐅 Maus: Kompakt, ohne math_utils, nur float2 aus <vector_types.h>.
// 🐼 Panda: Jeder ZoomCommand ist ein protokollierter Denkprozess – Grundlage für Analyse, Replay und Heatmap.
// Ziel: vollständige Nachvollziehbarkeit aller Zoom-Aktionen, ideal für Analyse und Debug.

#pragma once
#include <vector>
#include <string>
#include <cstdio>
#include <vector_types.h> // Für float2

struct ZoomCommand {
    int frameIndex = 0;
    float2 oldOffset = {0, 0};
    float2 newOffset = {0, 0};
    float zoomBefore = 1.0f;
    float zoomAfter = 1.0f;
    float entropy = 0.0f;
    float contrast = 0.0f;

    std::string toCSV() const {
        char buf[128];
        snprintf(buf, sizeof(buf),
                 "%d,%.5f,%.5f,%.1e,%.1e,%.4f,%.4f",
                 frameIndex, newOffset.x, newOffset.y,
                 zoomBefore, zoomAfter, entropy, contrast);
        return std::string(buf);
    }

    static std::string csvHeader() {
        return "Frame,X,Y,ZoomBefore,ZoomAfter,Entropy,Contrast";
    }
};

class CommandBus {
public:
    void push(const ZoomCommand& cmd) { commands.push_back(cmd); }
    const std::vector<ZoomCommand>& getHistory() const { return commands; }
    void clear() { commands.clear(); }

private:
    std::vector<ZoomCommand> commands;
};
