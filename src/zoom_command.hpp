// Datei: src/zoom_command.hpp
// Zeilen: 65
// 🐭 Maus-Kommentar: Struktur für jede Auto-Zoom-Entscheidung – deterministisch, replayfähig, testbar.
// math_utils entfernt – nur noch float2 aus <vector_types.h>. CommandBus übernimmt Replay/History.

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
int tileIndex = -1;

std::string toCSV() const {
    char buf[128];
    snprintf(buf, sizeof(buf),
             "%d,%.5f,%.5f,%.1e,%.1e,%.4f,%.4f,%d",
             frameIndex, newOffset.x, newOffset.y,
             zoomBefore, zoomAfter, entropy, contrast, tileIndex);
    return std::string(buf);
}
static std::string csvHeader() {
    return "Frame,X,Y,ZoomBefore,ZoomAfter,Entropy,Contrast,TileIndex";
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
