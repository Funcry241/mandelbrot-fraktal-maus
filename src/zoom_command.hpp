// Datei: src/zoom_command.hpp
// Zeilen: 65
/* 🐭 interner Maus-Kommentar:
   Diese Datei enthält die Struktur `ZoomCommand`, die jede Auto-Zoom-Entscheidung
   eindeutig beschreibt. Sie wird pro Frame erzeugt, gespeichert und ggf. reproduzierbar
   wiederverwendet. Der `CommandBus` speichert sie für Replay, Logging oder Tests.
   → Grundlage für deterministischen Zoomfluss, Analyse, Undo etc.
   → FIX: math_utils.hpp entfernt – stattdessen direkter CUDA-Typ-Import (float2 aus <vector_types.h>)
*/

#pragma once
#include <vector>
#include <optional>
#include <string>
#include <cstdio>
#include <vector_types.h> // ✅ notwendig für float2

struct ZoomCommand {
    int frameIndex = 0;           // globaler Frame-Zähler
    float2 oldOffset = {0, 0};    // vorheriger Mittelpunkt
    float2 newOffset = {0, 0};    // neues Ziel
    float zoomBefore = 1.0f;      // vorheriger Zoom
    float zoomAfter = 1.0f;       // neuer Zoom
    float entropy = 0.0f;         // Entropie an neuem Ziel
    float contrast = 0.0f;        // Kontrast an neuem Ziel
    int tileIndex = -1;           // Index der Zielkachel

    std::string toCSV() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer),
            "%d,%.5f,%.5f,%.1e,%.1e,%.4f,%.4f,%d",
            frameIndex, newOffset.x, newOffset.y,
            zoomBefore, zoomAfter,
            entropy, contrast, tileIndex);
        return std::string(buffer);
    }

    static std::string csvHeader() {
        return "Frame,X,Y,ZoomBefore,ZoomAfter,Entropy,Contrast,TileIndex";
    }
};

class CommandBus {
public:
    void push(const ZoomCommand& cmd) {
        commands.push_back(cmd);
    }

    const std::vector<ZoomCommand>& getHistory() const {
        return commands;
    }

    void clear() {
        commands.clear();
    }

private:
    std::vector<ZoomCommand> commands;
};
