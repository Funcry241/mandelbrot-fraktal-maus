// Datei: src/zoom_command.hpp
// 🐭 Maus: deterministische, replayfähige Kommandos; CSV stabil.
// 🦦 Otter: keine impliziten Makros; Logging via Host-Layer. (Bezug zu Otter)
// 🦊 Schneefuchs: trivially-copyable, keine heimlichen ABI-Fallen. (Bezug zu Schneefuchs)

#pragma once
#include <vector>
#include <string>
#include <cstdio>
#include <type_traits>
#include <vector_types.h> // float2

#ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning(disable: 4324) // MSVC padding warning wegen float2
#endif

class ZoomCommand {
public:
    int    frameIndex = 0;
    float2 oldOffset  = {0.0f, 0.0f};
    float2 newOffset  = {0.0f, 0.0f};
    float  zoomBefore = 1.0f;
    float  zoomAfter  = 1.0f;
    float  entropy    = 0.0f;
    float  contrast   = 0.0f;

    // CSV: Frame,X,Y,ZoomBefore,ZoomAfter,Entropy,Contrast
    [[nodiscard]] std::string toCSV() const {
        // Feste Präzision für deterministisches Diffen/Replays.
        char buf[192];
        std::snprintf(buf, sizeof(buf),
                      "%d,%.5f,%.5f,%.6e,%.6e,%.4f,%.4f",
                      frameIndex,
                      newOffset.x, newOffset.y,
                      static_cast<double>(zoomBefore),
                      static_cast<double>(zoomAfter),
                      entropy, contrast);
        return std::string(buf);
    }

    [[nodiscard]] static std::string csvHeader() {
        // Konstant, aber als std::string zurückgegeben für bequeme Nutzung.
        return "Frame,X,Y,ZoomBefore,ZoomAfter,Entropy,Contrast";
    }
};

static_assert(std::is_trivially_copyable<ZoomCommand>::value,
              "ZoomCommand must remain trivially copyable");

class CommandBus {
public:
    void push(const ZoomCommand& cmd)            { commands.push_back(cmd); }
    [[nodiscard]] const std::vector<ZoomCommand>& getHistory() const { return commands; }
    void clear()                                 { commands.clear(); }
    [[nodiscard]] std::size_t size() const       { return commands.size(); }
    void reserve(std::size_t n)                  { commands.reserve(n); }
    [[nodiscard]] std::size_t capacity() const   { return commands.capacity(); }

private:
    std::vector<ZoomCommand> commands;
};

// ─────────────────────────────────────────────────────────────────────────────
// Freie Helfer (Implementierung in zoom_command.cpp)
// ─────────────────────────────────────────────────────────────────────────────

// CSV-Datei schreiben (Header + alle Zeilen).
void exportCommandsToCSV(const CommandBus& bus, const std::string& filename);

// Letzte N Einträge als ASCII-CSV über LUCHS_LOG_HOST ausgeben.
// maxLines <= 0 ⇒ Default 10.
void printZoomHistory(const CommandBus& bus, int maxLines = 10);

#ifdef _MSC_VER
  #pragma warning(pop)
#endif
