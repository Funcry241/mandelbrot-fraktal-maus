// Datei: src/zoom_command.cpp
// 🐭 Maus: Klar getrenntes Host-Logging, deterministische Ausgabe.
// 🦦 Otter: keine stdout-Magie, alles durch LUCHS_LOG_HOST. (Bezug zu Otter)
// 🐑 Schneefuchs: robuste Fehlerpfade, ASCII-only.

#include "zoom_command.hpp"
#include "luchs_log_host.hpp"
#include <fstream>

void exportCommandsToCSV(const CommandBus& bus, const std::string& filename) {
    std::ofstream out(filename, std::ios::out | std::ios::trunc);
    if (!out) {
        LUCHS_LOG_HOST("[ZoomExport] Failed to open file: %s", filename.c_str());
        return;
    }

    out << ZoomCommand::csvHeader() << '\n';
    for (const auto& cmd : bus.getHistory()) {
        out << cmd.toCSV() << '\n';
    }
    out.flush();
    if (!out) {
        LUCHS_LOG_HOST("[ZoomExport] Write error while exporting to '%s'", filename.c_str());
        return;
    }

    LUCHS_LOG_HOST("[ZoomExport] Exported %zu commands to '%s'",
                   bus.getHistory().size(), filename.c_str());
}

void printZoomHistory(const CommandBus& bus, int maxLines) {
    const auto& hist = bus.getHistory();
    const int total  = static_cast<int>(hist.size());
    const int limit  = (maxLines > 0) ? maxLines : 10;
    const int begin  = (total > limit) ? (total - limit) : 0;

    LUCHS_LOG_HOST("[ZoomLog] Showing last %d of %d commands:", total - begin, total);
    LUCHS_LOG_HOST("%s", ZoomCommand::csvHeader().c_str());

    for (int i = begin; i < total; ++i) {
        LUCHS_LOG_HOST("%s", hist[static_cast<std::size_t>(i)].toCSV().c_str());
    }
}
