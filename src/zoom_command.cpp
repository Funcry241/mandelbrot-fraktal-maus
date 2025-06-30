// Datei: src/zoom_command.cpp
// Zeilen: 55
/* 🐭 interner Maus-Kommentar:
   Diese Datei implementiert CSV-Export und Logging für ZoomCommand.
   Noch kein Replay – Fokus auf: Export, Debug-Dump, optionales File-Logging.
   → Später auch: ReplayCommandBus, Load/Save, deterministischer Replay-Modus.
*/

#include "zoom_command.hpp"
#include <fstream>
#include <iostream>

void exportCommandsToCSV(const CommandBus& bus, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[ZoomExport] Failed to open file: " << filename << "\n";
        return;
    }

    out << ZoomCommand::csvHeader() << "\n";
    for (const auto& cmd : bus.getHistory()) {
        out << cmd.toCSV() << "\n";
    }

    out.close();
    std::cout << "[ZoomExport] Exported " << bus.getHistory().size()
              << " commands to '" << filename << "'\n";
}

void printZoomHistory(const CommandBus& bus, int maxLines = 10) {
    const auto& hist = bus.getHistory();
    int total = static_cast<int>(hist.size());
    int begin = std::max(0, total - maxLines);
    std::cout << "[ZoomLog] Showing last " << (total - begin) << " of "
              << total << " commands:\n";
    std::cout << ZoomCommand::csvHeader() << "\n";
    for (int i = begin; i < total; ++i) {
        std::cout << hist[i].toCSV() << "\n";
    }
}
