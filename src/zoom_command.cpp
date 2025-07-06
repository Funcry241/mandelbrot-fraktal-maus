// Datei: src/zoom_command.cpp
// Zeilen: 47
// 🐭 Maus-Kommentar: Implementiert CSV-Export und Zoom-Log für CommandBus.
// Noch kein Replay-Modus – Fokus auf Export & Debug-Ausgabe. Otter geprüft, Schneefuchs-Style.

#include "zoom_command.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

void exportCommandsToCSV(const CommandBus& bus, const std::string& filename) {
std::ofstream out(filename);
if (!out) {
std::cerr << "[ZoomExport] Failed to open file: " << filename << "\n";
return;
}
out << ZoomCommand::csvHeader() << "\n";
for (const auto& cmd : bus.getHistory())
out << cmd.toCSV() << "\n";
std::cout << "[ZoomExport] Exported " << bus.getHistory().size()
<< " commands to '" << filename << "'\n";
}

void printZoomHistory(const CommandBus& bus, int maxLines) {
const auto& hist = bus.getHistory();
int total = static_cast<int>(hist.size());
int begin = std::max(0, total - (maxLines > 0 ? maxLines : 10));
std::cout << "[ZoomLog] Showing last " << (total - begin) << " of " << total << " commands:\n"
<< ZoomCommand::csvHeader() << "\n";
for (int i = begin; i < total; ++i)
std::cout << hist[i].toCSV() << "\n";
}
