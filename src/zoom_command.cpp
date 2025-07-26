#include "zoom_logic.hpp"
#include "zoom_command.hpp"
#include "luchs_logger.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

void exportCommandsToCSV(const CommandBus& bus, const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        LUCHS_LOG("[ZoomExport] Failed to open file: %s\n", filename.c_str());
        return;
    }

    out << ZoomCommand::csvHeader() << "\n";
    for (const auto& cmd : bus.getHistory())
        out << cmd.toCSV() << "\n";

    LUCHS_LOG("[ZoomExport] Exported %zu commands to '%s'\n",
              bus.getHistory().size(), filename.c_str());
}

void printZoomHistory(const CommandBus& bus, int maxLines) {
    const auto& hist = bus.getHistory();
    int total = static_cast<int>(hist.size());
    int begin = std::max(0, total - (maxLines > 0 ? maxLines : 10));

    LUCHS_LOG("[ZoomLog] Showing last %d of %d commands:\n", total - begin, total);
    LUCHS_LOG("%s\n", ZoomCommand::csvHeader().c_str());

    for (int i = begin; i < total; ++i)
        LUCHS_LOG("%s\n", hist[i].toCSV().c_str());
}
