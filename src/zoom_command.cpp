///// Otter: Header-only Zoom commands; TU kept minimal for clean linkage; ASCII-only.
/// ///// Schneefuchs: No duplicate class/struct definitions; no #pragma once in .cpp.
/// ///// Maus: Stub TU that only includes the header; single source of truth in .hpp.
/// ///// Datei: src/zoom_command.cpp

#include "pch.hpp"
#include "zoom_command.hpp"

// Translation unit intentionally minimal.
// Rationale:
// - Keeps build systems that still list this .cpp happy.
// - Avoids duplicate definitions that previously came from header-like content in .cpp.
// - Ensures one authoritative declaration/implementation location in the header.
