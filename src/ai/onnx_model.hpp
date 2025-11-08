///// Otter: Lightweight ORT wrapper (real session load, CPU default), ASCII logs, dry-run safe.
///// Schneefuchs: Header stays clean (no ORT includes), /WX-safe, implementation gated via OTTER_USE_ORT in .cpp.
/// / Maus: On OTTER_USE_ORT=0 → graceful fallback with clear log; no API changes for AOP controller.
/// / Datei: src/ai/onnx_model.hpp

#pragma once
#include <string>

namespace Repl { namespace Policy {

    enum class Ep { CUDA, DML, CPU };

    struct OrtModel {
        bool        loaded = false;
        std::string ep;      // "cuda" | "dml" | "cpu" (actual in-use ep; may differ from preferred)
        std::string path;    // model path for logs/info

        // Load model; returns true on success. Chooses CPU by default unless provider wiring is added later.
        bool load(const std::string& modelPath, Ep preferredEp);

        bool is_loaded() const noexcept { return loaded; }
    };

}} // namespace Repl::Policy
