///// OtterDream — Replikatoren
///// File: src/ai/onnx_model.hpp
///// Purpose: ONNX Runtime Wrapper (leichtgewichtig) – Stubs
///// Phase: 2 (Policy-Replikatoren, AOP via ONNX)
///// Hooks: Init/Startup; AOP-Controller nutzt dieses Wrapper-Objekt
///// Depends: <string>
///// Build: /WX-safe ; keine ORT-Header im Header(!)
///// Log-Tags: [REPL/POLICY]  (Alias: [AI/ORT])
///// Created: 2025-11-06 (Europe/Berlin)
///// Notes: Implementierung gated via OTTER_USE_ORT in .cpp

#pragma once
#include <string>

namespace Repl { namespace Policy {

    enum class Ep { CUDA, DML, CPU };

    struct OrtModel {
        bool        loaded = false;
        std::string ep;      // "cuda" | "dml" | "cpu"
        std::string path;    // Modellpfad zur Info/Logs

        bool load(const std::string& modelPath, Ep preferredEp);
        bool is_loaded() const noexcept { return loaded; }
    };

}} // namespace Repl::Policy
