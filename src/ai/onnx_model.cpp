///// OtterDream — Replikatoren
///// File: src/ai/onnx_model.cpp
///// Purpose: ONNX Runtime Wrapper (leichtgewichtig) – Stubs
///// Phase: 2 (Policy-Replikatoren, AOP via ONNX)
///// Hooks: AOP-Controller Init
///// Depends: pch.hpp, luchs_log_host.hpp, ai/onnx_model.hpp
///// Build: /WX-safe ; kein ORT-Header, nur Logs
///// Log-Tags: [REPL/POLICY]  (Alias: [AI/ORT])
///// Created: 2025-11-06 (Europe/Berlin)
///// Notes: Bei !OTTER_USE_ORT: sauberer Fallback mit Log.

#include "pch.hpp"
#include "ai/onnx_model.hpp"
#include "luchs_log_host.hpp"

namespace Repl { namespace Policy {

static const char* ep_name(Ep ep) {
    switch (ep) {
        case Ep::CUDA: return "cuda";
        case Ep::DML:  return "dml";
        default:       return "cpu";
    }
}

bool OrtModel::load(const std::string& modelPath, Ep preferredEp) {
    path = modelPath;
#if OTTER_USE_ORT
    ep = ep_name(preferredEp);
    loaded = true; // Stub: echte ORT-Session folgt später
    LUCHS_LOG_HOST("[REPL/POLICY] ORT stub loaded path=%s ep=%s", path.c_str(), ep.c_str());
    return true;
#else
    (void)preferredEp;
    ep.clear();
    loaded = false;
    LUCHS_LOG_HOST("[REPL/POLICY] ORT missing – disabled");
    return false;
#endif
}

}} // namespace Repl::Policy
