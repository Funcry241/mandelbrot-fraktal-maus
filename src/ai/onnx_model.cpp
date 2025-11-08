///// Otter: ORT session creation with safe fallbacks; logs what EP is requested vs. used; no side effects beyond load.
/// / Schneefuchs: Provider-append intentionally neutral for cross-ORT builds; CPU is default until provider wiring is added.
/// / Maus: If OTTER_USE_ORT=0 → clear ASCII log and return false; exceptions are caught and logged.
/// / Datei: src/ai/onnx_model.cpp

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

#if OTTER_USE_ORT
  // ORT headers only when enabled to keep the public header clean.
  #include <onnxruntime_cxx_api.h>

  // Singleton ORT environment (lazy init).
  static Ort::Env& ort_env() {
      static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "otter");
      return env;
  }

  // Create session options. We log the requested EP; we default to CPU unless
  // explicit provider-append is added later in a targeted patch.
  static Ort::SessionOptions make_session_options(const char* requested_ep, const char** out_used_ep) {
      Ort::SessionOptions opts;
      // Conservative defaults; let ORT decide threads. Keep deterministic behavior.
      opts.SetLogSeverityLevel(3);   // WARNING
      opts.SetLogVerbosityLevel(0);

      // NOTE: Provider attachment intentionally omitted for broad compatibility.
      // Future patch may append CUDA/DML here if build ships provider factories.
      (void)requested_ep;
      if (out_used_ep) *out_used_ep = "cpu";
      return opts;
  }
#endif // OTTER_USE_ORT

bool OrtModel::load(const std::string& modelPath, Ep preferredEp) {
    path = modelPath;

#if OTTER_USE_ORT
    try {
        const char* requested = ep_name(preferredEp);
        const char* used = "cpu";

        auto& env = ort_env();
        auto opts = make_session_options(requested, &used);

        // Try to create a real session to validate the model path.
        Ort::Session session(env, path.c_str(), opts);

        // Optional: probe a tiny bit for diagnostics (kept minimal, ASCII-only).
        size_t in_count  = session.GetInputCount();
        size_t out_count = session.GetOutputCount();

        loaded = true;
        ep = used;

        LUCHS_LOG_HOST("[REPL/POLICY] ORT loaded path=%s ep_used=%s ep_requested=%s io=%zu/%zu",
            path.c_str(), ep.c_str(), requested, in_count, out_count);
        return true;
    } catch (const std::exception& e) {
        loaded = false;
        ep.clear();
        LUCHS_LOG_HOST("[REPL/POLICY] ORT load failed path=%s err=%s", path.c_str(), e.what());
        return false;
    } catch (...) {
        loaded = false;
        ep.clear();
        LUCHS_LOG_HOST("[REPL/POLICY] ORT load failed path=%s err=unknown", path.c_str());
        return false;
    }
#else
    (void)preferredEp;
    ep.clear();
    loaded = false;
    LUCHS_LOG_HOST("[REPL/POLICY] ORT disabled at build time (OTTER_USE_ORT=0)");
    return false;
#endif
}

}} // namespace Repl::Policy
