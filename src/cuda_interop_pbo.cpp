///// Otter: PBO registry management; pick active slot; enforce write-discard.
///// Schneefuchs: No GL sync here; pure resource lifecycle; uses shared state helpers.
///// Maus: Deterministic, low-noise logs; numeric rc only where applicable.
///// Datei: src/cuda_interop_pbo.cpp

#include "pch.hpp"
#include "cuda_interop.hpp"
#include "cuda_interop_state.hpp"

#include "luchs_log_host.hpp"
#include "settings.hpp"
#include "hermelin_buffer.hpp"
#include "bear_CudaPBOResource.hpp"

#include <cuda_gl_interop.h>
#include <unordered_map>

namespace CudaInterop {
using namespace Detail;

// Alias exakt auf den definierten Typ (im selben Namespace)
using PBORes = ::CudaInterop::bear_CudaPBOResource;

void registerAllPBOs(const GLuint* ids, int count) {
    ensureDeviceOnce();
    destroyEventsIfAny();

    for (auto& kv : s_pboMap) delete kv.second;
    s_pboMap.clear();
    s_pboActive = nullptr;

    if (!ids || count <= 0) return;

    for (int i = 0; i < count; ++i) {
        if (!ids[i]) continue;
        auto* res = new PBORes(ids[i]);
        if (res && res->get()) {
            enforceWriteDiscard(res);
            s_pboMap[ids[i]] = res;
        } else {
            delete res;
        }
    }
    for (int i = 0; i < count && !s_pboActive; ++i) {
        auto it = s_pboMap.find(ids[i]);
        if (it != s_pboMap.end()) s_pboActive = it->second;
    }
}

void unregisterAllPBOs() {
    destroyEventsIfAny();
    for (auto& kv : s_pboMap) delete kv.second;
    s_pboMap.clear();
    s_pboActive = nullptr;
}

void registerPBO(const Hermelin::GLBuffer& pbo) {
    ensureDeviceOnce();
    const auto id = static_cast<GLuint>(pbo.id());

    auto it = s_pboMap.find(id);
    if (it == s_pboMap.end()) {
        auto* res = new PBORes(id);
        if (res && res->get()) {
            enforceWriteDiscard(res);
            s_pboMap[id] = res;
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[CUDA-Interop] auto-registered PBO id=%u", id);
            }
        } else {
            delete res;
            LUCHS_LOG_HOST("[FATAL] failed to create CudaPBOResource id=%u", id);
            return;
        }
        it = s_pboMap.find(id);
    }
    s_pboActive = it->second;
}

void unregisterPBO() {
    destroyEventsIfAny();

    if (s_pboActive) {
        for (auto it = s_pboMap.begin(); it != s_pboMap.end(); ++it) {
            if (it->second == s_pboActive) {
                delete it->second;
                s_pboMap.erase(it);
                break;
            }
        }
        s_pboActive = nullptr;
    }
}

} // namespace CudaInterop
