///// Otter: Replikatoren - selbstlernender Contextual Bandit (LinUCB/RLS), on-device, ohne Fremdlibraries.
///// Schneefuchs: Deterministische ASCII-Telemetry, Budget-Guard via einfacher dt-Messung; A⁻¹-Update per Sherman-Morrison.
///// Maus: Nur LUCHS_LOG_HOST fürs Logging; kleine API für AOP-Controller (score/select_topk/update/save/load).
///// Datei: src/ai/otter_bandit.hpp

#pragma once

#include <cstdint>
#include <vector>
#include <string>

#include "luchs_log_host.hpp"

namespace otter {
namespace ai {

enum class BanditStage : uint8_t {
    Shadow   = 0,
    Assisted = 1,
    Auto     = 2
};

struct BanditParams {
    float alpha            = 0.8f;
    float epsilon          = 0.05f;
    float lambda           = 1.0e-2f;
    float beta             = 0.6f;
    int   topK             = 3;
    int   retargetInterval = 5;
    float rewardClampLo    = -1.0f;
    float rewardClampHi    = +1.0f;
    bool  persist          = false;
    BanditStage stage      = BanditStage::Shadow;
};

struct BanditScore {
    int   index  = -1;
    float score  = 0.0f;
    float ucb    = 0.0f;
};

class OtterBandit {
public:
    OtterBandit() = default;
    // ✔️ Reihenfolge gedreht: erst konfigurieren, dann init → λ greift in A⁻¹
    OtterBandit(int dim, const BanditParams& p) { configure(p); init(dim); }

    void init(int dim);
    void configure(const BanditParams& p);
    void reset();

    bool is_initialized() const { return m_dim > 0 && (int)m_Ainv.size() == m_dim*m_dim; }
    int  dim() const { return m_dim; }
    const BanditParams& params() const { return m_params; }

    void set_seed(uint32_t s);

    float predict(const float* x) const;
    float uncertainty(const float* x) const;

    std::vector<BanditScore> select_topk(const float* X, int numTiles, int stride) const;

    void update(const float* x, float reward);

    bool save(const char* path) const;
    bool load(const char* path);

    void get_weights(std::vector<float>& out) const;
    std::string brief() const;

private:
    double dot_w(const float* x) const;
    double quadform_Ainv(const float* x) const;
    void   sm_update_Ainv(const float* x);
    void   recompute_w();

    uint32_t rng_next() const;
    float    rng_uniform01() const;

private:
    int m_dim = 0;
    BanditParams m_params{};

    std::vector<double> m_Ainv;
    std::vector<double> m_b;
    std::vector<double> m_w;

    mutable uint32_t m_rngState = 0xC0FFEEu;
};

} // namespace ai
} // namespace otter
