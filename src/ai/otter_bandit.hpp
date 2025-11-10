///// Otter: Replikatoren – selbstlernender Contextual Bandit (LinUCB/RLS), on-device, ohne Fremdlibraries.
///// Schneefuchs: Deterministische ASCII-Telemetry, Budget-Guard via einfacher dt-Messung; A⁻¹-Update per Sherman–Morrison.
///// Maus: Nur LUCHS_LOG_HOST fürs Logging; kleine API für AOP-Controller (score/select_topk/update/save/load).
///// Datei: src/ai/otter_bandit.hpp
#pragma once

#include <cstdint>
#include <vector>
#include <string>

/// Vorwärts-Deklaration, damit wir optional ohne schwere Includes loggen können.
#include "luchs_log_host.hpp"

namespace otter {
namespace ai {

/// Betriebsstufe (Regenwurm-L1 kompatibel)
enum class BanditStage : uint8_t {
    Shadow   = 0,  ///< lernt nur, entscheidet nicht
    Assisted = 1,  ///< wählt Kandidaten, Guards/Hysterese filtern final
    Auto     = 2   ///< steuert Ziel (Guards bleiben Safety-Net)
};

/// Parameter für LinUCB/RLS („Replikatoren“-Defaults sind konservativ)
struct BanditParams {
    float alpha            = 0.8f;   ///< UCB-Skalierung (Exploration)
    float epsilon          = 0.05f;  ///< ε-Exploration-Wahrscheinlichkeit
    float lambda           = 1.0e-2f;///< RLS-Regularisierung (A = λI + Σ xxᵀ)
    float beta             = 0.6f;   ///< Gewichtung von Contrast im Reward (E + β·C)
    int   topK             = 3;      ///< Anzahl Top-Tiles
    int   retargetInterval = 5;      ///< Frames pro Retarget-Zyklus
    float rewardClampLo    = -1.0f;  ///< Reward-Clamp unten
    float rewardClampHi    = +1.0f;  ///< Reward-Clamp oben
    bool  persist          = false;  ///< A⁻¹/b Persistenz aktivieren
    BanditStage stage      = BanditStage::Shadow;
};

/// Ergebnis eines Score-Eintrags
struct BanditScore {
    int   index  = -1;   ///< Tile-Index
    float score  = 0.0f; ///< LinUCB-Score (inkl. Unsicherheit)
    float ucb    = 0.0f; ///< reiner UCB-Term (α·sqrt(xᵀA⁻¹x)), Diagnose
};

/// Selbstlernender Contextual-Bandit (LinUCB mit RLS/Sherman-Morrison)
class OtterBandit {
public:
    OtterBandit() = default;
    OtterBandit(int dim, const BanditParams& p) { init(dim); configure(p); }

    /// Initialisierung (setzt A⁻¹ = (1/λ)·I, b=0, w=0)
    void init(int dim);

    /// Konfiguration zur Laufzeit anpassen (ohne Reset der Gewichte)
    void configure(const BanditParams& p);

    /// Alles auf Anfang (behält die Dimension)
    void reset();

    /// Status
    bool is_initialized() const { return m_dim > 0 && (int)m_Ainv.size() == m_dim*m_dim; }
    int  dim() const { return m_dim; }
    const BanditParams& params() const { return m_params; }

    /// Deterministische Zufallsquelle setzen (ε-Exploration)
    void set_seed(uint32_t s);

    /// Einzel-Vorhersage (ohne UCB): ŷ = xᵀ w
    float predict(const float* x) const;

    /// Unsicherheitsmaß: u = sqrt(xᵀ A⁻¹ x)
    float uncertainty(const float* x) const;

    /// Batch-Auswahl: berechnet LinUCB-Scores und liefert Top-k Indizes
    /// X: Zeiger auf row-major Feature-Matrix [numTiles x stride], stride==dim()
    std::vector<BanditScore> select_topk(const float* X, int numTiles, int stride) const;

    /// Online-Update mit einem Sample (x, r). O(d²) via Sherman–Morrison; w neu aus b und A⁻¹.
    void update(const float* x, float reward);

    /// Persistenz
    bool save(const char* path) const;
    bool load(const char* path);

    /// Gewichte (w) herausgeben (z. B. für Debug/HUD)
    void get_weights(std::vector<float>& out) const;

    /// Kurze Text-Zusammenfassung (für Telemetrie)
    std::string brief() const;

private:
    // Mathe-Hilfen (double für Stabilität, extern bleiben Inputs float)
    double dot_w(const float* x) const;                // xᵀ w
    double quadform_Ainv(const float* x) const;        // xᵀ A⁻¹ x
    void   sm_update_Ainv(const float* x);             // Sherman–Morrison
    void   recompute_w();                              // w = A⁻¹ b

    // RNG (xorshift32) für ε-Exploration
    uint32_t rng_next() const;
    float    rng_uniform01() const;

private:
    int m_dim = 0;
    BanditParams m_params{};

    // RLS-Zustand in double
    std::vector<double> m_Ainv; // d×d, row-major
    std::vector<double> m_b;    // d
    std::vector<double> m_w;    // d

    mutable uint32_t m_rngState = 0xC0FFEEu;
};

} // namespace ai
} // namespace otter
