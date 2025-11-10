///// Otter: Feature packer für Replikatoren – Heatmap-NCHW + Bandit-Featurematrix (d≈20), deterministisch.
///// Schneefuchs: Bounds-safe, zero-fill, klare Shapes; Header/Source synchron; keine Fremdlibs, ASCII-Logs.
///// Maus: NCHW [1,2,Ty,Tx] für E/C; Bandit-Matrix row-major [Tiles × d]; Koords in NDC, Grad/Stats 3×3.
///// Datei: src/ai/feature_packer.hpp
#pragma once

#include <vector>
#include <cstddef>

namespace Repl { namespace Feat {

// -------------------- NCHW-Pack für Heatmap (bestehend) ---------------------

// NCHW layout, row-major: index = ((n*C + c)*H + y)*W + x
// N=1, C=2 (channels: 0=Entropy, 1=Contrast), H=Ty, W=Tx
struct PackedFeatures {
    std::vector<float> data;   // size = N*C*H*W (N=1, C=2)
    int N = 1;
    int C = 2;
    int H = 0;                 // Ty
    int W = 0;                 // Tx
    int tilesX = 0;            // Tx
    int tilesY = 0;            // Ty
    int statsPx = 0;           // Grid tile size (pixels)
};

// Packs heatmap features into a fixed NCHW tensor [1,2,Ty,Tx].
// - entropy/contrast: size should be Tx*Ty; we derive Tx,Ty from (width,height,statsPx)
// - width/height: current framebuffer resolution
// - statsPx: metrics grid tile size (pixels)
// Behavior on size mismatch: packs min(common, expected) elements and zero-fills the rest.
// Emits a short ASCII meta-log line via LUCHS_LOG_HOST.
PackedFeatures pack_heatmap_features(const std::vector<float>& entropy,
                                     const std::vector<float>& contrast,
                                     int width, int height, int statsPx);

// -------------------- Bandit-Featurematrix (neu) ----------------------------

// Row-major Featurematrix für den selbstlernenden Bandit:
// X: [numTiles × dim], stride==dim; Features je Tile:
//  - E/C Wert, 3×3-Mean/Var, Grad-Proxy |∂x|+|∂y|, Range (max-min)
//  - normierte Kachelkoords in NDC [-1,1], r, cos(θ), sin(θ), center-bias exp(-r^2/σ^2)
struct BanditMatrix {
    std::vector<float> data;   // size = numTiles * dim; row-major
    int dim     = 0;           // Feature-Dimension d
    int tilesX  = 0;           // Tx
    int tilesY  = 0;           // Ty
    int stride  = 0;           // == dim
    int statsPx = 0;           // Grid tile size (pixels)
};

// Erzeugt die Bandit-Featurematrix aus E/C-Heatmap und Grid-Layout.
// - width/height/statsPx definieren Tx×Ty wie beim Overlay
// - Feature-Dimension ist fest (d≈20); stabil dokumentiert in .cpp
// - Zero-fill bei Untermengen; Koords in NDC bezogen auf Tile-Center
BanditMatrix make_bandit_feature_matrix(const std::vector<float>& entropy,
                                        const std::vector<float>& contrast,
                                        int width, int height, int statsPx);

}} // namespace Repl::Feat
