///// Otter: Feature packer für Replikatoren - Bandit-Featurematrix (~20D), deterministisch.
///// Schneefuchs: Bounds-safe, klare Shapes; Header/Source synchron; keine Fremdlibs, ASCII-Logs.
///// Maus: Bandit-Matrix row-major [Tiles x d]; Koords in NDC, Grad/Stats 3x3; stride == dim.
///// Datei: src/ai/feature_packer.hpp
#pragma once

#include <vector>
#include <cstddef>

namespace Repl { namespace Feat {

// -------------------- Bandit-Featurematrix ----------------------------------

// Row-major Featurematrix für den selbstlernenden Bandit:
// X: [numTiles x dim], stride == dim; Features je Tile:
//  - E/C Wert, 3x3-Mean/Var, Grad-Proxy |d/dx|+|d/dy|, Range (max-min)
//  - normierte Kachelkoords in NDC [-1,1], r, cos(theta), sin(theta), center-bias exp(-r^2/sigma^2)
struct BanditMatrix {
    std::vector<float> data;   // size = numTiles * dim; row-major
    int dim     = 0;           // Feature-Dimension d
    int tilesX  = 0;           // Tx
    int tilesY  = 0;           // Ty
    int stride  = 0;           // == dim
    int statsPx = 0;           // Grid tile size (pixels)
};

// Erzeugt die Bandit-Featurematrix aus E/C-Heatmap und Grid-Layout.
// - width/height/statsPx definieren Tx x Ty wie beim Overlay
// - Feature-Dimension ist fest (d ~ 20); stabil dokumentiert in .cpp
// - Koords in NDC bezogen auf Tile-Center
BanditMatrix make_bandit_feature_matrix(const std::vector<float>& entropy,
                                        const std::vector<float>& contrast,
                                        int width, int height, int statsPx);

}} // namespace Repl::Feat
