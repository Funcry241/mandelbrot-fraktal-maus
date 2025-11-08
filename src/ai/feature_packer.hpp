///// Otter: Feature packer for AOP policy (dry-run) — packs heatmap E/C into NCHW tensor
///// Schneefuchs: ASCII logs; header/source in sync; no hidden macros; single-responsibility
///// Maus: Shape [1,2,Ty,Tx]; tiles derived from width/height & statsPx; zero-fill on mismatch
///// Datei: src/ai/feature_packer.hpp
#pragma once

#include <vector>
#include <cstddef>

// NCHW layout, row-major: index = ((n*C + c)*H + y)*W + x
// N=1, C=2 (channels: 0=Entropy, 1=Contrast), H=Ty, W=Tx
namespace Repl { namespace Feat {

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

}} // namespace Repl::Feat
