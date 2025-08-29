#include "nacktmull_types.hpp"
#include <vector>
namespace nm {
  struct View { nm::real cx, cy, spanX, spanY; int w, h; };
  // Füllt "iters" (size = w*h) mit Escape-Iteration (oder maxIter wenn innen)
  void render(const View& v, int maxIter, std::vector<int>& iters) {
    iters.assign(size_t(v.w)*v.h, 0);
    const nm::real two = 2, four = 4;
    for (int y=0; y<v.h; ++y) {
      for (int x=0; x<v.w; ++x) {
        nm::real cr = (nm::real(x)+0.5)/v.w * v.spanX + (v.cx - v.spanX/2);
        nm::real ci = (nm::real(y)+0.5)/v.h * v.spanY + (v.cy - v.spanY/2);
        nm::real zr = 0, zi = 0; int it=0;
        for (; it<maxIter; ++it) {
          // z = z^2 + c
          nm::real zr2 = zr*zr, zi2 = zi*zi;
          if (zr2 + zi2 > four) break;
          nm::real zr_new = zr2 - zi2 + cr;
          zi = (two*zr*zi) + ci;
          zr = zr_new;
        }
        iters[size_t(y)*v.w + x] = it;
      }
    }
  }
}
