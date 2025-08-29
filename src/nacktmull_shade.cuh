///// MAUS: declaration for Nacktmull shading kernel (header)
#pragma once
// 🦊 Schneefuchs: Nur Deklaration, damit Aufrufer früh binden; Implementation separat. (Bezug zu Schneefuchs)
#include <vector_types.h> // uchar4

extern "C" __global__
void shade_from_iterations(uchar4* rgba, const int* iterations,
                           int width, int height, int maxIter);
