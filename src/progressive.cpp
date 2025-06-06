#include <cstdio>
#include "progressive.hpp"

#ifdef __CUDACC__
__device__ __managed__ int currentMaxIter = 100;
__device__ __managed__ bool justResetFlag = false;
#else
int currentMaxIter = 100;
bool justResetFlag = false;
#endif

void resetIterations() {
    currentMaxIter = 100;
    justResetFlag = true;    // 🐭 Reset-Flag setzen
    std::fprintf(stdout, "[RESET] Iterations reset to %d.\n", currentMaxIter);
}

int getCurrentIterations() {
    return currentMaxIter;
}

bool wasJustReset() {
    bool flag = justResetFlag;
    justResetFlag = false;   // 🐭 Einmalig liefern
    return flag;
}
