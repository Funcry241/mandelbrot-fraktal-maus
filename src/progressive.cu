#include <cstdio>
#include "progressive.hpp"

// Definition — EINMAL
__device__ __managed__ int currentMaxIter = 100;
__device__ __managed__ bool justResetFlag = false;

void resetIterations() {
    currentMaxIter = 100;
    justResetFlag = true;    // 🐭 Reset-Flag setzen    
}

int getCurrentIterations() {
    return currentMaxIter;
}

bool wasJustReset() {
    bool flag = justResetFlag;
    justResetFlag = false;   // 🐭 Einmalig liefern
    return flag;
}
