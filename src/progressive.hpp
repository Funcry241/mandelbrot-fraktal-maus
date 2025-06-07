#pragma once

extern __device__ __managed__ int currentMaxIter;
extern __device__ __managed__ bool justResetFlag;

void resetIterations();
int getCurrentIterations();
void incrementIterations();
bool wasJustReset();
