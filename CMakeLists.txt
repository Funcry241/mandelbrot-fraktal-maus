# Datei: CMakeLists.txt
# Maus-Kommentar: Kernel-Refactoring, FindCUDA-Modul nutzen, dynamische Parallelität und cuFFT.

cmake_minimum_required(VERSION 3.18)
project(CudaUnifiedCore LANGUAGES CXX CUDA)

# Nutze klassisches FindCUDA
find_package(CUDA REQUIRED)

# Include-Verzeichnisse
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${CMAKE_SOURCE_DIR}/src)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Aktiviere separable Kompilierung für dynamische Parallelität
set(CUDA_SEPARABLE_COMPILATION ON)

# Statische Bibliothek aus dem CUDA-Kern
cuda_add_library(cuda_core STATIC
    src/core_kernel.cu
)

# cuda_add_library verlinkt CUDART automatisch
# Füge cuFFT hinzu
target_link_libraries(cuda_core
    ${CUDA_CUFFT_LIBRARY}
)

# Erstelle ausführbares Programm
cuda_add_executable(main_app src/main.cpp)

# Linke die Core-Bibliothek
target_link_libraries(main_app
    cuda_core
)
