# Datei: CMakeLists.txt
# Maus-Kommentar: Kernel-Refactoring-Projekt. Ein einzelner CUDA-Kern baut alle Module zusammen. Achte auf Architektur-Flags.

cmake_minimum_required(VERSION 3.18)
project(CudaUnifiedCore LANGUAGES CXX CUDA)

# CUDA finden (klassisches FindCUDA-Modul)
find_package(CUDA REQUIRED)

# Include-Verzeichnisse (global)
include_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Architekturen: SM_50 bis SM_80
set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80)

# Statische Bibliothek aus dem CUDA-Kern
cuda_add_library(cuda_core STATIC
    src/core_kernel.cu
)

# Linke cuFFT (cuda_add_library verlinkt CUDART automatisch)
target_link_libraries(cuda_core
    ${CUDA_CUFFT_LIBRARY}
)

# Erstelle ausführbares Programm
cuda_add_executable(main_app src/main.cpp)

# Linke die Core-Bibliothek
target_link_libraries(main_app
    cuda_core
)
