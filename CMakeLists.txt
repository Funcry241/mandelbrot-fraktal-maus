# Datei: CMakeLists.txt
# Maus-Kommentar: Kernel-Refactoring-Projekt. Ein einzelner CUDA-Kern baut alle Module zusammen. Achte auf Architektur-Flags.

cmake_minimum_required(VERSION 3.18)
project(CudaUnifiedCore LANGUAGES CXX CUDA)

# CUDA Toolkit finden für Include-Pfade und Libraries
find_package(CUDAToolkit REQUIRED)

# Include-Verzeichnisse
target_include_directories(cuda_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${CUDAToolkit_INCLUDE_DIRS}
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Architekturen: SM_50 bis SM_80
set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80)

# Erstelle statische Bibliothek aus dem CUDA-Kern
add_library(cuda_core STATIC
    src/core_kernel.cu
)

# Linke CUDA Runtime und cuFFT
target_link_libraries(cuda_core
    PUBLIC
        CUDAToolkit::cudart
        CUDAToolkit::cufft
)

# Erstelle ausführbares Programm
add_executable(main_app src/main.cpp)
target_link_libraries(main_app PRIVATE cuda_core)
