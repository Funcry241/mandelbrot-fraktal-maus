# Datei: CMakeLists.txt
# Maus-Kommentar: Kernel-Refactoring-Projekt. Ein einzelner CUDA-Kern baut alle Module zusammen. Achte auf Architektur-Flags.

cmake_minimum_required(VERSION 3.18)
project(CudaUnifiedCore LANGUAGES CXX CUDA)

# Include-Verzeichnis für Header
include_directories(${CMAKE_SOURCE_DIR}/src)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Architekturen: SM_50 bis SM_80
set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80)

add_library(cuda_core STATIC
    src/core_kernel.cu
    src/core_kernel.h
)

add_executable(main_app src/main.cpp)
target_link_libraries(main_app PRIVATE cuda_core)
