# Datei: CMakeLists.txt
# Maus-Kommentar: Kernel-Refactoring mit dynamischer Parallelität (nested kernel launches).

cmake_minimum_required(VERSION 3.18)
project(CudaUnifiedCore LANGUAGES CXX CUDA)

# Nutze modernes CMake CUDA-Paket
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Aktiviert separable Kompilierung für dynamische Parallelität
enable_language(CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Statische Bibliothek aus dem CUDA-Kern
add_library(cuda_core STATIC
    src/core_kernel.cu
)

# Include-Verzeichnisse
target_include_directories(cuda_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Linke CUDA-Runtime, cuFFT und Device-Runtime für dynamische Parallelität
target_link_libraries(cuda_core PUBLIC
    CUDAToolkit::cudart
    CUDAToolkit::cufft
    CUDAToolkit::cudadevrt
)

# Erstelle ausführbares Programm
add_executable(main_app src/main.cpp)

# Include für main_app
target_include_directories(main_app PUBLIC ${CMAKE_SOURCE_DIR}/src)

# Linke Core-Bibliothek
target_link_libraries(main_app PRIVATE cuda_core)
