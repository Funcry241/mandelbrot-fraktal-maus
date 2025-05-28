# Datei: CMakeLists.txt
# Maus-Kommentar: Kernel-Refactoring-Projekt – ein einziger dynamischer Kernel mit FindCUDA, nested Parallelität und cuFFT. Entry-Point ist main.cu und Executable heißt mandelbrot_otterdream.

cmake_minimum_required(VERSION 3.18)
project(CudaUnifiedCore LANGUAGES CXX CUDA)

# Klassisches FindCUDA-Modul nutzen
find_package(CUDA REQUIRED)

# Include-Verzeichnisse
include_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# C++ und CUDA Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Separable Compilation für nested kernel launches aktivieren
set(CUDA_SEPARABLE_COMPILATION ON)

# Statische Bibliothek für den CUDA-Core erstellen
cuda_add_library(cuda_core STATIC
    src/core_kernel.cu
)
# cuFFT hinzufügen (CUDART wird automatisch verlinkt)
target_link_libraries(cuda_core
    ${CUDA_CUFFT_LIBRARY}
)

# Executable aus main.cu bauen und auf den Namen mandelbrot_otterdream setzen
cuda_add_executable(mandelbrot_otterdream
    src/main.cu
)
# Auch hier separable compilation einschalten
set_target_properties(mandelbrot_otterdream PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Core-Bibliothek linken
target_link_libraries(mandelbrot_otterdream
    cuda_core
)
