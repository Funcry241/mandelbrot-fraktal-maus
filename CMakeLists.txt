# Datei: CMakeLists.txt
# Maus-Kommentar: Kernel-Refactoring, Build mit NVCC für .cu-Quellen (einschließlich main.cu) für Kernel-Launch-Syntax.

cmake_minimum_required(VERSION 3.18)
project(CudaUnifiedCore LANGUAGES CXX CUDA)

# Nutze klassisches FindCUDA für Kompatibilität
find_package(CUDA REQUIRED)

# Include-Verzeichnisse
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${CMAKE_SOURCE_DIR}/src)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Aktiviere separable Kompilierung für nested kernel launches
set(CUDA_SEPARABLE_COMPILATION ON)

# Statische Bibliothek für CUDA-Core
cuda_add_library(cuda_core STATIC
    src/core_kernel.cu
)
# cuda_add_library verlinkt CUDART automatisch; cuFFT hinzufügen
target_link_libraries(cuda_core
    ${CUDA_CUFFT_LIBRARY}
)

# Kompiliere Host-Code als .cu, damit <<< >>> unterstützt wird
file(RENAME ${CMAKE_SOURCE_DIR}/src/main.cpp ${CMAKE_SOURCE_DIR}/src/main.cu)

# Erstelle ausführbares Programm aus main.cu
cuda_add_executable(main_app src/main.cu)
# Linke Core-Bibliothek
target_link_libraries(main_app
    cuda_core
)
