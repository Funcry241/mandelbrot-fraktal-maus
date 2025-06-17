# Datei: CMakeLists.txt
# Zeilen: 120
# 👝 Maus-Kommentar: Haupt-CMake-Datei für das Mandelbrot-Projekt (CUDA + OpenGL), mit plattformübergreifendem Fix für GLEW, CUDA::cudart und PCH. Funktionalität bleibt 1:1 erhalten. Schneefuchs hätte gesagt: "Wenn’s unter Linux baut, ist es wahrer Code.".

cmake_minimum_required(VERSION 3.25)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # 🧠 Für IntelliSense und Clang-basierte Tools

project(OtterDreamMandelbrot LANGUAGES CXX CUDA)

# ---------------------------------------------------------
# 🔧 Vcpkg Integration
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(VCPKG_MANIFEST_MODE ON)

# ---------------------------------------------------------
# ⚙️ Standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ---------------------------------------------------------
# 🧠 CUDA-Architekturen
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(WARNING "CMAKE_CUDA_ARCHITECTURES ist nicht definiert. "
    "Idealerweise per Preset oder via -DCMAKE_CUDA_ARCHITECTURES gesetzt (z.B. 75;80;86;89)")
endif()

# ---------------------------------------------------------
# 📚 Pakete
find_package(OpenGL 4.3 REQUIRED)
find_package(glew CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# ---------------------------------------------------------
# 🛠️ Executable
add_executable(mandelbrot_otterdream
    src/main.cpp
    src/renderer_core.cu
    src/cuda_interop.cu
    src/core_kernel.cu
    src/hud.cpp
    src/opengl_utils.cpp
    src/progressive.cu
)

# 🧠 Precompiled Header (nur MSVC)
if (MSVC)
  target_precompile_headers(mandelbrot_otterdream PRIVATE src/pch.hpp)
endif()

# Includes
target_include_directories(mandelbrot_otterdream PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
)

# 🔧 CUDA-Includes für PCH-freundliche Header-Auflösung (nur MSVC)
if (MSVC)
    list(APPEND extra_cuda_includes
        "$ENV{CUDA_PATH}/include"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/include"  # Fallback
    )
    foreach(path IN LISTS extra_cuda_includes)
        if(EXISTS "${path}/cuda_runtime.h")
            target_include_directories(mandelbrot_otterdream PRIVATE "${path}")
            break()
        endif()
    endforeach()
endif()

# CUDA Eigenschaften
set_target_properties(mandelbrot_otterdream PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(mandelbrot_otterdream PRIVATE cxx_std_23)

target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive- /Zc:__cplusplus>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wno-unknown-pragmas>
    >
)

target_link_libraries(mandelbrot_otterdream PRIVATE
    OpenGL::GL
    GLEW::glew
    glfw
    Threads::Threads
    -lcudart
)

set_source_files_properties(
    src/renderer_core.cu
    src/cuda_interop.cu
    src/core_kernel.cu
    src/progressive.cu
    PROPERTIES LANGUAGE CUDA
)

# ---------------------------------------------------------
# 🧱 Build-Konfigurationen
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --fmad=false --prec-div=true")

# ---------------------------------------------------------
# 📦 Install-Ziel
install(TARGETS mandelbrot_otterdream RUNTIME DESTINATION bin)
