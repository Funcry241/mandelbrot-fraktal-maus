cmake_minimum_required(VERSION 3.18)

project(mandelbrot_otterdream
        VERSION 0.1.0
        LANGUAGES C CXX CUDA)

enable_language(CUDA)

# Release-Optimierungen
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")

# Versions-String
add_compile_definitions(OTTERDREAM_VERSION="${PROJECT_VERSION}-beta")

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

# Exportiere compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Globale Include-Pfade
include_directories(${CMAKE_SOURCE_DIR}/src)

# Abhängigkeiten
find_package(boost_multiprecision REQUIRED CONFIG)
find_package(OpenGL            REQUIRED)
find_package(glfw3             CONFIG REQUIRED)
find_package(GLEW              CONFIG REQUIRED)
find_package(imgui             CONFIG REQUIRED)
find_package(CUDAToolkit       REQUIRED)

# Quell-Dateien
set(SOURCE_FILES
    src/main.cpp
    src/gui.cpp
    src/settings.cpp
    src/mandelbrot.cu
    src/mandelbrot_dd.cu       # <– NEU!
    src/rendering/renderer.cpp
    src/compute/boundary.cpp
    src/compute/boundary_gpu.cpp
    src/input/callbacks.cpp
    src/utils/cuda_utils.cpp
)

# Executable erzeugen
add_executable(mandelbrot_otterdream ${SOURCE_FILES})

# Include-Pfade für dieses Target
target_include_directories(mandelbrot_otterdream PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Compiler-Warnungsflags
target_compile_options(mandelbrot_otterdream PRIVATE
    # C++-Warnungen
    $<$<COMPILE_LANGUAGE:CXX>:/W4;/WX;/wd4100>
    # CUDA-spezifische Flags
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=/W4
        -Xcompiler=/WX
        -Wno-deprecated-gpu-targets
    >
)

# Boost-Workaround für CUDA
# Deaktiviert C++14-constexpr in Boost.Math, um doppelte 'constexpr' zu vermeiden
target_compile_definitions(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:BOOST_NO_CXX14_CONSTEXPR>
    $<$<COMPILE_LANGUAGE:CUDA>:BOOST_MATH_NO_CONSTEXPR>
)

# Link-Bibliotheken
target_link_libraries(mandelbrot_otterdream PRIVATE
    Boost::multiprecision
    imgui::imgui
    OpenGL::GL
    glfw
    GLEW::GLEW
    CUDA::cudart
)
