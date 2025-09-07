# Otter: CUDA 13 guard; use CUDAToolkit include dirs (CCCL/CUB/Thrust); keep dynamic GLEW; purge GLEW_STATIC.
# Schneefuchs: Static CUDA runtime; /MT unified; separable compilation ON; host /WX; explicit OpenGL + Boost; deterministic, duplicate-free defines.
# Maus: Arch 80/86/89/90; CUDA C++20; zero GLEW policy warnings; no late vcpkg hacks; fail early if CUDA < 13.0.

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------- Architectures ----------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90 CACHE STRING "Default architectures if none provided")
endif()

# ---------------------- vcpkg Triplet ----------------------
if(NOT DEFINED VCPKG_TARGET_TRIPLET)
  if(WIN32)
    set(VCPKG_TARGET_TRIPLET "x64-windows")
  else()
    set(VCPKG_TARGET_TRIPLET "x64-linux")
  endif()
endif()

# ---------------------- Project ----------------------------
project(mandelbrot_otterdream LANGUAGES CXX CUDA)

# CUDA runtime policy: force STATIC everywhere
set(CMAKE_CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# ---------------------- Language Std -----------------------
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 23)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
endif()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# GLEW dynamisch (nur Intent dokumentieren)
set(GLEW_USE_STATIC_LIBS OFF CACHE BOOL "Link GLEW dynamically" FORCE)

# ---------------------- Packages ---------------------------
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)                 # OpenGL::GL

# WICHTIG: Unter Linux den CMake-MODULE-Mode erzwingen (um vcpkg-Wrapper-Bug zu umgehen).
# Unter Windows bleibt CONFIG (vcpkg exportiert dort sauber GLEW::GLEW).
if(WIN32)
  find_package(GLEW CONFIG REQUIRED)
else()
  find_package(GLEW MODULE REQUIRED)
endif()

# Robust gegen Namensvarianten der importierten Targets
if(TARGET GLEW::GLEW)
  set(GLEW_TARGET GLEW::GLEW)
elseif(TARGET GLEW::glew)
  set(GLEW_TARGET GLEW::glew)
elseif(TARGET glew::glew)
  set(GLEW_TARGET glew::glew)
else()
  message(FATAL_ERROR "Kein GLEW-CMake-Ziel gefunden (GESUCHT: GLEW::GLEW / GLEW::glew / glew::glew).")
endif()

find_package(glfw3 CONFIG REQUIRED)           # 'glfw'
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS container random system multiprecision)

# CUDA 13 Guard + Header-Check
if(CUDAToolkit_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR "CUDA 13.0 oder neuer ist erforderlich. Gefunden: ${CUDAToolkit_VERSION}")
endif()
find_file(CUDA_GL_INTEROP_HEADER cuda_gl_interop.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
if(NOT CUDA_GL_INTEROP_HEADER)
  message(FATAL_ERROR "cuda_gl_interop.h nicht im CUDAToolkit gefunden.")
endif()

# Legacy-Defines säubern (keine GLEW_STATIC-Leaks)
remove_definitions(-DGLEW_STATIC)
foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CUDA_FLAGS)
  if(DEFINED ${flag_var})
    string(REPLACE "/DGLEW_STATIC" "" ${flag_var} "${${flag_var}}")
    string(REPLACE "-DGLEW_STATIC" "" ${flag_var} "${${flag_var}}")
    string(REPLACE "/UGLEW_STATIC" "" ${flag_var} "${${flag_var}}")
    string(REPLACE "-UGLEW_STATIC" "" ${flag_var} "${${flag_var}}")
    set(${flag_var} "${${flag_var}}" CACHE STRING "" FORCE)
  endif()
endforeach()

# ---------------------- Executable -------------------------
add_executable(mandelbrot_otterdream
  src/main.cpp
  src/luchs_log_host.cpp
  src/luchs_cuda_log_buffer.cu
  src/renderer_core.cu
  src/renderer_loop.cpp
  src/renderer_state.cpp
  src/renderer_pipeline.cpp
  src/renderer_window.cpp
  src/renderer_resources.cpp
  src/opengl_utils.cpp
  src/heatmap_overlay.cpp
  src/warzenschwein_overlay.cpp
  src/cuda_interop.cu
  src/core_kernel.cu
  src/zoom_logic.cpp
  src/zoom_command.cpp
  src/frame_pipeline.cpp
  src/frame_context.cpp
  src/hermelin_buffer.cpp
  src/bear_CudaPBOResource.cpp
  src/frame_capture.cpp
  src/fps_meter.cpp
  src/hud_text.cpp
  src/nacktmull.cu
  src/nacktmull_anchor.cpp
  src/nacktmull_engine.cpp
  src/nacktmull_shade.cu
  src/nacktmull_host.cpp
  src/progressive_iteration.cu
  src/progressive_shade_impl.cu
  src/glfw_bootstrap.cpp
  src/async_frame_writer.cpp
  src/frame_budget_controller.cpp
)

# ---------------------- Compile-Options --------------------
# Windows: einheitliche /MT, Host-Warnungen als Fehler
if(MSVC)
  set_property(TARGET mandelbrot_otterdream PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/MT$<$<CONFIG:Debug>:d>>
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /permissive- /Zc:__cplusplus>
  )
else()
  target_compile_options(mandelbrot_otterdream PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
endif()

# CUDA-spezifisch
target_compile_options(mandelbrot_otterdream PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20 --expt-relaxed-constexpr>
)

# ---------------------- Target-Properties ------------------
set_target_properties(mandelbrot_otterdream PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_RUNTIME_LIBRARY Static
)

# ---------------------- Includes ---------------------------
target_include_directories(mandelbrot_otterdream PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  ${CUDAToolkit_INCLUDE_DIRS}  # CCCL / CUB / Thrust
)

# ---------------------- Defines ----------------------------
target_compile_definitions(mandelbrot_otterdream PRIVATE
  GLEW_NO_IMAGING
  GLEW_NO_GLU
)

# ---------------------- Link -------------------------------
target_link_libraries(mandelbrot_otterdream PRIVATE
  OpenGL::GL
  ${GLEW_TARGET}
  glfw
  CUDA::cudart_static
  Threads::Threads
  Boost::container
  Boost::random
  Boost::system
  Boost::multiprecision
)

# Linux: -pthread explizit
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(mandelbrot_otterdream PRIVATE -pthread)
endif()

# Windows: Host-Linker-Flags (nur C++-Host, nicht CUDA-Device-Link)
if(MSVC)
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/guard:cf>
  )
  target_link_options(mandelbrot_otterdream PRIVATE
    $<$<CONFIG:Release>:$<HOST_LINK:/INCREMENTAL:NO /OPT:REF /OPT:ICF>>
    $<$<CONFIG:Release>:$<HOST_LINK:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf>>
    $<$<CONFIG:RelWithDebInfo>:$<HOST_LINK:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf>>
  )
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "HIGHENTROPYVA erfordert 64-bit (/MACHINE:X64). Bitte 64-bit Toolchain verwenden.")
  endif()
endif()

# ---------------------- Defaults ---------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
