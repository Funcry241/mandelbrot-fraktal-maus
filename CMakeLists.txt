# Otter: CUDA 13 guard; use CUDAToolkit include dirs (CCCL/CUB/Thrust); keep dynamic GLEW; purge GLEW_STATIC.
# Schneefuchs: Static CUDA runtime; /MT unified; separable compilation ON; host /WX; explicit OpenGL + Boost; deterministic, duplicate-free defines.
# Maus: Arch 80/86/89/90; CUDA C++20; zero GLEW policy warnings; no late vcpkg hacks; fail early if CUDA < 13.0.

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(mandelbrot_otterdream LANGUAGES CXX CUDA)

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

# ---------------------- CUDA Runtime -----------------------
set(CMAKE_CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# ---------------------- Packages ---------------------------
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)                 # OpenGL::GL
find_package(glfw3 CONFIG REQUIRED)           # glfw
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# ---------------------- CUDA 13 Guard ----------------------
if(CUDAToolkit_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR "CUDA 13.0 oder neuer ist erforderlich. Gefunden: ${CUDAToolkit_VERSION}")
endif()
find_file(CUDA_GL_INTEROP_HEADER cuda_gl_interop.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
if(NOT CUDA_GL_INTEROP_HEADER)
  message(FATAL_ERROR "cuda_gl_interop.h nicht im CUDAToolkit gefunden.")
endif()

# ---------------------- GLEW (dynamisch) -------------------
# INTENT: dynamisch linken (kein GLEW_STATIC)
set(GLEW_USE_STATIC_LIBS OFF CACHE BOOL "Link GLEW dynamically" FORCE)

# Alte Leaks säubern
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

# Windows: regulär via CONFIG (vcpkg liefert GLEW::GLEW / GLEW::glew)
# Linux: Wrapper ist buggy -> manuell aus vcpkg_installed importieren (kein find_package für GLEW!)
if(WIN32)
  find_package(GLEW CONFIG REQUIRED)
  if(TARGET GLEW::GLEW)
    set(GLEW_TARGET GLEW::GLEW)
  elseif(TARGET GLEW::glew)
    set(GLEW_TARGET GLEW::glew)
  else()
    message(FATAL_ERROR "GLEW CMake-Target nicht gefunden (erwartet GLEW::GLEW oder GLEW::glew).")
  endif()
else()
  # vcpkg Installationswurzel im Buildbaum
  if(NOT DEFINED VCPKG_INSTALLED_DIR)
    if(DEFINED ENV{VCPKG_INSTALLED_DIR})
      set(VCPKG_INSTALLED_DIR "$ENV{VCPKG_INSTALLED_DIR}")
    else()
      set(VCPKG_INSTALLED_DIR "${CMAKE_BINARY_DIR}/vcpkg_installed")
    endif()
  endif()
  set(_VCPKG_PREFIX "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")

  # Header + Lib aus vcpkg aufnehmen
  find_path(GLEW_INCLUDE_DIR GL/glew.h
    HINTS "${_VCPKG_PREFIX}/include" REQUIRED)
  find_library(GLEW_LIBRARY NAMES GLEW glew
    HINTS "${_VCPKG_PREFIX}/lib" "${_VCPKG_PREFIX}/debug/lib" REQUIRED)

  # Importiertes Ziel bereitstellen wie von find_package erwartet
  add_library(GLEW::GLEW UNKNOWN IMPORTED)
  set_target_properties(GLEW::GLEW PROPERTIES
    IMPORTED_LOCATION "${GLEW_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GLEW_INCLUDE_DIR}"
  )
  set(GLEW_TARGET GLEW::GLEW)
endif()

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
)

# ---------------------- Compile-Options --------------------
if(MSVC)
  # Einheitliche /MT (und /MTd) – Host & NVCC-Host-Compiler
  set_property(TARGET mandelbrot_otterdream PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

  # Host (C++) streng, CUDA-Host nur /MT via -Xcompiler
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /permissive- /Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/MT$<$<CONFIG:Debug>:d>>
  )
else()
  # Host-Warnflags NUR für C++ (nicht für NVCC!)
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror>
  )
  # GCC/Clang: MSVC-Pragmas (#pragma warning ...) ignorieren, sonst mit -Werror fatal
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mandelbrot_otterdream PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>
    )
  endif()
endif()

# CUDA-spezifisch (reine Device-Flags)
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
)

# Linux: -pthread explizit
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(mandelbrot_otterdream PRIVATE -pthread)
endif()

# Windows: Host-Linker-Härtung (nur C++-Host, nicht Device-Link)
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
