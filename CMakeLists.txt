##### Otter: CUDA 13 guard; CCCL/CUB/Thrust via CUDAToolkit; dynamic GLEW (no GLEW_STATIC); Ninja response-files.
##### Schneefuchs: Static CUDA runtime; /MT unified; separable compilation ON; host /WX; duplicate-free defines; safe device-runtime link.
##### Maus: Arch 80/86/89/90; CUDA C++20; deterministic build; fail early if CUDA < 13.0; silence C4505 only for CUDA TUs; Capybara-only path.
##### Datei: CMakeLists.txt

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(mandelbrot_otterdream LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------- Ninja QoL --------------------------
if(CMAKE_GENERATOR MATCHES "Ninja")
  set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON)
  set_property(GLOBAL PROPERTY JOB_POOLS console_pool=1 link_pool=1)
  set(CMAKE_JOB_POOL_LINK link_pool)
endif()

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
set(GLEW_USE_STATIC_LIBS OFF CACHE BOOL "Link GLEW dynamically" FORCE)
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
# Linux: per importiertem Ziel aus vcpkg_installed (robust gg. Find-GLEW-Issues)
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
  if(NOT DEFINED VCPKG_INSTALLED_DIR)
    if(DEFINED ENV{VCPKG_INSTALLED_DIR})
      set(VCPKG_INSTALLED_DIR "$ENV{VCPKG_INSTALLED_DIR}")
    else()
      set(VCPKG_INSTALLED_DIR "${CMAKE_BINARY_DIR}/vcpkg_installed")
    endif()
  endif()
  set(_VCPKG_PREFIX "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")

  find_path(GLEW_INCLUDE_DIR GL/glew.h HINTS "${_VCPKG_PREFIX}/include" REQUIRED)
  find_library(GLEW_LIBRARY NAMES GLEW glew HINTS "${_VCPKG_PREFIX}/lib" "${_VCPKG_PREFIX}/debug/lib" REQUIRED)

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
  src/renderer_pipeline.cpp
  src/renderer_window.cpp
  src/renderer_resources.cpp
  src/heatmap_overlay.cpp
  src/hud_text.cpp
  src/warzenschwein_overlay.cpp
  src/cuda_interop.cu
  # Capybara path
  src/capybara_render_kernel.cu
  src/colorize_iterations.cu
  # Zoom & control
  src/zoom_logic.cpp  
  # Frame + context + utilities
  src/frame_pipeline.cpp
  src/frame_context.cpp
  src/hermelin_buffer.cpp
  src/bear_CudaPBOResource.cpp
  src/frame_capture.cpp
  src/fps_meter.cpp
  src/renderer_state_cuda.cpp
  src/renderer_state_gl.cpp    
  src/heatmap_metrics.cu
  src/edge_detector.cu
)

# ---------------------- Compile-Options --------------------
if(MSVC)
  # Einheitliche /MT (und /MTd) – Host & NVCC-Host-Compiler
  set_property(TARGET mandelbrot_otterdream PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

  # Host (C++) STRIKT mit /WX; CUDA-Host ebenfalls /WX, aber gezielt C4505 aus CUDA CRT unterdrücken
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /sdl /permissive- /Zc:__cplusplus /Brepro>
    $<$<COMPILE_LANGUAGE:CUDA>:
      -Xcompiler=/W4
      -Xcompiler=/WX
      -Xcompiler=/permissive-
      -Xcompiler=/Zc:__cplusplus
      -Xcompiler=/wd4505
      -Xcompiler=/MT$<$<CONFIG:Debug>:d>
      -Xcompiler=/Brepro
    >
  )
else()
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror>
  )
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      -Xcompiler=-Wall
      -Xcompiler=-Wextra
      -Xcompiler=-Wpedantic
      -Xcompiler=-Werror
    >
  )
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
  CAPY_DEFAULT_ON=1
)
# CUDA 13 vector-types deprecation guard (harmless if unused)
target_compile_definitions(mandelbrot_otterdream PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:__NV_NO_VECTOR_DEPRECATION_DIAG>
)

# ---------------------- Link -------------------------------
target_link_libraries(mandelbrot_otterdream PRIVATE
  OpenGL::GL
  ${GLEW_TARGET}
  glfw
  CUDA::cudart_static
  $<$<TARGET_EXISTS:CUDA::cudadevrt>:CUDA::cudadevrt>
  Threads::Threads
)

# Linux: -pthread explizit
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(mandelbrot_otterdream PRIVATE -pthread)
endif()

# Windows: Host-Linker-Härtung + deterministisch (Host-Link)
if(MSVC)
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/guard:cf>
  )
  target_link_options(mandelbrot_otterdream PRIVATE
    $<$<CONFIG:Release>:$<HOST_LINK:/INCREMENTAL:NO /OPT:REF /OPT:ICF>>
    $<$<CONFIG:Release>:$<HOST_LINK:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf>>
    $<$<CONFIG:RelWithDebInfo>:$<HOST_LINK:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf>>
    $<HOST_LINK:/Brepro>
  )
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "HIGHENTROPYVA erfordert 64-bit (/MACHINE:X64). Bitte 64-bit Toolchain verwenden.")
  endif()
endif()

# ---------------------- Defaults ---------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
