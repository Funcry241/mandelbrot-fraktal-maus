##### Otter: ORT entfernt; Replikatoren (Bandit) hinzugefügt; dynamisches GLEW; CUDA 13 Guard.
##### Schneefuchs: /WX-safe; keine GLU; AppLocal-Kopie; nvrtc optional; stabile Triplets & Archs.
##### Maus: Fügt src/ai/otter_bandit.cpp hinzu; keine ONNX/ORT-Flags mehr im Configure.
##### Datei: CMakeLists.txt
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(mandelbrot_otterdream LANGUAGES CXX CUDA)

if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)

if(CMAKE_GENERATOR MATCHES "Ninja")
  set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON)
  set_property(GLOBAL PROPERTY JOB_POOLS console_pool=1 link_pool=1)
  set(CMAKE_JOB_POOL_LINK link_pool)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90 CACHE STRING "Default architectures if none provided")
endif()

if(NOT DEFINED VCPKG_TARGET_TRIPLET)
  if(WIN32)
    set(VCPKG_TARGET_TRIPLET "x64-windows")
  else()
    set(VCPKG_TARGET_TRIPLET "x64-linux")
  endif()
endif()

if(NOT DEFINED VCPKG_INSTALLED_DIR)
  if(DEFINED ENV{VCPKG_INSTALLED_DIR})
    set(VCPKG_INSTALLED_DIR "$ENV{VCPKG_INSTALLED_DIR}")
  else()
    set(VCPKG_INSTALLED_DIR "${CMAKE_BINARY_DIR}/vcpkg_installed")
  endif()
endif()
set(_VCPKG_PREFIX "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")

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

option(OTTER_USE_NVRTC "Enable NVRTC coloring runtime (links provided by CUDA Toolkit)" OFF)
message(STATUS "OTTER_USE_NVRTC = ${OTTER_USE_NVRTC}")

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(GLFW_LINK_ITEM "")
if(WIN32)
  find_package(glfw3 CONFIG REQUIRED)
  if(TARGET glfw)
    set(GLFW_LINK_ITEM glfw)
  elseif(TARGET glfw3)
    set(GLFW_LINK_ITEM glfw3)
  else()
    find_library(GLFW_LINK_ITEM
      NAMES glfw3 glfw3dll glfw
      HINTS "${_VCPKG_PREFIX}/lib" "${_VCPKG_PREFIX}/debug/lib"
      REQUIRED)
  endif()
else()
  find_package(glfw3 CONFIG REQUIRED)
  set(GLFW_LINK_ITEM glfw)
endif()

set(GLEW_USE_STATIC_LIBS OFF CACHE BOOL "Link GLEW dynamically" FORCE)
remove_definitions(-DGLEW_STATIC)

set(GLEW_LINK_ITEM "")
if(WIN32)
  find_package(GLEW CONFIG REQUIRED)
  if(TARGET GLEW::GLEW)
    set(GLEW_LINK_ITEM GLEW::GLEW)
  elseif(TARGET GLEW::glew)
    set(GLEW_LINK_ITEM GLEW::glew)
  elseif(TARGET GLEW::glew32)
    set(GLEW_LINK_ITEM GLEW::glew32)
  else()
    find_library(GLEW_LINK_ITEM
      NAMES glew32 glew
      HINTS "${_VCPKG_PREFIX}/lib" "${_VCPKG_PREFIX}/debug/lib"
      REQUIRED)
  endif()
else()
  find_package(GLEW REQUIRED)
  if(TARGET GLEW::GLEW)
    set(GLEW_LINK_ITEM GLEW::GLEW)
  else()
    find_library(GLEW_LINK_ITEM NAMES GLEW glew REQUIRED)
  endif()
endif()

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
  src/fps_meter.cpp
  src/renderer_state_cuda.cpp
  src/renderer_state_gl.cpp
  src/heatmap_metrics.cu
  # Axolotel HUD
  src/axolotel_hud.cpp
  src/axolotel_coupler.cpp
  # Dachs HUD
  src/dachs_hud.cpp
  # --- Replikatoren / Orbit & Color ---
  src/perturb_core.cpp
  src/perturb_orbit_mp.cpp
  src/luchs_dsl.cpp
  src/coloring_runtime_nvrtc.cpp
  # --- AI (AOP/Policy) — ohne ORT
  src/ai/aop_controller.cpp
  src/ai/aop_telemetry.cpp
  src/ai/feature_packer.cpp
  src/ai/otter_bandit.cpp  
)

if(MSVC)
  set_property(TARGET mandelbrot_otterdream PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /sdl /permissive- /Zc:__cplusplus /EHsc /Brepro /MP>
    $<$<COMPILE_LANGUAGE:CUDA>:
      -Xcompiler=/W4
      -Xcompiler=/WX
      -Xcompiler=/diagnostics:caret
      -Xcompiler=/permissive-
      -Xcompiler=/Zc:__cplusplus
      -Xcompiler=/EHsc
      -Xcompiler=/wd4505
    >
  )
else()
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror>
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

target_compile_options(mandelbrot_otterdream PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20 --expt-relaxed-constexpr>
)

set_target_properties(mandelbrot_otterdream PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_RUNTIME_LIBRARY Shared
)

target_include_directories(mandelbrot_otterdream PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(mandelbrot_otterdream PRIVATE
  GLEW_NO_IMAGING
  GLEW_NO_GLU
  CAPY_DEFAULT_ON=1
  $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX WIN32_LEAN_AND_MEAN>
)
target_compile_definitions(mandelbrot_otterdream PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:__NV_NO_VECTOR_DEPRECATION_DIAG>
)

if(OTTER_USE_NVRTC)
  target_compile_definitions(mandelbrot_otterdream PRIVATE OTTER_USE_NVRTC=1)
endif()

target_link_libraries(mandelbrot_otterdream PRIVATE
  OpenGL::GL
  ${GLEW_LINK_ITEM}
  ${GLFW_LINK_ITEM}
  $<$<TARGET_EXISTS:CUDA::cudadevrt>:CUDA::cudadevrt>
  Threads::Threads
)

if(OTTER_USE_NVRTC)
  if(TARGET CUDA::nvrtc)
    target_link_libraries(mandelbrot_otterdream PRIVATE CUDA::nvrtc)
  else()
    message(FATAL_ERROR "OTTER_USE_NVRTC=ON, but CUDA::nvrtc target not found.")
  endif()
  if(TARGET CUDA::cuda_driver)
    target_link_libraries(mandelbrot_otterdream PRIVATE CUDA::cuda_driver)
  else()
    message(FATAL_ERROR "OTTER_USE_NVRTC=ON, but CUDA::cuda_driver target not found.")
  endif()
endif()

if(WIN32)
  get_target_property(_OTTER_LINK_LIBS mandelbrot_otterdream LINK_LIBRARIES)
  if(_OTTER_LINK_LIBS)
    set(_OTTER_LINK_LIBS_FILTERED "${_OTTER_LINK_LIBS}")
    list(FILTER _OTTER_LINK_LIBS_FILTERED EXCLUDE REGEX "([/\\\\]|^)[Gg][Ll][Uu]32(\\.lib)?$")
    list(FILTER _OTTER_LINK_LIBS_FILTERED EXCLUDE REGEX "OpenGL::GLU")
    if(NOT _OTTER_LINK_LIBS_FILTERED STREQUAL "${_OTTER_LINK_LIBS}")
      set_property(TARGET mandelbrot_otterdream PROPERTY LINK_LIBRARIES "${_OTTER_LINK_LIBS_FILTERED}")
    endif()
  endif()

  foreach(_glew_tgt IN ITEMS GLEW::GLEW GLEW::glew GLEW::glew32)
    if(TARGET ${_glew_tgt})
      get_target_property(_GLEW_LIBS ${_glew_tgt} INTERFACE_LINK_LIBRARIES)
      if(_GLEW_LIBS)
        set(_GLEW_LIBS_FILTERED "${_GLEW_LIBS}")
        list(FILTER _GLEW_LIBS_FILTERED EXCLUDE REGEX "([/\\\\]|^)[Gg][Ll][Uu]32(\\.lib)?$")
        list(FILTER _GLEW_LIBS_FILTERED EXCLUDE REGEX "OpenGL::GLU")
        if(NOT _GLEW_LIBS_FILTERED STREQUAL "${_GLEW_LIBS}")
          set_property(TARGET ${_glew_tgt} PROPERTY INTERFACE_LINK_LIBRARIES "${_GLEW_LIBS_FILTERED}")
        endif()
      endif()
    endif()
  endforeach()
endif()

option(OTTER_LINK_VERBOSE "Add /VERBOSE:LIB to MSVC link for diagnostics" OFF)
if(MSVC AND OTTER_LINK_VERBOSE)
  target_link_options(mandelbrot_otterdream PRIVATE /VERBOSE:LIB)
endif()

add_custom_command(TARGET mandelbrot_otterdream POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_RUNTIME_DLLS:mandelbrot_otterdream>
          $<TARGET_FILE_DIR:mandelbrot_otterdream>
  COMMAND_EXPAND_LISTS
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(mandelbrot_otterdream PRIVATE -pthread)
endif()

if(MSVC)
  target_link_options(mandelbrot_otterdream PRIVATE
    $<$<CONFIG:Release>:$<HOST_LINK:/INCREMENTAL:NO /OPT:REF /OPT:ICF>>
    $<$<CONFIG:Release>:$<HOST_LINK:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf>>
    $<$<CONFIG:RelWithDebInfo>:$<HOST_LINK:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf>>
    $<HOST_LINK:/Brepro>
  )
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "HIGHENTROPYVA erfordert 64-bit (/MACHINE:X64).")
  endif()
endif()

if(CUDAToolkit_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR "CUDA 13.0 oder neuer ist erforderlich. Gefunden: ${CUDAToolkit_VERSION}")
endif()
find_file(CUDA_GL_INTEROP_HEADER cuda_gl_interop.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
if(NOT CUDA_GL_INTEROP_HEADER)
  message(FATAL_ERROR "cuda_gl_interop.h nicht im CUDAToolkit gefunden.")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
