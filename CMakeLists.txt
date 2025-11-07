##### Otter: Gestern’s Layout, nur gefixt: Host-only Linkerflags via $<HOST_LINK:1>, keine $<DEVICE_LINK> in target_link_libraries; vcpkg-Seeding wie gehabt.
##### Schneefuchs: Dynamic GLEW, keine GLU-Header; CUDA13-Guard; Ninja-Responsefiles; klare Fehlermeldungen bei fehlendem glfw/glew.
##### Maus: Minimal invasiv, deterministisch; gleiche Sources; HIGHENTROPYVA nicht mehr im NVCC-dlink.
##### Datei: CMakeLists.txt

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(mandelbrot_otterdream LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Multi-config map
if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES "Release;Debug" CACHE STRING "" FORCE)
endif()
set(CMAKE_TRY_COMPILE_CONFIGURATION Release)
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL     Release)

# MSVC runtime via CMP0091
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

# Ninja QoL
if(CMAKE_GENERATOR MATCHES "Ninja")
  set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON)
endif()

# Arch
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90 CACHE STRING "" FORCE)
endif()

# vcpkg roots & prefix
if(NOT DEFINED VCPKG_TARGET_TRIPLET)
  set(VCPKG_TARGET_TRIPLET "x64-windows")
endif()
if(NOT DEFINED VCPKG_INSTALLED_DIR)
  if(DEFINED ENV{VCPKG_ROOT})
    set(VCPKG_INSTALLED_DIR "$ENV{VCPKG_ROOT}/installed")
  else()
    set(VCPKG_INSTALLED_DIR "${CMAKE_BINARY_DIR}/vcpkg_installed")
  endif()
endif()
set(_VCPKG_PREFIX "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
list(APPEND CMAKE_PREFIX_PATH
  "${_VCPKG_PREFIX}"
  "${_VCPKG_PREFIX}/share"
  "${_VCPKG_PREFIX}/share/glfw3"
  "${_VCPKG_PREFIX}/share/glew"
)

function(_otter_try_vcpkg_install)
  if(DEFINED ENV{VCPKG_ROOT})
    set(_VCPKG_EXE "$ENV{VCPKG_ROOT}/vcpkg.exe")
    if(EXISTS "${_VCPKG_EXE}")
      message(STATUS "-- Running vcpkg install")
      execute_process(COMMAND "${_VCPKG_EXE}" install "glfw3:${VCPKG_TARGET_TRIPLET}" "glew:${VCPKG_TARGET_TRIPLET}" "opengl:${VCPKG_TARGET_TRIPLET}"
                      RESULT_VARIABLE _ret OUTPUT_QUIET ERROR_QUIET)
      if(_ret EQUAL 0)
        message(STATUS "-- Running vcpkg install - done")
        list(APPEND CMAKE_PREFIX_PATH "${_VCPKG_PREFIX}" "${_VCPKG_PREFIX}/share")
        set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
      else()
        message(WARNING "vcpkg install failed (code=${_ret}) – continuing with fallbacks")
      endif()
    endif()
  endif()
endfunction()

# Language standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Core packages
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# GLFW robust lookup
set(GLFW_LINK_ITEM "")
find_package(glfw3 CONFIG QUIET)
if(TARGET glfw)
  set(GLFW_LINK_ITEM glfw)
elseif(TARGET glfw3)
  set(GLFW_LINK_ITEM glfw3)
else()
  _otter_try_vcpkg_install()
  find_package(glfw3 CONFIG QUIET)
  if(TARGET glfw)
    set(GLFW_LINK_ITEM glfw)
  elseif(TARGET glfw3)
    set(GLFW_LINK_ITEM glfw3)
  else()
    find_library(GLFW_LINK_ITEM NAMES glfw3 glfw3dll glfw
      HINTS "${_VCPKG_PREFIX}/lib" "${_VCPKG_PREFIX}/debug/lib")
    if(NOT GLFW_LINK_ITEM)
      message(FATAL_ERROR "GLFW not found. Use vcpkg (set %VCPKG_ROOT% and install glfw3) or pass -DCMAKE_TOOLCHAIN_FILE=.../vcpkg.cmake.")
    endif()
  endif()
endif()

# GLEW dynamic; no GLU
set(GLEW_USE_STATIC_LIBS OFF CACHE BOOL "" FORCE)
set(GLEW_LINK_ITEM "")
find_package(GLEW CONFIG QUIET)
if(TARGET GLEW::GLEW)
  set(GLEW_LINK_ITEM GLEW::GLEW)
elseif(TARGET GLEW::glew)
  set(GLEW_LINK_ITEM GLEW::glew)
elseif(TARGET GLEW::glew32)
  set(GLEW_LINK_ITEM GLEW::glew32)
else()
  _otter_try_vcpkg_install()
  find_package(GLEW CONFIG QUIET)
  if(TARGET GLEW::GLEW)
    set(GLEW_LINK_ITEM GLEW::GLEW)
  elseif(TARGET GLEW::glew)
    set(GLEW_LINK_ITEM GLEW::glew)
  elseif(TARGET GLEW::glew32)
    set(GLEW_LINK_ITEM GLEW::glew32)
  else()
    find_library(GLEW_LINK_ITEM NAMES glew32 glew GLEW
      HINTS "${_VCPKG_PREFIX}/lib" "${_VCPKG_PREFIX}/debug/lib")
    if(NOT GLEW_LINK_ITEM)
      message(FATAL_ERROR "GLEW not found. Use vcpkg (set %VCPKG_ROOT% and install glew).")
    endif()
  endif()
endif()

# Target
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
  src/capybara_render_kernel.cu
  src/colorize_iterations.cu
  src/zoom_logic.cpp
  src/frame_pipeline.cpp
  src/frame_context.cpp
  src/hermelin_buffer.cpp
  src/bear_CudaPBOResource.cpp
  src/fps_meter.cpp
  src/renderer_state_cuda.cpp
  src/renderer_state_gl.cpp
  src/heatmap_metrics.cu
  src/axolotel_hud.cpp
  src/axolotel_coupler.cpp
  src/dachs_hud.cpp
  src/perturb_core.cpp
  src/perturb_orbit_mp.cpp
  src/luchs_dsl.cpp
)

set_property(TARGET mandelbrot_otterdream PROPERTY LINKER_LANGUAGE CXX)
set_target_properties(mandelbrot_otterdream PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_RUNTIME_LIBRARY Shared
)

# Compile options
if(MSVC)
  set_property(TARGET mandelbrot_otterdream PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /sdl /permissive- /Zc:__cplusplus /EHsc /Brepro>
    $<$<COMPILE_LANGUAGE:CUDA>:
      -Xcompiler=/W4 -Xcompiler=/WX -Xcompiler=/diagnostics:caret
      -Xcompiler=/permissive- -Xcompiler=/Zc:__cplusplus -Xcompiler=/EHsc
      -Xcompiler=/wd4505
    >
  )
else()
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wpedantic -Xcompiler=-Werror>
  )
endif()
target_compile_options(mandelbrot_otterdream PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20 --expt-relaxed-constexpr>
)

# Includes & defines
target_include_directories(mandelbrot_otterdream PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  ${CUDAToolkit_INCLUDE_DIRS}
  "${_VCPKG_PREFIX}/include"
)
target_compile_definitions(mandelbrot_otterdream PRIVATE
  GLEW_NO_IMAGING GLEW_NO_GLU CAPY_DEFAULT_ON=1
  $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX WIN32_LEAN_AND_MEAN>
  $<$<COMPILE_LANGUAGE:CUDA>:__NV_NO_VECTOR_DEPRECATION_DIAG>
)

# Libraries (keine DEVICE_LINK-Genex hier!)
if(WIN32)
  set(_WIN_LIBS user32 gdi32 shell32 advapi32)
endif()
target_link_libraries(mandelbrot_otterdream PRIVATE
  ${OpenGL_GL_PREFERENCE} ${GLEW_LINK_ITEM} ${GLFW_LINK_ITEM} Threads::Threads
  ${_WIN_LIBS}
)

# Host-only link options (CMake 3.28: $<HOST_LINK:…> verfügbar)
option(OTTER_LINK_VERBOSE "MSVC /VERBOSE:LIB on host link" ON)
if(MSVC)
  target_link_options(mandelbrot_otterdream PRIVATE
    $<$<HOST_LINK:1>:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf /Brepro>
    $<$<AND:$<HOST_LINK:1>,$<CONFIG:Release>>:/INCREMENTAL:NO /OPT:REF /OPT:ICF>
    $<$<HOST_LINK:1>:/NODEFAULTLIB:GLU32 /NODEFAULTLIB:glu32>
    $<$<AND:$<HOST_LINK:1>,$<BOOL:${OTTER_LINK_VERBOSE}>>:/VERBOSE:LIB>
  )
endif()

# Copy runtime DLLs beside EXE (host link only executes this step)
add_custom_command(TARGET mandelbrot_otterdream POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:mandelbrot_otterdream> $<TARGET_FILE_DIR:mandelbrot_otterdream>
  COMMAND_EXPAND_LISTS
)

# CUDA 13 guard & GL interop header
if(CUDAToolkit_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR "CUDA 13.0 or newer required; found ${CUDAToolkit_VERSION}")
endif()
find_file(CUDA_GL_INTEROP_HEADER cuda_gl_interop.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
if(NOT CUDA_GL_INTEROP_HEADER)
  message(FATAL_ERROR "cuda_gl_interop.h not found in CUDAToolkit includes.")
endif()

# Default build type (single-config)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

message(STATUS "Otter Configs:")
get_target_property(_OTTER_LINK_LANG mandelbrot_otterdream LINKER_LANGUAGE)
message(STATUS "Otter Linker language: ${_OTTER_LINK_LANG}")
