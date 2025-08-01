// Datei: src/bear_CudaPBOResource.hpp
// 🐻 Bär: RAII-Wrapper für CUDA-PBO-Registrierung – automatisches Aufräumen im Destruktor.
// 🐻 Bär: Dieses Projekt steht unter dem Codenamen "Bär".

#pragma once
#ifndef BEAR_CUDA_PBO_RESOURCE_HPP
#define BEAR_CUDA_PBO_RESOURCE_HPP

#include "renderer_resources.hpp"
#include <cuda_gl_interop.h>

namespace CudaInterop {

// 🐻 Bär: Verwaltet das Lifetime des CUDA-GL-Interop-Resource-Handles
class bear_CudaPBOResource {
public:
    // 🐻 Bär: Konstruktor registriert den PBO als CUDA-Resource
    explicit bear_CudaPBOResource(GLuint pboId);
    
    // 🐻 Bär: Destruktor deregistriert automatisch die CUDA-Resource
    ~bear_CudaPBOResource();

    // 🐻 Bär: Liefert das CUDA-Resource-Handle für Mapping/Unmapping
    cudaGraphicsResource_t get() const noexcept;

    // 🐻 Bär: Nicht kopierbar, um doppelte Registrierung zu vermeiden
    bear_CudaPBOResource(const bear_CudaPBOResource&) = delete;
    bear_CudaPBOResource& operator=(const bear_CudaPBOResource&) = delete;
    
    // 🐻 Bär: Bewegbar, überträgt Ownership des Handles
    bear_CudaPBOResource(bear_CudaPBOResource&& other) noexcept;
    bear_CudaPBOResource& operator=(bear_CudaPBOResource&& other) noexcept;

private:
    cudaGraphicsResource_t resource_{nullptr};
};

} // namespace CudaInterop

#endif // BEAR_CUDA_PBO_RESOURCE_HPP
