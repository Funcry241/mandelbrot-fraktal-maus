///// Otter: PBO-RAII – kompatibel zur Frame-Pipeline (No-arg map, unmapAndLog).
///// Schneefuchs: Header/Source synchron; kein verdeckter Funktionswechsel.
///// Maus: ASCII-Logs; nur LUCHS_LOG_HOST im Hostpfad.
///// Datei: src/bear_CudaPBOResource.hpp

#pragma once
#ifndef BEAR_CUDA_PBO_RESOURCE_HPP
#define BEAR_CUDA_PBO_RESOURCE_HPP

#include <cstddef>          // size_t
#include <GL/glew.h>        // GLuint
#include <cuda_gl_interop.h>
#include <vector_types.h>   // uchar4

namespace CudaInterop {

// Baer: Verwaltet das Lifetime des CUDA-GL-Interop-Resource-Handles
class bear_CudaPBOResource {
public:
    // Baer: Konstruktor registriert den PBO als CUDA-Resource
    explicit bear_CudaPBOResource(GLuint pboId);
    
    // Baer: Destruktor deregistriert automatisch die CUDA-Resource
    ~bear_CudaPBOResource();

    // Baer: Liefert das CUDA-Resource-Handle fuer Mapping/Unmapping
    [[nodiscard]] cudaGraphicsResource_t get() const noexcept;

    // Baer: mappt CUDA-Resource und liefert Dev-Pointer zurueck, loggt Zustand
    [[nodiscard]] void*   mapAndLog(size_t& sizeOut);

    // Baer: Bequemer Overload: typisierter Pixelpointer (uchar4), Groesse wird intern geloggt
    [[nodiscard]] uchar4* mapAndLog();

    // Baer: unmappt CUDA-Resource
    void unmap();

    // Baer: unmappt mit Zeitmessung (PI/Perf-Logs konsistent zur Pipeline)
    void unmapAndLog();

    // Baer: Nicht kopierbar, um doppelte Registrierung zu vermeiden
    bear_CudaPBOResource(const bear_CudaPBOResource&) = delete;
    bear_CudaPBOResource& operator=(const bear_CudaPBOResource&) = delete;
    
    // Baer: Bewegbar, uebertraegt Ownership des Handles
    bear_CudaPBOResource(bear_CudaPBOResource&& other) noexcept;
    bear_CudaPBOResource& operator=(bear_CudaPBOResource&& other) noexcept;

private:
    cudaGraphicsResource_t resource_{nullptr};
};

} // namespace CudaInterop

#endif // BEAR_CUDA_PBO_RESOURCE_HPP
