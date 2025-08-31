///// Otter: Nacktmull-Host-Typen – Mehrpräzision nur für CPU; flexible Stellenzahl per Makro.
///// Schneefuchs: Host-only Guard (#error in .cu/.cuh); ASCII-only; deterministisch.
///// Maus: Keine Device-Dependencies; Header/Source synchron; API minimal (real, cplx).
///// Datei: src/nacktmull_types.hpp

#pragma once

// Host-only: niemals in .cu/.cuh inkludieren!
#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDACC_RTC__)
#error "nm:: multiprecision types are HOST-ONLY. Do not include this header from .cu/.cuh files."
#endif

#include <boost/multiprecision/cpp_dec_float.hpp>

namespace nm {

// Stellzahl konfigurierbar: per -DNM_DEC_DIGITS10=<N> (Default: 100)
#ifndef NM_DEC_DIGITS10
#define NM_DEC_DIGITS10 100
#endif

using real = boost::multiprecision::number<
    boost::multiprecision::cpp_dec_float<NM_DEC_DIGITS10>
>; // z.B. 100 Dezimalstellen (später anhebbar)

// Minimaler komplexer Typ für Host-Seite
struct cplx {
    real re{}, im{};
};

} // namespace nm
