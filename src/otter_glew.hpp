#pragma once
// Einheitlicher GLEW-Wrapper für dynamisches GLEW (glew32.dll)

#if defined(GLEW_STATIC)
#  undef GLEW_STATIC   // wir nutzen DLL, nicht static
#endif

#ifndef GLEW_NO_GLU
#define GLEW_NO_GLU
#endif
#ifndef GLEW_NO_IMAGING
#define GLEW_NO_IMAGING
#endif

#include <GL/glew.h>

// Schutz: Falls woanders doch wieder GLEW_STATIC auftaucht, sofort Fehler:
#ifdef GLEW_STATIC
#  error "Remove GLEW_STATIC – build uses dynamic GLEW (glew32.dll)."
#endif
