///// Otter: Header-only API für GL-Ressourcen; minimaler Surface; keine Seiteneffekte.
///// Schneefuchs: Nur Deklarationen; keine Implementierungen/inline; /WX-fest.
///// Maus: Stabile Signaturen ohne GL-Header-Abhängigkeit (unsigned int statt GLuint).
///// Datei: src/renderer_resources.hpp

#pragma once

namespace OpenGLUtils {

// Kontextlabel für Logs setzen (z. B. "init", "resize", "draw")
void         setGLResourceContext(const char* context);

// Immutable RGBA8-Textur anlegen (1 Mip-Level)
unsigned int createTexture(int width, int height);

// GL_PIXEL_UNPACK_BUFFER (PBO) anlegen, Größe = width*height*4 (RGBA8)
unsigned int createPBO(int width, int height);

// Vollflächen-Upload: Textur aus UNPACK-PBO aktualisieren
void         updateTextureFromPBO(unsigned int pbo, unsigned int tex, int width, int height);

// (peekPBO wurde vollständig entfernt)

} // namespace OpenGLUtils
