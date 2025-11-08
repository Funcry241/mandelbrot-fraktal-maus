///// Otter: Quiet & deterministic – Texel-Center-Recenter via textureSize() (ROOT-FIX).
///// Schneefuchs: Saubere Koordinaten (Panel oben-links, Texture unten-links); keine Heuristik.
///// Maus: Stil C = „Corner-Holo“ (dezent): vier kurze Bogen-Segmente + sanfter Glow, kein Vollring.
///// Flip: uv.y = 1.0 - uv.y; danach Half-Texel-Recenter in X und Y.
///// Datei: src/heatmap_shaders.hpp
///// Vertrag: CPU/ROI bleiben in Panel-Pixeln; Shader übernimmt Texel-Alignment.

#pragma once
// Header-only: enthält nur GLSL-Quellen.
// Kein C++-Code, keine GL-Includes.

namespace HeatmapShaders {

inline constexpr const char* PanelVS = R"GLSL(#version 430 core
layout(location=0) in vec2 aPosPx; layout(location=1) in vec3 aColor;
uniform vec2 uViewportPx; out vec3 vColor; out vec2 vPx;
vec2 toNdc(vec2 p, vec2 vp){ return vec2(p.x/vp.x*2.0-1.0, 1.0-p.y/vp.y*2.0); }
void main(){ vColor=aColor; vPx=aPosPx; gl_Position=vec4(toNdc(aPosPx,uViewportPx),0.0,1.0); }
)GLSL";

inline constexpr const char* PanelFS = R"GLSL(#version 430 core
in vec3 vColor; in vec2 vPx; out vec4 FragColor;
uniform vec4 uPanelRectPx; uniform float uRadiusPx,uAlpha,uBorderPx;
float sdRoundRect(vec2 p, vec2 b, float r){
  vec2 d = abs(p) - b + vec2(r);
  return length(max(d,0.0)) - r;
}
void main(){
  vec2 c = 0.5 * (uPanelRectPx.xy + uPanelRectPx.zw);
  vec2 b = 0.5 * (uPanelRectPx.zw - uPanelRectPx.xy);
  float d = sdRoundRect(vPx - c, b, uRadiusPx);
  float aa   = fwidth(d);
  float body = 1.0 - smoothstep(0.0, aa, max(d, 0.0));
  float inner = smoothstep(-uBorderPx*0.5, 0.0, d);
  vec3 borderCol = vec3(1.0, 0.82, 0.32);
  vec3 col = mix(vColor, borderCol, 0.08 * inner);
  FragColor = vec4(col, uAlpha * body);
}
)GLSL";

inline constexpr const char* HeatVS = R"GLSL(#version 430 core
layout(location=0) in vec2 aPosPx;
uniform vec2 uViewportPx; out vec2 vPx;
vec2 toNdc(vec2 p, vec2 vp){ return vec2(p.x/vp.x*2.0-1.0, 1.0-p.y/vp.y*2.0); }
void main(){ vPx=aPosPx; gl_Position=vec4(toNdc(aPosPx,uViewportPx),0.0,1.0); }
)GLSL";

inline constexpr const char* HeatFS = R"GLSL(#version 430 core
in vec2 vPx; out vec4 FragColor;
uniform vec4   uContentRectPx;
uniform sampler2D uGrid;
uniform float  uAlphaBase;

uniform float  uMarkEnable;
uniform vec2   uMarkCenterPx;
uniform float  uMarkRadiusPx;
uniform float  uMarkAlpha;

// Stil-Uniforms
uniform int    uHStyle;       // 2 = Corner-Holo (dieser Build)
uniform float  uHTime;        // Sekunden
uniform float  uHStrokePx;    // Strichstärke am Bogen
uniform float  uHGlowPx;      // Glow-Breite
uniform float  uHArcSpanDeg;  // Spannweite je Segment in Grad

vec3 mapGold(float v){
  float g = clamp(v,0.0,1.0);
  g = smoothstep(0.0,1.0,g);
  g = pow(g, 0.90);
  return mix(vec3(0.08,0.08,0.10), vec3(0.98,0.78,0.30), g);
}

float arcMaskDeg(float thetaDeg, float centerDeg, float spanDeg){
  // kleinste Winkel-Differenz (0..180)
  float diff = abs(((thetaDeg - centerDeg + 540.0) - 360.0));
  float halfSpan = max(4.0, 0.5*spanDeg);
  // weiche Flanken für dezente Kanten
  return smoothstep(halfSpan+8.0, halfSpan, diff);
}

void main(){
  // Panel-Pixel -> [0,1] -> Flip Y (OpenGL-UV) -------------------------------
  vec2 sizePx = uContentRectPx.zw - uContentRectPx.xy;
  vec2 uv = (vPx - uContentRectPx.xy) / sizePx;
  uv.y = 1.0 - uv.y;

  // Texel-Center-Alignment (Half-Texel Recenter) -----------------------------
  vec2 texDim = vec2(textureSize(uGrid, 0)); // (W, H)
  uv = ((texDim - 1.0) * uv + 0.5) / texDim;

  if(any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))){
    FragColor = vec4(0.0); return;
  }

  // Heat + Gold
  float v = texture(uGrid, uv).r;
  float a = smoothstep(0.05, 0.65, v) * uAlphaBase;
  vec4 base = vec4(mapGold(v), a);

  // Marker: Stil C – Corner-Holo (vier kurze Bogen-Segmente) -----------------
  float m = 0.0;
  if(uMarkEnable > 0.5){
    vec2  d2   = vPx - uMarkCenterPx;
    float r    = length(d2);
    float edge = abs(r - uMarkRadiusPx);
    float aa   = fwidth(edge) + 0.75;

    // Grund-Stroke + sanfter Glow am Radius
    float stroke = 1.0 - smoothstep(uHStrokePx+aa, uHStrokePx, edge);
    float glow   = 1.0 - smoothstep(uHStrokePx+2.0, uHStrokePx + max(4.0, uHGlowPx), edge);

    // Winkel in Grad [0,360)
    float theta = degrees(atan(d2.y, d2.x));
    if(theta < 0.0) theta += 360.0;

    // Corner-Bögen bei 45/135/225/315°
    float arc =
        max(arcMaskDeg(theta,  45.0, uHArcSpanDeg),
        max(arcMaskDeg(theta, 135.0, uHArcSpanDeg),
        max(arcMaskDeg(theta, 225.0, uHArcSpanDeg),
            arcMaskDeg(theta, 315.0, uHArcSpanDeg))));

    // Leichtes „Breathing“ für Futurismus
    float breath = 0.88 + 0.12 * sin(uHTime * 2.4);

    m = (breath * stroke + 0.30 * glow) * arc * uMarkAlpha;
  }

  // dezentes Cyan für HUD-Akzent
  vec3 markCol = vec3(0.55, 0.95, 1.00);
  vec4 outCol  = mix(base, vec4(markCol, 1.0), m);
  outCol.a     = max(base.a, max(outCol.a, m));
  FragColor    = outCol;
}
)GLSL";

} // namespace HeatmapShaders
