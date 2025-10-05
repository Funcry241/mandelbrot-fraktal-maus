///// Otter: Quiet & deterministic — Root-fix: re-center UVs to texel centers via textureSize.
///// Schneefuchs: Koordinaten sauber (Panel oben-links, Texture unten-links); keine Heuristik.
///// Maus: Einzeilige Freude; hübsches Gold; Kreuz bleibt in Panel-Pixeln (kein Extra-Flip).
///// Flip: uv.y = 1.0 - uv.y (Orientation); danach Half-Texel-Recenter in X und Y.
///// Datei: heatmap_shaders.hpp
///// Vertrag: CPU/ROI nutzen weiterhin Buffer-/Panel-Pixel; Shader übernimmt Alignment.

#pragma once
// Header-only: hält nur GLSL-Quellen zusammen.
// Kein C++-Code, keine GL-Abhängigkeit.

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

vec3 mapGold(float v){
  float g = clamp(v,0.0,1.0);
  g = smoothstep(0.0,1.0,g);
  g = pow(g, 0.90);
  return mix(vec3(0.08,0.08,0.10), vec3(0.98,0.78,0.30), g);
}

void main(){
  // Panel-Pixel -> [0,1] -> Flip Y (OpenGL-UV) -------------------------------
  vec2 sizePx = uContentRectPx.zw - uContentRectPx.xy;
  vec2 uv = (vPx - uContentRectPx.xy) / sizePx;
  uv.y = 1.0 - uv.y;

  // ROOT FIX: Vertex-zu-Texel-Zentren re-centern (half-texel Korrektur) ------
  // Wenn das Grid als N Kanten-Samples (0..N-1) erzeugt wurde, liegen die
  // texel-Zentren bei (j + 0.5)/N. Das Mapping unten sorgt dafür, dass
  // uv=0..1 genau diese Zentren trifft (und nicht die Kanten).
  vec2 texDim = vec2(textureSize(uGrid, 0));         // (W, H)
  uv = ((texDim - 1.0) * uv + 0.5) / texDim;

  // Out-of-bounds? (durch AA am Rand) ----------------------------------------
  if(any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))){
    FragColor = vec4(0.0); return;
  }

  // Heat-Sampling + Gold-Mapping ---------------------------------------------
  float v = texture(uGrid, uv).r;
  float a = smoothstep(0.05, 0.65, v) * uAlphaBase;
  vec4 base = vec4(mapGold(v), a);

  // Marker (Fadenkreuz/Ring) in Panel-Pixeln – KEIN zusätzlicher Flip --------
  float m = 0.0;
  if(uMarkEnable > 0.5){
    vec2  d2   = vPx - uMarkCenterPx;
    float r    = length(d2);
    float edge = abs(r - uMarkRadiusPx);
    float aa   = fwidth(edge) + 0.75;
    float ring = 1.0 - smoothstep(1.5, 1.5+aa, edge);
    float cx   = 1.0 - smoothstep(0.6, 0.6+aa, abs(d2.x));
    float cy   = 1.0 - smoothstep(0.6, 0.6+aa, abs(d2.y));
    m = max(ring, max(cx, cy)) * uMarkAlpha;
  }

  vec3 markCol = vec3(0.60, 1.00, 0.60);
  vec4 outCol  = mix(base, vec4(markCol, 1.0), m);
  outCol.a     = max(base.a, max(outCol.a, m));
  FragColor    = outCol;
}
)GLSL";

} // namespace HeatmapShaders
