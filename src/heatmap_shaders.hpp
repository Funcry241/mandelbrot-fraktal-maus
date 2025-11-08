///// Otter: Dezent – dünner, weicher Ring + minimaler Crosshair/Glow (subtiler Marker).
///// Schneefuchs: Texel-Center-Fix via textureSize(); Panel→UV sauber; /WX-safe, ASCII-only.
/// /// Maus: Keine neuen Uniforms; nur Ring-Formel weicher + dünner; Alpha wird extern gesetzt.
/// /// Datei: src/heatmap_shaders.hpp

#pragma once

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

// Marker-Uniforms (unverändert)
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
  // Panel-Pixel -> [0,1], Flip Y
  vec2 sizePx = uContentRectPx.zw - uContentRectPx.xy;
  vec2 uv = (vPx - uContentRectPx.xy) / sizePx;
  uv.y = 1.0 - uv.y;

  // Texel-Center Recenter (Half-Texel in X/Y)
  vec2 texDim = vec2(textureSize(uGrid, 0));
  uv = ((texDim - 1.0) * uv + 0.5) / texDim;

  if(any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))){
    FragColor = vec4(0.0); return;
  }

  // Heat + Gold
  float v = texture(uGrid, uv).r;
  float a = smoothstep(0.05, 0.65, v) * uAlphaBase;
  vec4 base = vec4(mapGold(v), a);

  // ——— Subtiler Ring (dünn + weich) -----------------------------------------
  float m = 0.0;
  if(uMarkEnable > 0.5){
    vec2  d    = vPx - uMarkCenterPx;
    float r    = length(d);
    float edge = abs(r - uMarkRadiusPx);

    // Dünner, weicher Strich: Stärke ~ fwidth(edge), mit engerer Kernzone
    float aa   = max(0.5, fwidth(edge));
    float t    = max(0.6, 0.9 * aa);            // Basisdicke (dezent)
    float ring = 1.0 - smoothstep(t, t + 2.5*aa, edge);
           ring *= 0.35;                         // niedrigere Intensität

    // Sehr zartes Crosshair (optional, stark gedämpft)
    float cx   = 1.0 - smoothstep(1.2, 1.2 + 2.0*aa, abs(d.x));
    float cy   = 1.0 - smoothstep(1.2, 1.2 + 2.0*aa, abs(d.y));
    float cross= max(cx, cy) * 0.12;

    // Weicher Rand-Glow entlang des Rings (sehr schwach)
    float halo = exp(-(edge*edge) / max(1.0, (6.0*aa)*(6.0*aa))) * 0.15;

    m = uMarkAlpha * (ring + cross + halo);
  }

  // Akzentfarbe
  vec3 markCol = vec3(0.60, 1.00, 0.60);
  vec4 outCol  = mix(base, vec4(markCol, 1.0), m);
  outCol.a     = max(base.a, max(outCol.a, m));
  FragColor    = outCol;
}
)GLSL";

} // namespace HeatmapShaders
