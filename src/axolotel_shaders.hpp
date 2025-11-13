///// Otter: Axolotel-HUD shaders - fullscreen triangle VS + additive glow FS (rings + gill-petals).
///// Schneefuchs: GLSL 430 core; uniform-only (no textures); monotonic clamps; gamma-aware; ASCII identifiers.
///// Maus: Pulses: vec4(x,y,t0,type), units normalized; FS sums soft SDF arcs+rings; idle breath on alpha.
///// Datei: src/axolotel_shaders.hpp

#pragma once

namespace AxolotelShaders
{
    // Fullscreen triangle via gl_VertexID (no VBO).
    static constexpr const char* kVertex = R"(#version 430 core
    out vec2 vUV;
    void main(){
        // IDs 0,1,2 -> one big triangle covering the viewport
        vUV = vec2( (gl_VertexID == 2) ? 2.0 : 0.0,
                    (gl_VertexID == 1) ? 2.0 : 0.0);
        vec2 ndc = vUV * 2.0 - 1.0;   // uv in [0,1] -> NDC [-1,1]
        gl_Position = vec4(ndc, 0.0, 1.0);
    })";

    static constexpr const char* kFragment = R"(#version 430 core
    layout (location=0) out vec4 oColor;
    in vec2 vUV;

    // Viewport + time
    uniform vec2  uViewport;     // (w,h) — reserved for future use
    uniform float uTime;         // seconds

    // Behavior
    uniform float uPulseMs;      // duration of a pulse in milliseconds
    uniform float uBreathAmp;    // 0..1 alpha modulation
    uniform float uBreathHz;     // Hz (cycles per second)
    uniform int   uPulseCount;   // 0..MAX_PULSES

    // Colors per group
    uniform vec3 uColorNav;
    uniform vec3 uColorOverlay;
    uniform vec3 uColorSystem;
    uniform vec3 uColorOther;

    // Pulses: (xNorm, yNorm, t0Sec, type)
    // x/y in [0,1] normalized to viewport origin at bottom-left; type ∈ {0,1,2,3}
    const int MAX_PULSES = 16;
    uniform vec4 uPulses[MAX_PULSES];

    float ringSDF(vec2 p, vec2 c, float r){ return abs(length(p-c) - r); }

    float smoothPulse(float x, float w){
        float a = clamp(1.0 - (x/w), 0.0, 1.0);
        return a*a*(3.0 - 2.0*a);
    }

    float petalsMask(vec2 p, vec2 c, float innerR, float outerR, float petals, float phase){
        vec2 d = p - c;
        float r = length(d);
        if(r < innerR || r > outerR) return 0.0;
        float ang = atan(d.y, d.x) + phase;
        float seg = fract(ang / (3.14159265 / petals));
        float m   = 1.0 - abs(seg - 0.5)*2.0; // triangle wave 0..1
        return m;
    }

    vec3 groupColor(int t){
        if(t == 0) return uColorNav;
        if(t == 1) return uColorOverlay;
        if(t == 2) return uColorSystem;
        return uColorOther;
    }

    void main(){
        vec2 p  = clamp(vUV, 0.0, 1.0);

        // Idle breathing alpha (baseline shimmer)
        float breath = 0.5 + 0.5 * sin(6.2831853 * uBreathHz * uTime);
        float idleAlpha = clamp(uBreathAmp * breath, 0.0, 1.0);

        vec3  accum  = vec3(0.0);
        float accumA = idleAlpha * 0.35;

        // Additive glow from pulses (expanding ring + gill petals)
        for(int i=0;i<uPulseCount;i++){
            vec4 pl = uPulses[i];
            vec2  c  = clamp(pl.xy, 0.0, 1.0);
            float t0 = pl.z;
            int   ty = int(pl.w + 0.5);

            float dt   = max(uTime - t0, 0.0);
            float life = clamp(dt * 1000.0 / max(uPulseMs, 1.0), 0.0, 1.0);
            float fade = pow(1.0 - life, 1.6);

            float r  = mix(0.015, 0.45, life);
            float d  = ringSDF(p, c, r);
            float d2 = ringSDF(p, c, r*0.66);
            float d3 = ringSDF(p, c, r*1.33);

            float g  = smoothPulse(d, 0.006) * 0.90
                     + smoothPulse(d2,0.008) * 0.60
                     + smoothPulse(d3,0.010) * 0.35;

            float pmask  = petalsMask(p, c, r*0.65, r*1.05, 6.0, uTime*0.7 + float(ty)*0.7) * 0.75;

            vec3  col = groupColor(ty);
            float a   = fade * (g + pmask);

            accum  += col * a;
            accumA += a * 0.5;
        }

        // Mild gamma shaping for a creamy glow
        vec3  color = pow(max(accum, vec3(0.0)), vec3(0.5));
        float alpha = clamp(accumA, 0.0, 1.0);

        oColor = vec4(color, alpha);
    })";
}
