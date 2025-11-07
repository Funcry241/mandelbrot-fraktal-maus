///// Otter: Axolotel-HUD — center-bottom anchor; additive glow + energy tap for Zoom-Coupler.
///// Schneefuchs: Deterministic clamps; ASCII logs; GL 4.3 core path; no hot-path allocs; stable shader limit.
///// Maus: activityEnergy() with group weights (nav/overlay/system/other) and smooth fade; WOW + function.
///// Datei: src/axolotel_hud.cpp

#include "pch.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "luchs_log_host.hpp"
#include "axolotel_hud.hpp"
#include "axolotel_shaders.hpp"
#include "settings_axolotel.hpp"
#include "settings.hpp"

#include <array>
#include <algorithm>
#include <cmath>

namespace AxolotelHUD {

namespace {
    // Keep C++ side in sync with shader MAX_PULSES
    constexpr int kShaderMaxPulses = 16;

    // Group weights for the energy coupler (nav strongest)
    constexpr float kWNav     = 1.00f;
    constexpr float kWOverlay = 0.80f;
    constexpr float kWSystem  = 0.50f;
    constexpr float kWOther   = 0.60f;

    // Normalization so ~3 weighted fresh pulses ≈ E ~ 1.0
    constexpr float kEnergyNorm = 3.0f;

    struct Pulse { float x, y, t0, type; }; // normalized coords, seconds, group
    static std::array<Pulse, kShaderMaxPulses> sPulses{};
    static int   sCount    = 0;     // current live pulses
    static bool  sEnabled  = Settings::Axolotel::enabled;

    // GL objects + uniform locations
    struct GLState {
        GLuint prog = 0;
        GLuint vao  = 0;
        // uniforms
        GLint uViewport=-1, uTime=-1, uPulseMs=-1, uBreathAmp=-1, uBreathHz=-1, uPulseCount=-1, uPulses=-1;
        GLint uColorNav=-1, uColorOverlay=-1, uColorSystem=-1, uColorOther=-1;
    };
    static GLState g;

    // Settings cache
    static float sPulseMs   = Settings::Axolotel::pulseMs;
    static float sBreathAmp = Settings::Axolotel::breathAmp;
    static float sBreathHz  = Settings::Axolotel::breathHz;
    static int   sMaxPulses = std::clamp(Settings::Axolotel::maxPulses, 1, kShaderMaxPulses);
    // Perf-Log nur aktiv, wenn globales Perf-Logging **und** PerfLog::enabled **und** Axolotel::perfLog aktiv ist.
    static bool  sPerfLog   = (Settings::performanceLogging && Settings::PerfLog::enabled && Settings::Axolotel::perfLog);

    static float sColorNav[3]     = { Settings::Axolotel::colorNav[0],     Settings::Axolotel::colorNav[1],     Settings::Axolotel::colorNav[2]     };
    static float sColorOverlay[3] = { Settings::Axolotel::colorOverlay[0], Settings::Axolotel::colorOverlay[1], Settings::Axolotel::colorOverlay[2] };
    static float sColorSystem[3]  = { Settings::Axolotel::colorSystem[0],  Settings::Axolotel::colorSystem[1],  Settings::Axolotel::colorSystem[2]  };
    static float sColorOther[3]   = { Settings::Axolotel::colorOther[0],   Settings::Axolotel::colorOther[1],   Settings::Axolotel::colorOther[2]   };

    // --- GL helpers ---------------------------------------------------------
    GLuint compile(GLenum type, const char* src){
        GLuint sh = glCreateShader(type);
        glShaderSource(sh, 1, &src, nullptr);
        glCompileShader(sh);
        GLint ok = GL_FALSE; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if(!ok){
            char log[2048]; GLsizei n=0; glGetShaderInfoLog(sh, 2047, &n, log);
            log[std::min<int>(n,2047)] = '\0';
            LUCHS_LOG_HOST("[AXO][ERR] shader compile failed (%u): %s", (unsigned)type, log);
            glDeleteShader(sh);
            return 0;
        }
        return sh;
    }

    GLuint link(GLuint vs, GLuint fs){
        GLuint p = glCreateProgram();
        glAttachShader(p, vs);
        glAttachShader(p, fs);
        glLinkProgram(p);
        GLint ok = GL_FALSE; glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if(!ok){
            char log[2048]; GLsizei n=0; glGetProgramInfoLog(p, 2047, &n, log);
            log[std::min<int>(n,2047)] = '\0';
            LUCHS_LOG_HOST("[AXO][ERR] program link failed: %s", log);
            glDeleteProgram(p);
            return 0;
        }
        glDetachShader(p, vs); glDetachShader(p, fs);
        glDeleteShader(vs); glDeleteShader(fs);
        return p;
    }

    void ensureGL(){
        if (g.prog) return;
        GLuint vs = compile(GL_VERTEX_SHADER,   AxolotelShaders::kVertex);
        GLuint fs = compile(GL_FRAGMENT_SHADER, AxolotelShaders::kFragment);
        if (!vs || !fs){ if(vs) glDeleteShader(vs); if(fs) glDeleteShader(fs); return; }
        g.prog = link(vs, fs);
        if (!g.prog) return;

        glGenVertexArrays(1, &g.vao);

        g.uViewport   = glGetUniformLocation(g.prog, "uViewport");
        g.uTime       = glGetUniformLocation(g.prog, "uTime");
        g.uPulseMs    = glGetUniformLocation(g.prog, "uPulseMs");
        g.uBreathAmp  = glGetUniformLocation(g.prog, "uBreathAmp");
        g.uBreathHz   = glGetUniformLocation(g.prog, "uBreathHz");
        g.uPulseCount = glGetUniformLocation(g.prog, "uPulseCount");
        g.uPulses     = glGetUniformLocation(g.prog, "uPulses");

        g.uColorNav     = glGetUniformLocation(g.prog, "uColorNav");
        g.uColorOverlay = glGetUniformLocation(g.prog, "uColorOverlay");
        g.uColorSystem  = glGetUniformLocation(g.prog, "uColorSystem");
        g.uColorOther   = glGetUniformLocation(g.prog, "uColorOther");

        if (sPerfLog) {
            LUCHS_LOG_HOST("[AXO] init ok (prog=%u vao=%u)", (unsigned)g.prog, (unsigned)g.vao);
        }
    }

    // Anchor normalized (x,y) from Settings + viewport (for pixel margin)
    inline std::pair<float,float> anchorNorm(int /*vpW*/, int vpH){
        const float ax = std::clamp(Settings::Axolotel::anchorFracX, 0.0f, 1.0f);
        const float fracBottom = std::clamp(Settings::Axolotel::anchorFracBottom, 0.0f, 0.5f);
        const float pixFrac    = (vpH > 0) ? (Settings::Axolotel::marginBottomPx / (float)vpH) : Settings::Axolotel::anchorFracBottom;
        const float ay = std::max(fracBottom, pixFrac);
        return { ax, ay };
    }

    // Ring-buffer push: store with placeholder coords; draw() overwrites to anchor
    inline void pushPulseNow(int type, float nowSec){
        const Pulse p{ 0.5f, 0.5f, nowSec, float(type) }; // x/y ignored later
        if (sCount < sMaxPulses) {
            sPulses[sCount++] = p;
        } else {
            // drop oldest
            for (int i=1;i<sMaxPulses;i++) sPulses[i-1] = sPulses[i];
            sPulses[sMaxPulses-1] = p;
        }
    }

    // Classify keys into groups for color tinting
    inline int classifyKey(int key, int mods){
        (void)mods;
        if(key==262||key==263||key==264||key==265) return 0;      // arrows
        if(key=='W'||key=='A'||key=='S'||key=='D') return 0;      // WASD
        if(key=='H'||key=='O'||key=='P') return 1;                // overlay toggles
        if(key==256 /*ESC*/ || (key>=290 && key<=314)) return 2;  // Esc/F-keys
        return 3;
    }

    inline float groupWeight(int ty){
        switch(ty){
            case 0: return kWNav;
            case 1: return kWOverlay;
            case 2: return kWSystem;
            default: return kWOther;
        }
    }

} // anon

// --- Public API ---------------------------------------------------------------

void init(){
    ensureGL();
}

void shutdown(){
    if (g.prog) { glDeleteProgram(g.prog); g.prog = 0; }
    if (g.vao)  { glDeleteVertexArrays(1, &g.vao); g.vao = 0; }
    sCount = 0;
    sEnabled = false; // ensure disabled state after teardown
}

void setEnabled(bool enabled){ sEnabled = enabled; }
bool isEnabled(){ return sEnabled; }

void noteKeyPress(int key, int mods){
    if (!sEnabled) return;
    ensureGL();
    const int type = classifyKey(key, mods);
    // Stamp time; position gets anchored during draw()
    pushPulseNow(type, static_cast<float>(glfwGetTime()));
    if (sPerfLog) {
        LUCHS_LOG_HOST("[AXO] key=%d mods=%d type=%d pulses=%d/%d", key, mods, type, sCount, sMaxPulses);
    }
}

void draw(int viewportWidth, int viewportHeight, double timeSeconds){
    if (!sEnabled) return;
    ensureGL();
    if (!g.prog || !g.vao) return;

    // Cull expired pulses by lifetime
    const float tNow = static_cast<float>(timeSeconds);
    const float lifeMax = 1.0f;
    int w = 0;
    for (int r=0; r<sCount; r++){
        const float life = (tNow - sPulses[r].t0) * 1000.0f / std::max(sPulseMs, 1.0f);
        if(life <= lifeMax){
            if (w!=r) sPulses[w] = sPulses[r];
            ++w;
        }
    }
    sCount = w;

    // Overwrite all pulse positions with the *center-bottom* anchor.
    const auto [ax, ay] = anchorNorm(viewportWidth, viewportHeight);
    std::array<Pulse, kShaderMaxPulses> upload = sPulses;
    for (int i=0; i<sCount; ++i) { upload[i].x = ax; upload[i].y = ay; }

    // Save & set blend state (additive for creamy glow)
    GLboolean wasBlend = glIsEnabled(GL_BLEND);
    GLint oldSrcRGB=0, oldDstRGB=0, oldSrcA=0, oldDstA=0;
    glGetIntegerv(GL_BLEND_SRC_RGB,   &oldSrcRGB);
    glGetIntegerv(GL_BLEND_DST_RGB,   &oldDstRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &oldSrcA);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &oldDstA);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    glUseProgram(g.prog);
    glBindVertexArray(g.vao);

    // Uniforms
    if (g.uViewport   >= 0) glUniform2f(g.uViewport, (float)viewportWidth, (float)viewportHeight);
    if (g.uTime       >= 0) glUniform1f(g.uTime, tNow);
    if (g.uPulseMs    >= 0) glUniform1f(g.uPulseMs, sPulseMs);
    if (g.uBreathAmp  >= 0) glUniform1f(g.uBreathAmp, std::clamp(sBreathAmp, 0.0f, 1.0f));
    if (g.uBreathHz   >= 0) glUniform1f(g.uBreathHz,  std::clamp(sBreathHz,  0.05f, 3.0f));
    if (g.uPulseCount >= 0) glUniform1i(g.uPulseCount, std::min(sCount, sMaxPulses));

    if (g.uColorNav     >= 0) glUniform3f(g.uColorNav,     sColorNav[0],     sColorNav[1],     sColorNav[2]);
    if (g.uColorOverlay >= 0) glUniform3f(g.uColorOverlay, sColorOverlay[0], sColorOverlay[1], sColorOverlay[2]);
    if (g.uColorSystem  >= 0) glUniform3f(g.uColorSystem,  sColorSystem[0],  sColorSystem[1],  sColorSystem[2]);
    if (g.uColorOther   >= 0) glUniform3f(g.uColorOther,   sColorOther[0],   sColorOther[1],   sColorOther[2]);

    if (g.uPulses >= 0 && sCount > 0) {
        glUniform4fv(g.uPulses, std::min(sCount, sMaxPulses),
                     reinterpret_cast<const GLfloat*>(upload.data()));
    }

    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindVertexArray(0);
    glUseProgram(0);

    // Restore blend state
    glBlendFuncSeparate(oldSrcRGB, oldDstRGB, oldSrcA, oldDstA);
    if (!wasBlend) glDisable(GL_BLEND);

    if (sPerfLog && sCount>0) {
        LUCHS_LOG_HOST("[AXO] draw OK (center-bottom anchor; pulses=%d)", sCount);
    }
}

void setColors(float nr,float ng,float nb,
               float or_,float og,float ob,
               float sr,float sg,float sb,
               float tr,float tg,float tb){
    sColorNav[0]=nr; sColorNav[1]=ng; sColorNav[2]=nb;
    sColorOverlay[0]=or_; sColorOverlay[1]=og; sColorOverlay[2]=ob;
    sColorSystem[0]=sr; sColorSystem[1]=sg; sColorSystem[2]=sb;
    sColorOther[0]=tr; sColorOther[1]=tg; sColorOther[2]=tb;
}

void configure(float pulseMs, float breathAmp, float breathHz, int maxPulsesClamped, bool perfLog){
    sPulseMs   = std::max(100.0f, pulseMs);
    sBreathAmp = std::clamp(breathAmp, 0.0f, 1.0f);
    sBreathHz  = std::clamp(breathHz,  0.05f, 3.0f);
    sMaxPulses = std::clamp(maxPulsesClamped, 1, kShaderMaxPulses);
    // honor runtime toggle but keep master gates
    sPerfLog   = (Settings::performanceLogging && Settings::PerfLog::enabled && perfLog);
}

// --- Energy tap ---------------------------------------------------------------

float activityEnergy(){
    if (!sEnabled || sCount == 0) return 0.0f;
    const float now = static_cast<float>(glfwGetTime());
    float sum = 0.0f;

    for (int i=0; i<sCount; ++i){
        const float dtMs = (now - sPulses[i].t0) * 1000.0f;
        const float life = dtMs / std::max(sPulseMs, 1.0f); // 0..1 window
        if (life < 0.0f || life > 1.0f) continue;

        const float fade = std::pow(1.0f - life, 1.6f); // same spirit as shader
        int ty = static_cast<int>(sPulses[i].type + 0.5f);
        if (ty < 0) ty = 0; if (ty > 3) ty = 3;

        sum += groupWeight(ty) * fade;
    }

    float E = sum / kEnergyNorm;
    if (E < 0.0f) E = 0.0f;
    if (E > 1.0f) E = 1.0f;

    if (sPerfLog && E > 0.0f) {
        LUCHS_LOG_HOST("[AXO][E] energy=%.3f sum=%.3f cnt=%d", E, sum, sCount);
    }
    return E;
}

} // namespace AxolotelHUD
