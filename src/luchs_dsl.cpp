///// OtterDream — Replikatoren
///// File: src/luchs_dsl.cpp
///// Purpose: LuchsScript Mini-DSL (Prompt->Coloring AST) – Stubs
///// Phase: 3 (Color-Replikatoren)
///// Hooks: prompt_coloring / NVRTC-Pipeline
///// Depends: pch.hpp, luchs_log_host.hpp, luchs_dsl.hpp
///// Build: /WX-safe
///// Log-Tags: [REPL/COLOR]  (Alias: [DSL])
///// Created: 2025-11-06 (Europe/Berlin)
///// Notes: Echte Parser-/Validator-Fehler später.

#include "pch.hpp"
#include "luchs_dsl.hpp"
#include "luchs_log_host.hpp"

namespace LuchsDSL {

Program compile_from_prompt(const std::string& prompt) {
    Program p;
    p.src = prompt;
    p.ok  = !prompt.empty();
    LUCHS_LOG_HOST("[REPL/COLOR] prompt->dsl ok=%d (stub)", p.ok ? 1 : 0);
    return p;
}

} // namespace LuchsDSL
