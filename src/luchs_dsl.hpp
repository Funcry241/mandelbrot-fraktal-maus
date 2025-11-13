///// OtterDream — Replikatoren
///// File: src/luchs_dsl.hpp
///// Purpose: LuchsScript Mini-DSL (Prompt->Coloring AST) - Stubs
///// Phase: 3 (Color-Replikatoren)
///// Hooks: prompt_coloring / NVRTC-Pipeline
///// Depends: <string>
///// Build: /WX-safe
///// Log-Tags: [REPL/COLOR]  (Alias: [DSL])
///// Created: 2025-11-06 (Europe/Berlin)
///// Notes: Kleines Containerobjekt; später AST-Knoten & Validator.

#pragma once
#include <string>

namespace LuchsDSL {

    struct Program {
        std::string src;   // kanonische DSL-Repräsentation
        bool        ok = false;
    };

    // Einfache Prompt->DSL-Übersetzung (Stub).
    Program compile_from_prompt(const std::string& prompt);

} // namespace LuchsDSL
