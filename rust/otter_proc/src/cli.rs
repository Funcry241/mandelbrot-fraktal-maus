///// Otter: CLI-Definition (Clap) – Full/Clean/Autogit; mapping: PS '/build' ⇒ Full, PS '/branch' ⇒ Autogit --branch wupp.
///// Schneefuchs: Defaults wie zuvor (cfg=RelWithDebInfo; Presets optional); ASCII-only; keine Logik.
///// Maus: Minimal-Signaturen, kompatibel zu bestehendem main.rs und commands::*.
///// Datei: rust/otter_proc/src/cli.rs

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "otter_proc", version = "0.1.0", about = "Otter Runner: full | clean | autogit")]
pub struct Cli { 
    /// Project root (defaults to current working directory)
    #[arg(long = "root", value_name = "PATH")]
    pub root: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Configure + Build via CMake Presets (PS '/build' → Full)
    Full {
        /// Build configuration (e.g., RelWithDebInfo, Release, Debug)
        #[arg(long = "cfg", default_value = "RelWithDebInfo")]
        cfg: String,

        /// CMake configure preset override (optional; defaulting handled in code)
        #[arg(long = "configure-preset")]
        configure_preset: Option<String>,

        /// CMake build preset override (optional; defaulting handled in code)
        #[arg(long = "build-preset")]
        build_preset: Option<String>,

        /// Parallel build jobs (passes '--parallel N' to 'cmake --build')
        #[arg(long = "parallel")]
        parallel: Option<u32>,
    },

    /// Safe cleanup of build artifacts
    Clean {
        /// Dry-run: only print what would be removed
        #[arg(long = "dry-run", default_value_t = false)]
        dry_run: bool,

        /// Hard mode: also remove Rust/third-party caches (e.g., 'target', 'vcpkg_installed')
        #[arg(long = "hard", default_value_t = false)]
        hard: bool,

        /// Additional relative paths to remove (multiple allowed)
        #[arg(long = "extra", value_name = "REL_PATH")]
        extra: Vec<PathBuf>,
    },

    /// Auto add/commit/push with SSH→HTTPS remote fallback (PS '/branch' → Autogit --branch wupp)
    Autogit {
        /// Commit message (if omitted, a generic one is used)
        #[arg(short = 'm', long = "message")]
        message: Option<String>,

        /// Allow empty commit even if there are no changes
        #[arg(long = "allow-empty", default_value_t = false)]
        allow_empty: bool,

        /// Remote name to push to
        #[arg(long = "remote", default_value = "origin")]
        remote: String,

        /// Branch to push (default: current HEAD if omitted; PS wrapper uses 'wupp')
        #[arg(long = "branch")]
        branch: Option<String>,

        /// Automatically switch SSH remote URL to HTTPS on push failure
        #[arg(long = "auto-https-fallback", default_value_t = true)]
        auto_https_fallback: bool,
    },
}
