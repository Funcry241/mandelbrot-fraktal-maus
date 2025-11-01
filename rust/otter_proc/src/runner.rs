// Otter: Prozessstart & Streaming-Logs (ohne externe Crates); stdout pro Zeile getaggt.
// Schneefuchs: Fehlercodes sauber weiterreichen; CWD optional; ENV-Overlay möglich.
// Maus: [INFO]/[ERR]/[WARN] präfixe, ASCII-only, deterministisch.
// Datei: rust/otter_proc/src/runner.rs

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};

pub fn out_info(source: &str, msg: &str) { println!("[INFO]  [{}] {}", source, msg); }
pub fn out_warn(source: &str, msg: &str) { println!("[WARN]  [{}] {}", source, msg); }
pub fn out_err (source: &str, msg: &str) { eprintln!("[ERR]   [{}] {}", source, msg); }

#[derive(Default)]
pub struct RunResult { pub code: i32 }

pub fn run_streamed_with_env(exe: &str, args: &[String], env_overlay: Option<&HashMap<String,String>>, cwd: Option<&Path>) -> RunResult {
    let mut cmd = Command::new(exe);
    cmd.args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(d) = cwd { cmd.current_dir(d); }
    if let Some(envmap) = env_overlay {
        for (k,v) in envmap.iter() { cmd.env(k, v); }
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            out_err("RUN", &format!("spawn failed exe={} err={}", exe, e));
            return RunResult { code: 1 };
        }
    };

    // stdout
    let mut out_reader = BufReader::new(child.stdout.take().unwrap());
    let mut err_reader = BufReader::new(child.stderr.take().unwrap());

    // pump both (simple interleave)
    let mut out_buf = String::new();
    let mut err_buf = String::new();
    loop {
        let mut progressed = false;

        out_buf.clear();
        if let Ok(n) = out_reader.read_line(&mut out_buf) {
            if n > 0 {
                print!("[INFO]  [CMAKE] {}", out_buf); // generischer Tag; Aufrufer plant CMAKE/NINJA-Phasen
                progressed = true;
            }
        }
        err_buf.clear();
        if let Ok(n) = err_reader.read_line(&mut err_buf) {
            if n > 0 {
                eprint!("[ERR]   [CMAKE] {}", err_buf);
                progressed = true;
            }
        }
        if !progressed {
            match child.try_wait() {
                Ok(Some(st)) => {
                    return RunResult { code: st.code().unwrap_or(1) };
                }
                Ok(None) => { std::thread::sleep(std::time::Duration::from_millis(5)); }
                Err(_) => { return RunResult { code: 1 }; }
            }
        }
    }
}

// Beibehaltener Name für Abwärtskompatibilität (falls extern benutzt)
pub fn run_streamed(exe: &str, args: &[String]) -> RunResult {
    run_streamed_with_env(exe, args, None, None)
}
