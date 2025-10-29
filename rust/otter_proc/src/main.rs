///// Otter: Robust external process runner with line-by-line filtering and logging.
///// Schneefuchs: MSVC/CMake/Ninja/vcpkg patterns, concurrent stdout/stderr, UTF-8 lossy decode.
///// Maus: Windows-first, zero deps beyond clap (regex optional later), stable exit code.
///// Datei: rust/otter_proc/src/main.rs

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

use clap::Parser;

/// otter_proc — robust line-stream runner with filtering and logfile.
/// Usage example:
///   otter_proc --exe cmake --arg --version --cwd . --log build/configure.log --source CMAKE
#[derive(Parser, Debug)]
#[command(name = "otter_proc", version, about)]
struct Args {
    /// Executable to run (e.g., cmake, ninja, vcpkg.exe)
    #[arg(long = "exe")]
    exe: PathBuf,

    /// Arguments to pass to the executable (repeatable)
    #[arg(long = "arg")]
    arg: Vec<String>,

    /// Working directory (default: current)
    #[arg(long = "cwd")]
    cwd: Option<PathBuf>,

    /// Path to logfile (will be created/overwritten)
    #[arg(long = "log")]
    log: PathBuf,

    /// Logical source tag for prefixing output (CMAKE|NINJA|GEN|SUPPORT|VCPKG)
    #[arg(long = "source", default_value = "GEN")]
    source: String,

    /// If set, do not add [INFO]/[WARN]/[ERR] prefixes — only [SOURCE].
    #[arg(long = "no-severity-prefix")]
    no_severity_prefix: bool,

    /// If set, also mirror raw tool lines to stdout before the filtered line.
    #[arg(long = "tee-raw")]
    tee_raw: bool,
}

#[derive(Copy, Clone, Debug)]
enum Severity {
    Info,
    Warn,
    Err,
}

fn classify_line(line: &str) -> Severity {
    // Case-insensitive, robust heuristics for typical MSVC/CMake/Ninja/CUDA output
    let l = line.to_lowercase();
    let t = l.trim_start();

    // ----- errors -----
    if t.starts_with("ninja: build stopped:")
        || l.contains("fatal error")
        || l.contains("cmake error")
        || l.contains("nvcc fatal")
        || l.contains("ptxas fatal")
        || l.contains(" error c")       // cl.exe: "error Cxxxx"
        || (l.contains("lnk") && l.contains("error"))
        || (l.contains("msb") && l.contains("error"))
        || l.contains(" fehler")        // localized
        || l.contains(" error:")
    {
        return Severity::Err;
    }

    // ----- warnings -----
    if l.contains("cmake warning")
        || l.contains(" warning c")     // cl.exe: "warning Cxxxx"
        || (l.contains("lnk") && l.contains("warning"))
        || (l.contains("msb") && l.contains("warning"))
        || l.contains(" warnung")       // localized
    {
        return Severity::Warn;
    }

    Severity::Info
}

fn write_prefixed<W: Write>(
    mut out: W,
    sev: Severity,
    source: &str,
    line: &str,
    use_sev_prefix: bool,
) -> std::io::Result<()> {
    if use_sev_prefix {
        match sev {
            Severity::Info => writeln!(out, "[INFO]  [{}] {}", source, line)?,
            Severity::Warn => writeln!(out, "[WARN]  [{}] {}", source, line)?,
            Severity::Err  => writeln!(out, "[ERR]   [{}] {}", source, line)?,
        }
    } else {
        writeln!(out, "[{}] {}", source, line)?;
    }
    Ok(())
}

fn open_log(path: &PathBuf) -> std::io::Result<BufWriter<File>> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    Ok(BufWriter::new(file))
}

#[derive(Copy, Clone)]
enum StreamKind {
    Stdout,
    Stderr,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    // Prepare logfile writer
    let mut log = open_log(&args.log)?;

    // Spawn child process
    let mut cmd = Command::new(&args.exe);
    cmd.args(&args.arg);
    if let Some(cwd) = &args.cwd {
        cmd.current_dir(cwd);
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(p) => p,
        Err(e) => {
            let msg = format!("Failed to start process '{}': {}", args.exe.display(), e);
            // Emit to both console and log
            let _ = write_prefixed(
                std::io::stdout(),
                Severity::Err,
                &args.source,
                &msg,
                !args.no_severity_prefix,
            );
            writeln!(log, "{}", msg)?;
            let _ = log.flush();
            std::process::exit(101);
        }
    };

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    let (tx, rx) = mpsc::channel::<(StreamKind, String)>();

    // Dedicated pump for STDOUT
    {
        let tx = tx.clone();
        thread::spawn(move || {
            let rdr = BufReader::new(stdout);
            for line_bytes in rdr.split(b'\n') {
                match line_bytes {
                    Ok(mut bytes) => {
                        while bytes.last().copied() == Some(b'\r') { bytes.pop(); }
                        let text = String::from_utf8_lossy(&bytes).to_string();
                        let _ = tx.send((StreamKind::Stdout, text));
                    }
                    Err(_) => break,
                }
            }
        });
    }
    // Dedicated pump for STDERR
    {
        let tx = tx.clone();
        thread::spawn(move || {
            let rdr = BufReader::new(stderr);
            for line_bytes in rdr.split(b'\n') {
                match line_bytes {
                    Ok(mut bytes) => {
                        while bytes.last().copied() == Some(b'\r') { bytes.pop(); }
                        let text = String::from_utf8_lossy(&bytes).to_string();
                        let _ = tx.send((StreamKind::Stderr, text));
                    }
                    Err(_) => break,
                }
            }
        });
    }
    drop(tx);

    // Consume lines in order of arrival; classify + log + print
    while let Ok((kind, line)) = rx.recv() {
        // Mirror raw line if requested (helps debugging filters)
        if args.tee_raw && !line.trim().is_empty() {
            match kind {
                StreamKind::Stdout => println!("{}", line),
                StreamKind::Stderr => eprintln!("{}", line),
            }
        }

        // Classify & print filtered line
        if !line.trim().is_empty() {
            let sev = classify_line(&line);
            let _ = write_prefixed(
                std::io::stdout(),
                sev,
                &args.source,
                &line,
                !args.no_severity_prefix,
            );
        }

        // Always append raw to logfile (unfiltered)
        writeln!(log, "{}", line)?;
    }

    // Wait for child exit and return same code
    let status = child.wait()?;
    let _ = log.flush();

    match status.code() {
        Some(code) => std::process::exit(code),
        None => std::process::exit(102),
    }
}
