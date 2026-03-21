//! Cross-platform service installation helpers.
//!
//! `anima-server --install`  — install as a system/user service (auto-start on boot)
//! `anima-server --uninstall` — remove the service
//! `anima-server --service-status` — check if installed and running

use std::path::{Path, PathBuf};

pub enum ServiceAction {
    Install,
    Uninstall,
    Status,
}

pub fn run(action: ServiceAction) -> anyhow::Result<()> {
    match action {
        ServiceAction::Install => install(),
        ServiceAction::Uninstall => uninstall(),
        ServiceAction::Status => status(),
    }
}

fn anima_binary() -> anyhow::Result<PathBuf> {
    std::env::current_exe()
        .map_err(|e| anyhow::anyhow!("cannot determine binary path: {e}"))
}

fn anima_working_dir() -> anyhow::Result<PathBuf> {
    std::env::current_dir()
        .map_err(|e| anyhow::anyhow!("cannot determine working directory: {e}"))
}

// ---------- Linux (systemd) ----------

#[cfg(target_os = "linux")]
fn install() -> anyhow::Result<()> {
    let bin = anima_binary()?;
    let cwd = anima_working_dir()?;
    let config = cwd.join("config.toml");
    let user = std::env::var("USER").unwrap_or_else(|_| "anima".into());

    let unit = format!(
        r#"[Unit]
Description=Anima Memory Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={user}
WorkingDirectory={cwd}
ExecStart={bin} {config}
Restart=on-failure
RestartSec=5
TimeoutStopSec=35
EnvironmentFile=-{cwd}/.env
StandardOutput=journal
StandardError=journal
SyslogIdentifier=anima-server

[Install]
WantedBy=multi-user.target
"#,
        user = user,
        cwd = cwd.display(),
        bin = bin.display(),
        config = config.display(),
    );

    let service_path = Path::new("/etc/systemd/system/anima-server.service");
    std::fs::write(service_path, &unit)
        .map_err(|e| anyhow::anyhow!("failed to write {}: {e} (try running with sudo)", service_path.display()))?;

    println!("Wrote {}", service_path.display());

    // Reload systemd and enable
    let cmds = [
        ("systemctl", &["daemon-reload"][..]),
        ("systemctl", &["enable", "anima-server"]),
        ("systemctl", &["start", "anima-server"]),
    ];
    for (cmd, args) in &cmds {
        let status = std::process::Command::new(cmd).args(*args).status();
        match status {
            Ok(s) if s.success() => println!("{cmd} {} — ok", args.join(" ")),
            Ok(s) => eprintln!("{cmd} {} — exit {}", args.join(" "), s),
            Err(e) => eprintln!("{cmd} {} — {e}", args.join(" ")),
        }
    }

    println!("\nAnima installed as a systemd service.");
    println!("  Logs:   journalctl -u anima-server -f");
    println!("  Stop:   sudo systemctl stop anima-server");
    println!("  Status: sudo systemctl status anima-server");
    Ok(())
}

#[cfg(target_os = "linux")]
fn uninstall() -> anyhow::Result<()> {
    let cmds = [
        ("systemctl", &["stop", "anima-server"][..]),
        ("systemctl", &["disable", "anima-server"]),
    ];
    for (cmd, args) in &cmds {
        let _ = std::process::Command::new(cmd).args(*args).status();
    }

    let service_path = Path::new("/etc/systemd/system/anima-server.service");
    if service_path.exists() {
        std::fs::remove_file(service_path)?;
        println!("Removed {}", service_path.display());
    }
    let _ = std::process::Command::new("systemctl").arg("daemon-reload").status();
    println!("Anima service uninstalled.");
    Ok(())
}

#[cfg(target_os = "linux")]
fn status() -> anyhow::Result<()> {
    let output = std::process::Command::new("systemctl")
        .args(["is-active", "anima-server"])
        .output();
    match output {
        Ok(o) => {
            let state = String::from_utf8_lossy(&o.stdout).trim().to_string();
            println!("anima-server: {state}");
        }
        Err(e) => println!("Could not query systemd: {e}"),
    }
    Ok(())
}

// ---------- macOS (launchd) ----------

#[cfg(target_os = "macos")]
fn plist_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join("Library/LaunchAgents/dev.anima.server.plist")
}

#[cfg(target_os = "macos")]
fn install() -> anyhow::Result<()> {
    let bin = anima_binary()?;
    let cwd = anima_working_dir()?;
    let config = cwd.join("config.toml");
    let log_dir = cwd.join("logs");
    std::fs::create_dir_all(&log_dir)?;

    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>dev.anima.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>{bin}</string>
        <string>{config}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{cwd}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>{log_dir}/anima-server.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/anima-server.err</string>
    <key>ThrottleInterval</key>
    <integer>5</integer>
</dict>
</plist>"#,
        bin = bin.display(),
        config = config.display(),
        cwd = cwd.display(),
        log_dir = log_dir.display(),
    );

    let dest = plist_path();
    std::fs::write(&dest, &plist)?;
    println!("Wrote {}", dest.display());

    let status = std::process::Command::new("launchctl")
        .args(["load", "-w"])
        .arg(&dest)
        .status();
    match status {
        Ok(s) if s.success() => println!("launchctl load — ok"),
        Ok(s) => eprintln!("launchctl load — exit {s}"),
        Err(e) => eprintln!("launchctl load — {e}"),
    }

    println!("\nAnima installed as a launchd agent (starts on login).");
    println!("  Logs:   tail -f {}/anima-server.log", log_dir.display());
    println!("  Stop:   launchctl unload {}", dest.display());
    println!("  Status: anima-server --service-status");
    Ok(())
}

#[cfg(target_os = "macos")]
fn uninstall() -> anyhow::Result<()> {
    let dest = plist_path();
    if dest.exists() {
        let _ = std::process::Command::new("launchctl")
            .args(["unload", "-w"])
            .arg(&dest)
            .status();
        std::fs::remove_file(&dest)?;
        println!("Removed {}", dest.display());
    }
    println!("Anima service uninstalled.");
    Ok(())
}

#[cfg(target_os = "macos")]
fn status() -> anyhow::Result<()> {
    let output = std::process::Command::new("launchctl")
        .args(["list", "dev.anima.server"])
        .output();
    match output {
        Ok(o) if o.status.success() => {
            println!("anima-server: running");
            println!("{}", String::from_utf8_lossy(&o.stdout));
        }
        _ => println!("anima-server: not installed or not running"),
    }
    Ok(())
}

// ---------- Windows (sc.exe) ----------

#[cfg(target_os = "windows")]
fn install() -> anyhow::Result<()> {
    let bin = anima_binary()?;
    let cwd = anima_working_dir()?;
    let config = cwd.join("config.toml");

    // Use sc.exe to create a service.
    // Note: sc.exe requires the binary to implement the Windows Service API,
    // which we don't. For now, use NSSM if available, otherwise fall back to
    // a scheduled task that runs at startup.
    let nssm = which_nssm();
    if let Some(nssm_path) = nssm {
        let status = std::process::Command::new(&nssm_path)
            .args([
                "install",
                "anima-server",
                &bin.to_string_lossy(),
                &config.to_string_lossy(),
            ])
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => return Err(anyhow::anyhow!("nssm install failed: exit {s}")),
            Err(e) => return Err(anyhow::anyhow!("nssm install failed: {e}")),
        }

        // Set working directory
        let _ = std::process::Command::new(&nssm_path)
            .args(["set", "anima-server", "AppDirectory", &cwd.to_string_lossy()])
            .status();

        let _ = std::process::Command::new(&nssm_path)
            .args(["start", "anima-server"])
            .status();

        println!("Anima installed as a Windows service via NSSM.");
        println!("  Stop:   nssm stop anima-server");
        println!("  Remove: nssm remove anima-server confirm");
    } else {
        // Fallback: create a scheduled task
        let task_xml = format!(
            r#"schtasks /create /tn "Anima Server" /tr "\"{}\" \"{}\"" /sc onlogon /rl highest /f"#,
            bin.display(),
            config.display(),
        );
        println!("NSSM not found. To install as a startup task, run:");
        println!("  {task_xml}");
        println!("\nOr install NSSM (https://nssm.cc) and re-run --install.");
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn which_nssm() -> Option<PathBuf> {
    std::process::Command::new("where")
        .arg("nssm")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .map(|s| PathBuf::from(s.trim()))
        })
}

#[cfg(target_os = "windows")]
fn uninstall() -> anyhow::Result<()> {
    if let Some(nssm_path) = which_nssm() {
        let _ = std::process::Command::new(&nssm_path)
            .args(["stop", "anima-server"])
            .status();
        let _ = std::process::Command::new(&nssm_path)
            .args(["remove", "anima-server", "confirm"])
            .status();
        println!("Anima service uninstalled.");
    } else {
        println!("Run: schtasks /delete /tn \"Anima Server\" /f");
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn status() -> anyhow::Result<()> {
    if let Some(nssm_path) = which_nssm() {
        let output = std::process::Command::new(&nssm_path)
            .args(["status", "anima-server"])
            .output();
        match output {
            Ok(o) => println!("{}", String::from_utf8_lossy(&o.stdout).trim()),
            Err(e) => println!("Could not query service: {e}"),
        }
    } else {
        println!("NSSM not installed — cannot query service status.");
    }
    Ok(())
}
