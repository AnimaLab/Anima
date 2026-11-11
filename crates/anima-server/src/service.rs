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

// ---------- Windows (Task Scheduler) ----------
//
// The server is a console program, not a Windows service, so it is run by a
// per-user scheduled task that starts at logon. This needs no administrator
// rights and no extra software. `conhost --headless` keeps the console window
// from appearing, and a small launcher script sets the working directory
// (the config uses relative paths) and sends output to a log file.

#[cfg(target_os = "windows")]
const TASK_NAME: &str = "Anima Server";
#[cfg(target_os = "windows")]
const LAUNCHER: &str = "anima-service.cmd";
#[cfg(target_os = "windows")]
const LOG_FILE: &str = "anima-server.log";

/// Quote a value for a PowerShell single-quoted string.
#[cfg(any(target_os = "windows", test))]
fn ps_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

/// The launcher script: run from its own folder, append output to the log.
#[cfg(any(target_os = "windows", test))]
fn launcher_script(bin: &Path, config_name: &str, log_name: &str) -> String {
    format!(
        "@echo off\r\nrem Started by the \"Anima Server\" scheduled task. Created by anima-server --install.\r\ncd /d \"%~dp0\"\r\n\"{}\" \"{config_name}\" >> \"{log_name}\" 2>&1\r\n",
        bin.display()
    )
}

/// PowerShell that registers the logon task and starts it now.
#[cfg(any(target_os = "windows", test))]
fn register_task_script(task: &str, launcher: &Path, cwd: &Path) -> String {
    let argument = format!("--headless \"{}\"", launcher.display());
    format!(
        "$ErrorActionPreference = 'Stop'; \
         $action = New-ScheduledTaskAction -Execute 'conhost.exe' -Argument {arg} -WorkingDirectory {cwd}; \
         $trigger = New-ScheduledTaskTrigger -AtLogOn -User \"$env:USERDOMAIN\\$env:USERNAME\"; \
         $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries \
         -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 \
         -RestartInterval (New-TimeSpan -Minutes 1); \
         Register-ScheduledTask -TaskName {task} -Action $action -Trigger $trigger -Settings $settings \
         -Description 'Anima memory server' -Force | Out-Null; \
         Start-ScheduledTask -TaskName {task}",
        arg = ps_quote(&argument),
        cwd = ps_quote(&cwd.to_string_lossy()),
        task = ps_quote(task),
    )
}

#[cfg(target_os = "windows")]
fn powershell(script: &str) -> anyhow::Result<String> {
    let output = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .output()
        .map_err(|e| anyhow::anyhow!("cannot run PowerShell: {e}"))?;
    if !output.status.success() {
        anyhow::bail!("{}", String::from_utf8_lossy(&output.stderr).trim());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(target_os = "windows")]
fn install() -> anyhow::Result<()> {
    let bin = anima_binary()?;
    let cwd = anima_working_dir()?;
    anyhow::ensure!(
        cwd.join("config.toml").is_file(),
        "no config.toml in {}. Run --install from the folder that holds Anima's config and data.",
        cwd.display()
    );
    let launcher = cwd.join(LAUNCHER);
    std::fs::write(&launcher, launcher_script(&bin, "config.toml", LOG_FILE))?;

    powershell(&register_task_script(TASK_NAME, &launcher, &cwd)).map_err(|e| {
        anyhow::anyhow!("could not register the scheduled task \"{TASK_NAME}\": {e}")
    })?;

    println!("Anima installed. It is starting now and will start at each logon.");
    println!("  Folder: {}", cwd.display());
    println!("  Log:    {}", cwd.join(LOG_FILE).display());
    println!("  Status: anima-server --service-status");
    println!("  Remove: anima-server --uninstall");
    Ok(())
}

/// PowerShell that stops the servers started from `bin` with a config file,
/// leaving `mcp`, `recall` and other client subcommands alone.
#[cfg(target_os = "windows")]
fn stop_servers_script(bin: &Path) -> String {
    format!(
        "Get-CimInstance Win32_Process -Filter \"Name='anima-server.exe'\" | \
         Where-Object {{ $_.ExecutablePath -eq {bin} -and $_.CommandLine -like '*config.toml*' }} | \
         ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}",
        bin = ps_quote(&bin.to_string_lossy()),
    )
}

#[cfg(target_os = "windows")]
fn uninstall() -> anyhow::Result<()> {
    let bin = anima_binary()?;
    let task = ps_quote(TASK_NAME);
    let script = format!(
        "$t = Get-ScheduledTask -TaskName {task} -ErrorAction SilentlyContinue; \
         if ($t) {{ Stop-ScheduledTask -TaskName {task} -ErrorAction SilentlyContinue; \
         Unregister-ScheduledTask -TaskName {task} -Confirm:$false; 'removed' }} else {{ 'absent' }}; {stop}",
        stop = stop_servers_script(&bin),
    );
    let outcome = powershell(&script)?;
    if let Ok(cwd) = anima_working_dir() {
        let _ = std::fs::remove_file(cwd.join(LAUNCHER));
    }
    if outcome.contains("removed") {
        println!("Anima service uninstalled. Its data and config were left in place.");
    } else {
        println!("No \"{TASK_NAME}\" scheduled task was installed.");
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn status() -> anyhow::Result<()> {
    let task = ps_quote(TASK_NAME);
    let script = format!(
        "$t = Get-ScheduledTask -TaskName {task} -ErrorAction SilentlyContinue; \
         if ($t) {{ 'installed, task ' + $t.State.ToString().ToLower() }} else {{ 'not installed' }}"
    );
    println!("Anima service: {}", powershell(&script)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powershell_strings_are_quoted_safely() {
        assert_eq!(ps_quote("C:\\Users\\O'Neil\\Anima"), "'C:\\Users\\O''Neil\\Anima'");
    }

    #[test]
    fn launcher_runs_from_its_folder_and_logs() {
        let script = launcher_script(Path::new("C:\\Program Files\\Anima\\anima-server.exe"), "config.toml", "anima-server.log");
        assert!(script.contains("cd /d \"%~dp0\"\r\n"));
        assert!(script.contains("\"C:\\Program Files\\Anima\\anima-server.exe\" \"config.toml\" >> \"anima-server.log\" 2>&1"));
    }

    #[test]
    fn task_registration_hides_the_console_and_needs_no_admin() {
        let script = register_task_script(
            "Anima Server",
            Path::new("C:\\Anima\\anima-service.cmd"),
            Path::new("C:\\Anima"),
        );
        assert!(script.contains("-Execute 'conhost.exe' -Argument '--headless \"C:\\Anima\\anima-service.cmd\"'"));
        assert!(script.contains("-WorkingDirectory 'C:\\Anima'"));
        assert!(script.contains("-AtLogOn"));
        assert!(!script.to_lowercase().contains("highest"), "must not ask for elevation");
    }
}
