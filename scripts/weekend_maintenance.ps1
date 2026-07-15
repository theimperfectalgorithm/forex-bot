# Weekend maintenance for the VPS (run Saturdays -- market closed).
# MT5's LiveUpdate installs on terminal launch; restarting the terminals
# weekly while the market is closed means the modal "update ready"
# dialog never appears mid-week to block the terminal's message loop
# (observed 2026-07-12: popup stalled the API during trading hours).
#
# Order matters: stop bot(s) FIRST (a bot whose terminal dies mid-run
# could re-attach to the wrong terminal via a bare initialize()), then
# restart terminals, then start bots (each re-binds to its own terminal
# from config at startup).
#
# EDIT THESE PATHS for your VPS, then schedule with (run once, as admin):
#   schtasks /Create /TN "ForexBot Weekend Maintenance" /SC WEEKLY /D SAT `
#     /ST 12:00 /RL HIGHEST /TR "powershell -ExecutionPolicy Bypass -File C:\path\to\forex-bot\scripts\weekend_maintenance.ps1"

$bots = @(
    @{ Repo = 'C:\path\to\forex-bot'      ; Terminal = 'C:\Program Files\MetaTrader 5\terminal64.exe' }
    # uncomment when the 5ers instance goes live:
    # ,@{ Repo = 'C:\path\to\forex-bot-5ers'; Terminal = 'C:\MT5-5ers\terminal64.exe' }
)
$python = 'python'   # or the full path to the venv's python.exe

# 1. stop all bot processes (match on main_agent.py in the command line)
Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" |
    Where-Object { $_.CommandLine -match 'main_agent\.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Seconds 5

# 2. stop all MT5 terminals (updates install on next launch)
Get-Process terminal64 -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 10

# 3. relaunch terminals and give them time to update + auto-login
foreach ($b in $bots) { Start-Process $b.Terminal }
Start-Sleep -Seconds 120

# 4. restart the bots (each re-binds to its own terminal at startup)
foreach ($b in $bots) {
    Start-Process $python -ArgumentList 'src\agents\main_agent.py' `
        -WorkingDirectory $b.Repo -WindowStyle Minimized
}
