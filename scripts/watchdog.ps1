# Watchdog for the VPS -- checks every N minutes (see the schtasks /Create
# command below) that all 4 processes are actually alive, and restarts
# (via schtasks /run) whichever isn't.
#
# WHY THIS EXISTS (2026-08-xx): Windows Task Scheduler's own restart-on-
# failure setting (RestartCount, already configured on all 4 tasks) only
# catches the scheduler's OWN launched process exiting on its own. It does
# NOT catch the pattern that has actually caused missed trading days on
# this VPS: an operator manually stops a process (Stop-Process -Force,
# e.g. mid-deploy or while debugging) and then forgets the follow-up
# `schtasks /run`. This script closes that gap by checking real, current
# process/port state on a short interval, independent of *why* something
# is down.
#
# Paths below are hardcoded to this VPS's known, fixed layout (unlike
# weekend_maintenance.ps1, which was written before the 5ers clone
# existed and used placeholder paths) -- lives in the demo clone, but
# checks all 4 processes across both clones; only ONE instance of this
# script/task needs to run on the whole VPS.
#
# One-time setup (run once, as Administrator):
#   schtasks /Create /TN "ForexBotWatchdog" /SC MINUTE /MO 15 /RL HIGHEST /RU SYSTEM `
#     /TR "powershell -ExecutionPolicy Bypass -File C:\forex-bot\scripts\watchdog.ps1"
#
# SYSTEM is fine (and preferred -- no password to store) for the watchdog
# itself: it never touches MT5 directly, it only reads process/port state
# and calls `schtasks /run`, which starts the target task under THAT
# task's own configured Run-As-User (Administrator), not the watchdog's.

$logDir  = 'C:\forex-bot\data\logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logFile = Join-Path $logDir 'watchdog.log'

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    Add-Content -Path $logFile -Value $line
    Write-Host $line
}

# -- Bots: main_agent.py's CommandLine always includes the full repo path,
# so demo vs 5ers can be told apart directly.
$procs = Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'"

$botChecks = @(
    @{ Task = 'ForexBot';      Pattern = [regex]::Escape('C:\forex-bot\src\agents\main_agent.py') }
    @{ Task = 'ForexBot-5ers'; Pattern = [regex]::Escape('C:\forex-bot-5ers\src\agents\main_agent.py') }
)
foreach ($c in $botChecks) {
    $alive = $procs | Where-Object { $_.CommandLine -match $c.Pattern }
    if (-not $alive) {
        Log "MISSING  $($c.Task) -- restarting"
        schtasks /run /tn $c.Task | Out-Null
    }
}

# -- Dashboards: mcp\server.py is launched with a relative path (via
# start_mcp.bat), so its CommandLine can't distinguish demo from 5ers --
# check by listening port instead (8000 demo, 8001 5ers). This also
# proves the server is actually accepting connections, not just that a
# python process happens to exist.
$dashChecks = @(
    @{ Task = 'ForexBotMCP';      Port = 8000 }
    @{ Task = 'ForexBotMCP-5ers'; Port = 8001 }
)
foreach ($c in $dashChecks) {
    $listening = Get-NetTCPConnection -LocalPort $c.Port -State Listen -ErrorAction SilentlyContinue
    if (-not $listening) {
        Log "MISSING  $($c.Task) (port $($c.Port) not listening) -- restarting"
        schtasks /run /tn $c.Task | Out-Null
    }
}
