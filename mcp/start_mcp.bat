@echo off
REM Forex Bot MCP Server -- Windows startup script.
REM Intended to run as a Task Scheduler job ("ForexBotMCP") on the VPS,
REM same pattern as the main bot's own startup job.
REM
REM Assumes the repo's venv\ (same convention as the rest of this
REM project) exists one level up from this script, at ..\venv\. If it
REM doesn't exist yet, run this once first:
REM   cd /d "%~dp0.."
REM   python -m venv venv
REM   venv\Scripts\pip install -r mcp\requirements.txt

setlocal

cd /d "%~dp0.."

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo [start_mcp] WARNING: venv\Scripts\activate.bat not found -- using system python
)

if not exist mcp\.env (
    echo [start_mcp] ERROR: mcp\.env not found. Copy mcp\.env.example to mcp\.env
    echo [start_mcp]        and set a real MCP_API_KEY before starting the server.
    exit /b 1
)

echo [start_mcp] Starting Forex Bot MCP server on port 8000 ...
python mcp\server.py

endlocal
