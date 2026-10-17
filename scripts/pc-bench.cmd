@echo off
rem Benchmarks of record on the PC. No admin needed. Usage: C:\lwf-bench\scripts\pc-bench.cmd
setlocal
set "NODEDIR=%LOCALAPPDATA%\node-v22.22.3-win-x64"
if not exist "%NODEDIR%\node.exe" (
  echo Downloading portable Node 22.22.3 ...
  curl -fsSL -o "%LOCALAPPDATA%\node22.zip" https://nodejs.org/dist/v22.22.3/node-v22.22.3-win-x64.zip || exit /b 1
  tar -xf "%LOCALAPPDATA%\node22.zip" -C "%LOCALAPPDATA%" || exit /b 1
  del "%LOCALAPPDATA%\node22.zip"
)
set "PATH=%NODEDIR%;%PATH%"
cd /d "%~dp0.."
call npm ci --no-audit --no-fund || exit /b 1
call npm test || exit /b 1
call npm run bench
