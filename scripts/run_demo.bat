@echo off
REM Settings reads .env as data; inherited environment takes precedence.
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. See docs/runtime-and-migration.md
  exit /b 1
)
".venv\Scripts\python.exe" -m scripts.launch
exit /b %errorlevel%
