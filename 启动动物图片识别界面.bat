@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON=C:\Users\dell\.conda\envs\openset\python.exe"
if not exist "%PYTHON%" (
  echo 未找到 openset 环境：%PYTHON%
  echo 请确认 Conda 环境路径后再运行。
  pause
  exit /b 1
)
"%PYTHON%" app.py --dataset dataset2 --device gpu
pause
