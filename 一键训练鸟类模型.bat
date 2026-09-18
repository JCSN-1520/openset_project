@echo off
chcp 65001 >nul
cd /d "%~dp0"
call D:\anaconda\Scripts\activate.bat openset
python train.py --dataset dataset1 --device gpu
pause
