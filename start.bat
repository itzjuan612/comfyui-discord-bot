@echo off
setlocal

:: Move to the folder containing this script
cd /d %~dp0

:: Install dependencies if not already installed
python -c "import discord, aiohttp, yaml, onnxruntime" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    python -m pip install -r requirements.txt
)

echo Starting ComfyUI Discord bot...
python main.py --allow-cors

endlocal
