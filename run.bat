@echo off
chcp 65001
REM 激活conda环境并运行应用

echo Activating conda environment 'lightroom'...
call conda activate lightroom

echo Running Python Lightroom Tool...
python "%~dp0main.py"

pause
