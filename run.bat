@echo off
chcp 65001

echo Activating conda environment 'lightroom'...
call conda activate lightroom

echo Running Python Lightroom Tool...
python "%~dp0main.py"

pause
