@echo off
call "C:\Users\micha\miniconda3\Scripts\activate.bat" "C:\Users\micha\Desktop\agent_prototype\env"
cd /d "%~dp0.."
python life_expense\life_expense.py %*
pause
