@echo off
call "C:\Users\micha\miniconda3\Scripts\activate.bat" llamafactory
set API_PORT=8000
llamafactory-cli api --model_name_or_path "C:\Users\micha\models\qwen\Qwen2___5-1___5B-Instruct"
pause
