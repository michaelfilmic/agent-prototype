@echo off
call "C:\Users\micha\miniconda3\Scripts\activate.bat" "C:\Users\micha\Desktop\agent_prototype\env"
cd "C:\Users\micha\Desktop\agent_prototype"
python -c "from knowledge import index_documents; index_documents()"
pause
