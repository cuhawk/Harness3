@echo off
set LITELLM_BASE_URL=http://localhost:8088/v1
set LITELLM_API_KEY=sk-anything
set MODEL=llama-glimmer
python "%~dp0main.py" --workspace "C:\Users\soura"
