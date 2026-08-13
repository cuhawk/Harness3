@echo off
set LITELLM_BASE_URL=http://localhost:4000/v1
set LITELLM_API_KEY=sk-master-change-me
set MODEL=mistral-large
python "%~dp0main.py" --workspace "C:\Users\soura"
