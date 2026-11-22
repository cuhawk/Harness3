$env:LITELLM_BASE_URL = "http://127.0.0.1:8462/v1" # 8462 - qwen, 8097 - ds
$env:LITELLM_API_KEY = "sk-master-change-me"
$env:MODEL = "deepseek-flash"
python "$PSScriptRoot\main.py" --workspace "$HOME"
