@echo off
setlocal

REM 修改为实际项目目录
set PROJECT_DIR=D:\search_similar_style

REM 修改为实际虚拟环境 Python 路径
set PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe

REM 配置文件
set SEARCH_CONFIG=config/search_config.10k_12g_candidates.json

REM 线程与 tokenizer 配置
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false

REM 第三方 API Key
set SILICONFLOW_API_KEY=sk

cd /d %PROJECT_DIR%

if not exist logs mkdir logs

%PYTHON_EXE% -m uvicorn src.api_server:app --host 127.0.0.1 --port 8000

endlocal
