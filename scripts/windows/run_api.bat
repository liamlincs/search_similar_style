@echo off
setlocal

REM 修改为实际项目目录
set PROJECT_DIR=D:\search_similar_style

REM 修改为实际虚拟环境 Python 路径
set PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe

REM 配置文件
set SEARCH_CONFIG=config/search_config.10k_12g_candidates.json

REM 标样图片目录，需与 SEARCH_CONFIG 里的 paths.standard_dir 保持一致
set STANDARD_DIR=data\standard_samples

REM 线程与 tokenizer 配置
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false

REM 第三方 API Key
set SILICONFLOW_API_KEY=sk

cd /d %PROJECT_DIR%
if errorlevel 1 (
    echo Failed to enter project directory: %PROJECT_DIR%
    pause
    exit /b 1
)

if not exist logs mkdir logs

if not exist "%STANDARD_DIR%" (
    echo Standard image directory not found: %PROJECT_DIR%\%STANDARD_DIR%
    echo Copy standard sample images into this directory, or update paths.standard_dir in:
    echo   %PROJECT_DIR%\%SEARCH_CONFIG%
    pause
    exit /b 1
)

set HAS_STANDARD_IMAGE=
for %%F in ("%STANDARD_DIR%\*.jpg" "%STANDARD_DIR%\*.jpeg" "%STANDARD_DIR%\*.png") do (
    if exist "%%~F" (
        echo %%~nxF | findstr /b /i "MY-" >nul
        if errorlevel 1 set HAS_STANDARD_IMAGE=1
    )
)
if not defined HAS_STANDARD_IMAGE (
    echo No searchable standard images found in: %PROJECT_DIR%\%STANDARD_DIR%
    echo Put .jpg/.jpeg/.png standard sample images in this folder.
    echo Note: images starting with MY- are skipped by search indexing.
    pause
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo Python executable not found: %PYTHON_EXE%
    echo Create the virtual environment first:
    echo   cd /d %PROJECT_DIR%
    echo   py -3.10 -m venv .venv
    echo   .\.venv\Scripts\activate
    echo   python -m pip install --upgrade pip
    echo   python -m pip install -r requirements.txt
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import uvicorn" >nul 2>nul
if errorlevel 1 (
    echo Missing Python dependency: uvicorn
    echo Install project dependencies into this virtual environment:
    echo   cd /d %PROJECT_DIR%
    echo   "%PYTHON_EXE%" -m pip install --upgrade pip
    echo   "%PYTHON_EXE%" -m pip install -r requirements.txt
    pause
    exit /b 1
)

%PYTHON_EXE% -m uvicorn src.api_server:app --host 127.0.0.1 --port 8000

endlocal
