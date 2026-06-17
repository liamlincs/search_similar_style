# Windows 服务部署（NSSM + uvicorn）

## 1. 启动脚本

仓库已提供模板：

`scripts/windows/run_api.bat`

使用前修改这两个变量：

- `PROJECT_DIR`
- `PYTHON_EXE`

以及按需修改这些环境变量：

- `SEARCH_CONFIG`
- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`
- `TOKENIZERS_PARALLELISM`
- `SILICONFLOW_API_KEY`

示例：

```bat
set PROJECT_DIR=D:\search_similar_style
set PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe
set SEARCH_CONFIG=config/search_config.10k_12g_candidates.json
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false
set SILICONFLOW_API_KEY=sk
```

手动验证启动：

```bat
cd /d D:\search_similar_style
D:\search_similar_style\.venv\Scripts\python.exe -m uvicorn src.api_server:app --host 127.0.0.1 --port 8000
```

## 2. NSSM 注册服务

假设 `nssm.exe` 放在：

```bat
C:\tools\nssm\nssm.exe
```

管理员 CMD 执行：

```bat
C:\tools\nssm\nssm.exe install SearchSimilarStyle D:\search_similar_style\scripts\windows\run_api.bat
C:\tools\nssm\nssm.exe set SearchSimilarStyle AppDirectory D:\search_similar_style
C:\tools\nssm\nssm.exe set SearchSimilarStyle Start SERVICE_AUTO_START
C:\tools\nssm\nssm.exe set SearchSimilarStyle AppStdout D:\search_similar_style\logs\service_stdout.log
C:\tools\nssm\nssm.exe set SearchSimilarStyle AppStderr D:\search_similar_style\logs\service_stderr.log
C:\tools\nssm\nssm.exe set SearchSimilarStyle AppRotateFiles 1
C:\tools\nssm\nssm.exe set SearchSimilarStyle AppRotateOnline 1
```

启动服务：

```bat
sc start SearchSimilarStyle
```

停止服务：

```bat
sc stop SearchSimilarStyle
```

重启服务：

```bat
powershell -Command "Restart-Service SearchSimilarStyle"
```

查看状态：

```bat
sc query SearchSimilarStyle
```

## 3. Windows nginx 反代示例

```nginx
server {
    listen       80;
    server_name  your-domain.com;

    client_max_body_size 50m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

## 4. 注意事项

- 用 `python -m uvicorn`，不要直接用裸 `uvicorn`
- 必须配置正确的工作目录 `AppDirectory`
- 先确认 `.venv` 内依赖完整
- 如果模型、配置、图片目录使用相对路径，更依赖工作目录正确
