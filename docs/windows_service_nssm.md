# Windows 服务部署（NSSM + uvicorn）

## 1. 创建 venv

建议在 Windows 上优先使用 `Python 3.10.x`。

先确认 Python：

```bat
python --version
py -3.10 --version
```

在项目目录创建虚拟环境：

```bat
cd /d D:\search_similar_style
py -3.10 -m venv .venv
```

如果系统里 `python` 已经是目标版本，也可以：

```bat
python -m venv .venv
```

激活虚拟环境：

### CMD

```bat
.\.venv\Scripts\activate
```

### PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

升级 `pip`：

```bat
python -m pip install --upgrade pip
```

安装项目依赖：

```bat
python -m pip install -r requirements.txt
```

确认关键模块可导入：

```bat
python -c "import fastapi, uvicorn; print('api deps ok')"
```

退出虚拟环境：

```bat
deactivate
```

## 2. 启动脚本

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

## 3. NSSM 注册服务

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

## 4. Windows nginx 反代示例

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

## 5. 注意事项

- 用 `python -m uvicorn`，不要直接用裸 `uvicorn`
- 必须配置正确的工作目录 `AppDirectory`
- 先确认 `.venv` 内依赖完整
- 如果模型、配置、图片目录使用相对路径，更依赖工作目录正确

## 6. Windows Server 2012 R2 与 Win11 的区别

### 结论

- `Win11`：更推荐，兼容新 Python 和新依赖更好
- `Windows Server 2012 R2`：可以部署，但更容易遇到二进制依赖兼容问题

### 主要差异

#### Python 版本

- `Windows Server 2012 R2` 建议优先使用 `Python 3.10.x`
- 不建议在 `2012 R2` 上直接使用过新的 Python 版本
- `Win11` 上通常 `Python 3.10.x / 3.11.x` 都更容易跑通

#### 容易出问题的依赖

这个项目里更容易受系统版本影响的依赖包括：

- `torch`
- `transformers`
- `onnxruntime`
- `opencv-python`
- `rapidocr_onnxruntime`
- `numpy / pandas / scikit-learn`

在 `Windows Server 2012 R2` 上更容易遇到：

- wheel 不兼容
- DLL 加载失败
- VC++ Runtime 缺失
- CPU 指令集或底层运行库不兼容

#### 服务托管

- `NSSM` 在 `Win11` 和 `Windows Server 2012 R2` 上都可以用
- `run_api.bat + NSSM` 的部署方式没有本质区别
- 真正的区别主要在 Python 环境和依赖是否能稳定安装运行

### 2012 R2 部署建议

如果必须部署在 `Windows Server 2012 R2`，建议：

1. 固定 `Python 3.10.x`
2. 单独创建虚拟环境
3. 先手动运行项目，确认完全正常，再注册服务
4. 锁定依赖版本，不要随意升级
5. 优先验证以下模块都能正常导入：

```bat
python -c "import fastapi, uvicorn, torch, transformers, onnxruntime, cv2, rapidocr_onnxruntime; print('ok')"
```

### 建议

- 新部署优先选择 `Win11` 或更新版本的 Windows Server
- `Windows Server 2012 R2` 仅建议在必须沿用旧服务器环境时使用
