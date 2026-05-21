# 以图搜款（两步流程）

## 流程

1. 从 `standard_samples` 图片左上角提取款号，并将标样重命名为：
- `款号_000.png`
- `款号_001.png`
- 同款号自动递增

2. 输入一张测试图（无款号），做近似检索，返回前 K 个近似款号（直接由命中标样文件名解析）。

## 配置

配置文件：`config/search_config.json`

预置两套可切换配置：
- `config/search_config.fast.json`（速度优先）
- `config/search_config.accurate.json`（精度优先）
- `config/search_config.10k_12g.json`（10k数据量 + 12G GPU / 32G内存主力档位）

关键项：
- `ocr.backend`：当前为 `rapidocr`（本地OCR，不依赖 deepseek / tesseract）
- `paths.standard_dir`：标样目录
- `paths.standard_pattern`：标样匹配模式（建议 `*`）
- `paths.image_exts`：支持的图片后缀（如 `["png","jpg","jpeg"]`）
- `search.top_k`：返回前 K 个近似款号
- `search.candidate_multiplier`：先召回更多图片再按款号去重
- `search.feature_backend`：`clip` 或 `classic`（推荐 `clip`）

## 运行

```bash
cd /Users/tk/Workspace/search_similar_style
. .venv/bin/activate
```

### 0) 下载 CLIP 本地模型（首次一次）

```bash
python scripts/download_clip_model.py
```

说明：检索脚本只使用本地 CLIP 模型目录 `models/clip-vit-base-patch32`。

### 1) 提取款号并重命名标样

```bash
python src/extract_style_codes.py
```

可先预览：

```bash
python src/extract_style_codes.py --dry-run
```

### 2) 检索单张测试图并返回近似款号 JSON

```bash
python src/search_similar_return_code.py data/test_samples/T01.png
# 或
python src/search_similar_return_code.py data/test_samples/T01.jpg
```

按配置文件切换：

```bash
python src/search_similar_return_code.py data/test_samples/T01.png --config config/search_config.fast.json
python src/search_similar_return_code.py data/test_samples/T01.png --config config/search_config.accurate.json
python src/search_similar_return_code.py data/test_samples/T01.png --config config/search_config.10k_12g.json
```

输出：标准输出 JSON（可自行重定向到文件）

### 3) FastAPI 对外服务（保持原命令行不变，新增 API）

前置安装与激活：

```bash
cd /Users/tk/Workspace/search_similar_style
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

若在 Ubuntu 启动时报错 `ImportError: libGL.so.1: cannot open shared object file`，先安装系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

启动：

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

按配置文件启动（推荐）：

```bash
# 速度优先
SEARCH_CONFIG=config/search_config.fast.json uvicorn src.api_server:app --host 0.0.0.0 --port 8000

# 精度优先
SEARCH_CONFIG=config/search_config.accurate.json uvicorn src.api_server:app --host 0.0.0.0 --port 8000

# 10k主力档（12G GPU / 32G RAM）
SEARCH_CONFIG=config/search_config.10k_12g.json uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl -s http://127.0.0.1:8000/health
```

就绪检查（建议业务调用前先探测，200 表示可检索）：

```bash
curl -i http://127.0.0.1:8000/ready
```

上传图片检索：

```bash
curl -s -X POST "http://127.0.0.1:8000/search" \
  -H "X-API-Key: replace-with-real-key-a" \
  -F "file=@data/test_samples/T03.jpg"
```

返回 topk 并内嵌 base64 图片（示例：仅前2张）：

```bash
curl -s -X POST "http://127.0.0.1:8000/search?include_image_base64=true&base64_topn=2" \
  -H "X-API-Key: replace-with-real-key-a" \
  -F "file=@data/test_samples/T03.jpg"
```

返回字段：
- `style_code`：命中款号
- `best_standard_image`：命中的标样图片文件名
- `best_standard_image_url`：可直接给其它系统展示的图片 URL
- `best_standard_image_base64`：图片 base64（仅当 `include_image_base64=true`）
- `best_standard_image_mime`：图片 MIME（如 `image/jpeg`）
- `score`：分数

鉴权配置：
- `config/search_config.json` 的 `auth.enabled` 控制是否启用 API Key
- `auth.api_keys` 可配置多个 `{user, key}`，用于给不同用户分配不同 key
- `/search` 需要请求头 `X-API-Key`
- `/images/{image_name}` 支持两种方式：
- 方式1：请求头 `X-API-Key`
- 方式2：签名 URL（`best_standard_image_url` 已自动附带 `exp` + `sig`，适合小程序 `image` 组件）
- `/image-url?image_name=xxx`：返回新的签名图片 URL（用于签名过期后的刷新）
- `auth.image_url_secret`：图片签名密钥（强烈建议改成高强度随机串）
- `auth.image_url_ttl_sec`：签名 URL 过期时间（秒，默认 600）

Cloudflare 回源防火墙（UFW）一键更新：

```bash
sudo bash scripts/update_ufw_cloudflare.sh
```

说明：
- 会自动拉取 Cloudflare 官方 IPv4/IPv6 网段并更新 80/443 放行规则
- 会保留 SSH（22/OpenSSH）放行，避免把自己锁在机器外
- 会添加 80/443 的默认 deny 作为兜底

## 图片近似检索原理

当前检索采用“向量召回 + 款号去重”的方式：

1. **特征提取**
- `feature_backend=clip` 时：使用本地 CLIP（`clip_features.py`）提取图像语义向量。
- `feature_backend=classic` 时：使用手工特征（颜色/纹理/边缘/结构，`features.py`）。

2. **相似度计算**
- 查询图向量与标样向量做余弦相似度（归一化点积）。

3. **候选召回**
- 先取 `top_k * candidate_multiplier` 张相似图片。

4. **按款号去重**
- 从命中图片文件名解析款号前缀（`款号_序号.png` 的 `款号`）。
- 同一款号保留最高分图片。

5. **返回结果**
- 输出前 `top_k` 个近似款号，每项包含：
  - `style_code`
  - `best_standard_image`
  - `score`

## 4) 微信小程序前端（上传/拍照以图搜款）

小程序目录：`miniprogram/`

### 功能
- 从相册上传图片检索
- 直接拍照检索
- 卡片式展示 topk 结果（款号、分数、结果图）
- 点击结果图可预览大图

### 使用步骤
1. 启动后端 API（默认 `http://127.0.0.1:8000`）
2. 打开微信开发者工具，导入 `miniprogram` 目录
3. 修改 `miniprogram/utils/config.js`：
   - `baseUrl`：默认已配置为 `https://api.seekfire.cloud`
   - `apiKey`：对应的 `X-API-Key`
   - `printBaseUrl`：拼图打印服务地址（可与 `baseUrl` 相同）
   - `printPaths`：拼图打印接口 URI（默认使用 `/api/v1/...`）
4. 运行后即可上传或拍照搜款

### 说明
- 当前接口调用：`POST /search`（`multipart/form-data`，字段名 `file`）
- 渲染使用返回字段 `topk_style_codes[].best_standard_image_url`
- 内置重试：默认最多重试 4 次（网络错误、408、429、5xx 会重试；4xx 一般不重试）
- 重试参数可在 `miniprogram/utils/config.js` 的 `retry` 配置中调整
- 若后端启用 HTTPS 域名，建议将 `baseUrl` 切到 HTTPS
- 拼图打印页面不再在界面配置地址，统一读取 `miniprogram/utils/config.js`

### 局部改色（MVP）

新增接口：`POST /recolor`（`multipart/form-data`）
- `file`：原图
- `target_hex`：目标颜色（如 `FF5500`）
- `x_ratio` / `y_ratio`：矩形区域起点（0~1）
- `w_ratio` / `h_ratio`：矩形区域宽高比例（0~1）
- `strength`：改色强度（0~1）
- `feather_ratio`：边缘羽化比例（0~0.25）
- `auto_mask`：`1` 时自动抠主体换色（推荐），`0` 时按矩形参数换色

返回：
- `recolored_url`：改色结果图地址（`/recolor-static/outputs/...`）
- `bbox`：服务端实际使用的像素矩形

主体抠图说明：
- 标准换色优先使用 `rembg(U2Net)` 生成主体蒙版（分割更稳定）
- 若未安装 `rembg` 或模型加载失败，自动回退到规则分割
- `rembg` 首次运行会下载 U2Net 模型文件，请确保服务器可联网

小程序原型页：
- `pages/recolor/index`
- 底部导航可从“搜款/拼图打印”跳转到“局部改色”

### AI改色（SiliconFlow / Qwen-Image-Edit-2509）

配置环境变量（服务端）：

```bash
export SILICONFLOW_API_KEY="你的_siliconflow_api_key"
```

新增接口：`POST /recolor-ai`（`multipart/form-data`）
- 必填：`file`
- 基础参数：
  - `model`（默认：`Qwen/Qwen-Image-Edit-2509`）
  - `target_hex`
  - `x_ratio` / `y_ratio` / `w_ratio` / `h_ratio`
  - `strength`
- 可选参数：
  - `negative_prompt`
  - `seed`
  - `num_inference_steps`
  - `image2`
  - `image3`

说明：
- 后端调用 SiliconFlow `POST /v1/images/generations`，并将框选区域转为蒙版图（作为 `image2`）辅助编辑。
- 返回字段 `used_params` 会给出本次实际生效的参数。

### Nginx 反向代理（按 URI 区分搜款与打印）

示例：统一后端都在 `127.0.0.1:8000`（搜款 + 拼图打印同一服务）。

```nginx
server {
    listen 443 ssl;
    server_name api.seekfire.cloud;

    # 搜款接口
    location /search {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /image-url {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /images/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # 拼图打印接口（已整合进同一个后端）
    location /api/v1/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # 拼图打印静态资源（预览图和PDF）
    location /print-static/ {
        proxy_pass http://127.0.0.1:8000;
    }

    location /print-storage/ {
        proxy_pass http://127.0.0.1:8000;
    }

    # 局部改色接口
    location = /recolor {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # AI改色接口
    location = /recolor-ai {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # 改色结果图
    location /recolor-static/ {
        proxy_pass http://127.0.0.1:8000;
    }
}
```

对应小程序配置示例：

```js
baseUrl: "https://api.seekfire.cloud",
printBaseUrl: "https://api.seekfire.cloud",
printPaths: {
  templates: "/api/v1/templates",
  upload: "/api/v1/images/upload",
  render: "/api/v1/render"
}
```

### Cloudflare 规则模板（小程序 API）

如果你在小程序里看到 `Sorry, you have been blocked` 或 `Please enable cookies`，通常是 Cloudflare 对 API 路径启用了挑战页（JS/Cookie Challenge），小程序无法完成挑战。

推荐规则顺序如下（从上到下）：

1. `Skip` 规则（优先，放在最前）
- 作用：对 API 路径跳过 WAF/Bot 挑战
- 表达式：

```txt
starts_with(http.request.uri.path, "/api/")
or http.request.uri.path eq "/search"
or http.request.uri.path eq "/image-url"
or starts_with(http.request.uri.path, "/images/")
or starts_with(http.request.uri.path, "/print-static/")
or starts_with(http.request.uri.path, "/print-storage/")
or http.request.uri.path eq "/recolor"
or http.request.uri.path eq "/recolor-ai"
or starts_with(http.request.uri.path, "/recolor-static/")
```

- Action：`Skip`
- Skip components：`Managed WAF`、`Super Bot Fight Mode`、`Bot Fight Mode`（按你账号里可选项勾选）

2. 可选 `Allow` 规则（更严格）
- 作用：仅对带有效 `X-API-Key` 的请求放行 API
- 表达式示例（把 `replace-with-real-key-a` 换成真实 key）：

```txt
(
  (http.request.uri.path starts_with "/api/")
  or (http.request.uri.path eq "/search")
  or (http.request.uri.path eq "/image-url")
)
and (http.request.headers["x-api-key"][0] eq "replace-with-real-key-a")
```

- Action：`Allow`

3. 避免对 API 路径使用 Challenge
- 不要给上述 API 路径配置 `Managed Challenge` / `JS Challenge` / `Interactive Challenge`。
- 这些挑战适合浏览器页面，不适合小程序接口。
