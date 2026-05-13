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
- 除 `/health` 外，其它接口都需要请求头 `X-API-Key`

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
