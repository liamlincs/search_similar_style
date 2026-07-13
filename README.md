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
- `config/search_config.10k_12g_speed.json`（10k数据量 + 12G GPU / 32G内存，极致速度优先）

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
python src/search_similar_return_code.py data/test_samples/T01.png --config config/search_config.10k_12g_speed.json
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

# 10k主力档极速版（速度优先）
SEARCH_CONFIG=config/search_config.10k_12g_speed.json uvicorn src.api_server:app --host 0.0.0.0 --port 8000

# 10k候选列表版（快 + 近似候选3个）
SEARCH_CONFIG=config/search_config.10k_12g_candidates.json uvicorn src.api_server:app --host 0.0.0.0 --port 8000

# 10k模特图候选版（适合上身图）
SEARCH_CONFIG=config/search_config.10k_12g_model_photo.json uvicorn src.api_server:app --host 0.0.0.0 --port 8000

# 10k CPU轻量版（4核16G / 无GPU）
SEARCH_CONFIG=config/search_config.10k_4c16g_cpu.json uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

说明：
- `10k_12g_speed` 已内置“自适应二次检索”：默认先走极速参数；当 top1 分偏低或 top1/top2 过于接近时，自动触发一次高精度复检，改善误匹配。

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
- `tags`：该款在产品库中的标签列表
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

## 3.1) 产品库（SQLite）

当前已新增独立产品库，不再把标签等主数据绑在文件名里。

- 数据库路径：`config/search_config.json` -> `catalog.db_path`
- 默认库文件：`data/product_catalog.db`
- 图片仍然保留在 `data/standard_samples`
- 首次启动 API 时会自动按文件名把图片同步进产品库：
  - `style_code` 仍从 `款号_序号.jpg` 推导
  - 标签单独存 SQLite，可多选、可新增

管理页：

```bash
http://127.0.0.1:8000/catalog
```

批量导入目录可直接写在配置里：

```json
"catalog": {
  "db_path": "data/product_catalog.db",
  "import_source_dir": "/data/new_samples"
}
```

Windows 路径也支持，但 JSON 里要写双反斜杠：

```json
"import_source_dir": "D:\\catalog\\incoming"
```

登录配置：

- `config/search_config.json` -> `catalog.web_auth`
- 默认启用登录保护
- 支持多组账号密码，推荐写法：

```json
"web_auth": {
  "enabled": true,
  "captcha_enabled": true,
  "captcha_timezone": "Asia/Shanghai",
  "users": [
    { "username": "admin", "password": "change-me" },
    { "username": "merch", "password": "change-me-too" }
  ],
  "session_secret": "replace-with-catalog-session-secret",
  "session_ttl_sec": 43200,
  "cookie_name": "catalog_session"
}
```

- 兼容旧写法（单组账号）：
  - `catalog.web_auth.username`
  - `catalog.web_auth.password`
- 强烈建议上线前修改：
  - `catalog.web_auth.users`
  - `catalog.web_auth.session_secret`
- 默认还启用一个后端校验验证码：
  - `catalog.web_auth.captcha_enabled`
  - `catalog.web_auth.captcha_timezone`
  - 验证码规则只在后端生效，前端登录页不会暴露规则

登录/退出路径：

```bash
http://127.0.0.1:8000/catalog/login
http://127.0.0.1:8000/catalog/logout
```

小程序 WebView H5 入口：

```text
https://api.openfire.cloud/catalog?type=product&token=第三方用户token
```

- `type=product` 打开产品库 H5。
- 第三方小程序每次打开 H5 时传当前登录用户的 token，不再使用小程序内置固定 `catalogH5Token` 或后端 `apiKey`。
- 第三方不需要传用户资料；后端会使用 `token` 调第三方验 token / 用户信息接口获取稳定用户 ID，用于“个人产品”隔离。
- 内网调测和公网调测可以分开切：`miniprogram/utils/config.js` 里 `catalogH5BaseUrls.public` 默认是 `https://api.openfire.cloud`，`catalogH5BaseUrls.lan` 可填内网地址，例如 `http://192.168.0.106:8000`；跳转时带 `env=lan` 使用内网，带 `env=public` 使用公网，也可以临时传 `h5_base_url=` 覆盖。
- 页面首次打开会把 `token` 或 `access_token` 缓存在浏览器本地，并在接口请求中通过 `X-Catalog-Token` 透传回后端；直接请求接口时也支持 `X-Catalog-Token` 或 `Authorization: Bearer <token>`。
- 后端可通过 `catalog.external_token_auth.verify_url` 调用第三方验 token 接口，验 token 结果会按 `cache_ttl_sec` 短期缓存，避免每个列表/图片请求都打到第三方。
- 验 token 接口返回的 `user_id` / `userId` / `user` / `openid` / `sub` 会作为当前用户 ID；返回的 `permissions`、`perms` 或 `scope` 会用于后端权限校验，并注入 H5 控制前端入口展示；如果 token 是 JWT，也保留从 payload 读取权限的兼容逻辑。
- 支持权限值：`product:view`、`product:create`、`color:view`、`color:create`，也支持 `*`。

调测建议：

```text
# 内网，适合微信开发者工具或同局域网真机调试
/pages/catalog_webview/index?type=product&env=lan&token=<USER_TOKEN>

# 公网，适合真实第三方小程序验收
/pages/catalog_webview/index?type=product&env=public&token=<USER_TOKEN>

# 临时覆盖，不改 config.js
/pages/catalog_webview/index?type=product&h5_base_url=http%3A%2F%2F192.168.0.106%3A8000&token=<USER_TOKEN>
```

注意：

- 真机小程序 `web-view` 受微信业务域名限制，公网调测需要在微信后台配置 `https://api.openfire.cloud`；内网 HTTP 地址通常只适合开发者工具调试。
- 公网调测时，`api.openfire.cloud` 所在服务器必须能访问 `catalog.external_token_auth.verify_url`。如果第三方验 token 服务只在内网可达，公网环境需要配置公网验 token 地址或打通专线/VPN。
- 切换用户或切换环境时，H5 会用新 URL 上的 `token` 覆盖本地缓存；没有传 token 时会复用浏览器本地缓存的 token。

第三方验 token 配置示例：

```json
"external_token_auth": {
  "enabled": true,
  "verify_url": "https://third.example.com/api/auth/verify",
  "verify_method": "POST",
  "verify_timeout_sec": 3,
  "cache_ttl_sec": 300,
  "fail_open": false,
  "allow_unverified_tokens": false,
  "allowed_tokens": [],
  "default_permissions": ["product:view"]
}
```

`verify_method` 支持 `POST` 或 `GET`。后端都会带 `Authorization: Bearer <token>` 和 `X-Catalog-Token`；`POST` 时还会发送 JSON：`{"token":"..."}`。验 token 接口建议返回：

```json
{
  "valid": true,
  "user_id": "u_123",
  "permissions": ["product:view", "product:create"]
}
```

盈彩汇 `GET getUserInfo` 配置示例：

```json
"external_token_auth": {
  "enabled": true,
  "verify_url": "https://yingcaihui.net/api/getUserInfo",
  "verify_method": "GET",
  "verify_timeout_sec": 3,
  "cache_ttl_sec": 300,
  "fail_open": false,
  "allow_unverified_tokens": false,
  "allowed_tokens": [],
  "default_permissions": ["product:view", "color:view"]
}
```

这种返回格式也兼容：

```json
{
  "code": 200,
  "succeed": true,
  "data": {
    "userId": 1,
    "permissions": ["product:view", "color:view"]
  }
}
```

目录接口：

```bash
# 列表
curl -s "http://127.0.0.1:8000/api/v1/catalog/products?style_code=GZ25&tags=棉,羊毛"

# 单款详情
curl -s "http://127.0.0.1:8000/api/v1/catalog/products/GZ25-1177-1"

# 标签列表
curl -s "http://127.0.0.1:8000/api/v1/catalog/tags"

# 新增标签
curl -s -X POST "http://127.0.0.1:8000/api/v1/catalog/tags" \
  -H "Content-Type: application/json" \
  -d '{"name":"羊毛"}'

# 替换某款标签
curl -s -X PUT "http://127.0.0.1:8000/api/v1/catalog/products/GZ25-1177-1/tags" \
  -H "Content-Type: application/json" \
  -d '{"tags":["羊毛","针织"]}'

# 手动同步 standard_samples 到产品库
curl -s -X POST "http://127.0.0.1:8000/api/v1/catalog/sync"

# 服务器目录预处理导入（先 OCR 出候选文件名）
curl -s -X POST "http://127.0.0.1:8000/api/v1/catalog/imports/prepare" \
  -H "Content-Type: application/json" \
  -d '{"source_dir":"/data/new_samples"}'

# 查询导入任务进度
curl -s "http://127.0.0.1:8000/api/v1/catalog/imports/<job_id>"
```

说明：
- 文件名现在只负责图片定位和初始化款号；
- 款号检索、标签检索应走产品库接口；
- 小程序后续应直接消费产品库返回的 `style_code + tags + image_url`，不要再自行解析文件名。
- Web 产品库支持表单登录；小程序仍通过 `X-API-Key` 调用目录接口，不受影响。
- Web 产品库现在支持“服务器目录批量导入”：
  - 输入服务器本地目录；
  - 后端复用 `src/extract_style_codes.py` 的 OCR 提款号逻辑生成候选文件名；
  - 页面会显示识别进度；
  - 导入前可手工修改每张图的目标文件名；
  - 确认后复制到 `data/standard_samples` 并自动同步到产品库。

### 调试：对比开发者工具与真机上传图

为排查“同一张图在微信开发者工具和真机检索结果不同”的情况，后端可保存每次 `/search` 实际收到并标准化后的查询图。

配置项：

```json
"debug": {
  "save_query_images": true,
  "query_image_dir": "outputs/debug_queries"
}
```

默认：
- `save_query_images=true`
- 输出目录：`outputs/debug_queries`

每次检索会在日志里打印：
- 上传文件名
- 上传字节数
- 后端最终使用的图片尺寸
- 保存后的调试图路径

排查方式：

1. 在开发者工具里用同一张图检索一次
2. 在真机里再检索一次
3. 对比 `outputs/debug_queries` 下两张实际输入图：
   - 尺寸是否一致
   - 方向是否一致
   - 画面边缘是否有裁切差异
   - 压缩痕迹是否不同

如果两边输入图像素内容不同，检索结果不同通常就是预期现象，而不是模型随机漂移。

### 新增标样图片后的操作

当 `data/standard_samples` 有新增图片时，要区分两件事：

1. **同步到产品库（SQLite）**
2. **重载以图搜款的特征检索库**

建议命名仍保持：

```text
款号_序号.jpg
```

例如：

```text
GZ26-0001_000.jpg
GZ26-0001_001.jpg
```

操作步骤：

#### 方式 A：推荐 Web 批量导入

1) 打开：

```bash
http://127.0.0.1:8000/catalog
```

2) 点击“目录批量导入”

3) 输入服务器本地目录，例如：

```bash
/data/new_samples
```

4) 等待 OCR 识别完成：
- 系统会自动提取款号；
- 自动生成候选文件名，如 `GZ26-0001_000.jpg`；
- 自动识别年份标签，规则是“第一个 `-` 前字段末尾两位数字 + `20`”，例如 `BM23-J0831` -> `2023`；
- 若 OCR 失败，仍会给出一个可编辑的候选文件名。

5) 在导入列表中可点击“源文件”打开原图大图预览，人工确认后按需手工修改：
- 目标文件名；
- 批量标签（在“开始识别”下方统一添加，可选已有标签，也可直接输入新增标签）；
- 年份标签（如 `2024`）。

“新增标签”和“批量标签”两个入口下方都会显示已有标签，并可直接删除。

6) 点击“确认导入”

7) 导入确认后，系统会把填写的年份和批量标签一起写入产品库

### 给老数据批量回填年份标签

如果老图片以前已经导入过，只是缺少年份标签，不要删图重导，直接运行：

```bash
python src/backfill_catalog_year_tags.py --dry-run
python src/backfill_catalog_year_tags.py
```

如果要把已有年份标签统一覆盖成按款号重新解析的结果：

```bash
python src/backfill_catalog_year_tags.py --overwrite
```

7) 导入完成后：
- 产品库会自动同步；
- 但以图搜款仍需重启服务，新增图片才会进入特征检索库。

#### 方式 B：人工拷贝 + 同步

1) 把新图片放入：

```bash
data/standard_samples
```

2) 同步到产品库：

```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/catalog/sync"
```

或者在 Web 管理页点击：

```text
/catalog -> 同步图片
```

3) 重启 API 服务，让以图搜款加载新图片特征：

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

如果原服务已在运行，先停止再启动。

#### 为什么要分两步

- 产品库同步只会更新 SQLite 中的：
  - `products`
  - `product_images`
  - `cover_image`
- 以图搜款使用预加载的特征库与缓存，但目录批量导入提交后会立即触发热重载；
- 所以**通过 Web 目录批量导入的新图，不需要重启服务也能参与以图搜款**。

#### 对小程序的影响

- **产品库检索**：完成 `/api/v1/catalog/sync` 后即可查到新款；
- **同款多图预览**：同步后即可生效；
- **以图搜款**：目录批量导入 `commit` 后会自动热重载检索索引；若是手工直接往 `standard_samples` 放图，仍需要显式同步/重载。

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

### 内容安全校验

后端会在用户图片/文本进入业务处理前调用微信内容安全接口：

- 图片上传：调用图片内容安全检查。
- 文本输入：调用 `msgSecCheck`，覆盖款库查询、标签、融合说明等用户输入。

生产环境需要配置：

```bash
export WECHAT_CONTENT_SECURITY_ENABLED=1
export WECHAT_APPID="你的小程序 AppID"
export WECHAT_APPSECRET="你的小程序 AppSecret"
# 可选：使用 msgSecCheck v2 时提供用户 openid
export WECHAT_SECURITY_OPENID="用户 openid"
```

也可在 `config/search_config*.json` 的 `content_security.wechat` 中配置。不要把真实 `AppSecret` 提交到仓库。

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
- 服务器内存较小时，建议使用轻量模型：`export REMBG_MODEL=u2netp`（默认已是 `u2netp`）；如机器资源充足可改为 `u2net`

小程序原型页：
- `pages/recolor/index`
- 底部导航可从“搜款/拼图打印”跳转到“局部改色”

### AI出图（SiliconFlow / Qwen-Image-Edit-2509）

配置环境变量（服务端）：

```bash
export SILICONFLOW_API_KEY="你的_siliconflow_api_key"
```

新增接口：`POST /recolor-ai`（`multipart/form-data`）
- 必填：`file`（主图，对应 SiliconFlow `image`）
- 基础参数：
  - `model`（默认：`Qwen/Qwen-Image-Edit-2509`）
  - `prompt`
  - `target_hex`（可选；用于提示词换色或后处理校色）
  - `x_ratio` / `y_ratio` / `w_ratio` / `h_ratio`
  - `strength`
- 可选参数：
  - `negative_prompt`
  - `seed`
  - `cfg`（小程序默认 `4`，与 SiliconFlow Playground 常用值一致；兼容旧字段 `cfg_scale`）
  - `num_inference_steps`（小程序默认 `20`）
  - `image2`（参考图 data URL；例如衣领图）
  - `image3`（第二张参考图 data URL）
  - `postprocess`（`0/1`；多图合成时建议 `0`，避免颜色后处理影响合成）

说明：
- 后端调用 SiliconFlow `POST /v1/images/generations`。没有参考图时，后端可将框选区域转为蒙版图（作为 `image2`）辅助改色；传入 `image2/image3` 时，它们会作为真实参考图透传给 SiliconFlow，不再被蒙版覆盖。
- 小程序 AI 出图支持“主图 + 参考图1 + 参考图2 + 提示词”。用户可写“把参考图1的衣领合并到主图上”，发送前会自动换算为 SiliconFlow 更容易识别的“把 image 2 的衣领合并到 image 1 上”，不额外追加长约束，尽量保持与 Playground 行为一致。
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

    # AI出图接口
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
