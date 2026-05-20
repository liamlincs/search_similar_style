# 腾讯云 + Tailscale + 笔记本 RTX4080 部署方案

## 目标
- 公网入口放在腾讯云（稳定、可控）
- 检索推理放在笔记本（RTX 4080 + 64G）
- 云上 API 层与本地检索层通过 Tailscale 内网互联

## 一、推荐拓扑

```text
Client
  |
  | HTTPS(443)
  v
Cloudflare (可选，但强烈推荐)
  |
  v
Tencent CVM (公网)
  - Nginx: 443/80
  - API Gateway(可选): FastAPI
  - 仅将检索请求转发到 Tailscale 内网
  |
  | Tailscale WireGuard
  v
Laptop (RTX 4080 + 64G)
  - Search Service (FastAPI/uvicorn)
  - GPU 推理 + 检索
```

## 二、端口规划

### 腾讯云 CVM
- `22/tcp`：SSH 管理
- `80/tcp`：ACME 验证、HTTPS 跳转
- `443/tcp`：公网 API 入口
- 不对公网开放 `8000`（仅本机回环）

### 笔记本
- `9000/tcp`：检索服务（仅监听 Tailscale IP）
- 不对公网开放

### 内网（Tailscale）
- CVM -> Laptop(Tailscale IP): `9000/tcp`

## 三、安装步骤

## 3.1 CVM 安装基础组件
```bash
sudo apt-get update
sudo apt-get install -y nginx curl git
```

## 3.2 CVM 安装 Tailscale
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale ip -4
```

## 3.3 笔记本（Ubuntu 22.04）安装 Tailscale
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```
登录同一 tailnet 后查看 IP：
```bash
tailscale ip -4
```

## 3.4 笔记本部署检索服务
```bash
cd /path/to/search_similar_style
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

建议服务仅监听 Tailscale 地址（示例 `100.88.10.20`）：
```bash
SEARCH_CONFIG=config/search_config.10k_12g.json \
uvicorn src.api_server:app --host 100.88.10.20 --port 9000 --workers 1 --timeout-keep-alive 75
```

## 四、CVM 反向代理配置（转发到笔记本）

将 `api.example.com` 替换为你的真实 API 域名，`100.88.10.20` 替换为笔记本 Tailscale IP。

文件：`/etc/nginx/sites-available/search_api.conf`

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name api.example.com;
    location /.well-known/acme-challenge/ { root /var/www/html; }
    location / { return 301 https://$host$request_uri; }
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    client_max_body_size 20m;
    keepalive_timeout 65;
    proxy_connect_timeout 5s;
    proxy_send_timeout 120s;
    proxy_read_timeout 120s;

    location = /health {
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_pass http://100.88.10.20:9000;
    }

    location = /ready {
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_pass http://100.88.10.20:9000;
    }

    location = /search {
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_pass http://100.88.10.20:9000;
    }

    location ^~ /images/ {
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_pass http://100.88.10.20:9000;
    }

    location / { return 444; }
}
```

启用配置：
```bash
sudo ln -sf /etc/nginx/sites-available/search_api.conf /etc/nginx/sites-enabled/search_api.conf
sudo nginx -t
sudo systemctl reload nginx
```

## 五、证书申请（CVM 上）
```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.example.com
```

续期检查：
```bash
sudo certbot renew --dry-run
```

## 六、系统服务（建议）

## 6.1 笔记本（Ubuntu 22.04）上用 systemd 持久化
创建服务：
```bash
sudo tee /etc/systemd/system/search-similar-laptop.service >/dev/null <<'SERVICE'
[Unit]
Description=Search Similar API on Laptop (Tailscale)
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/workspaces/search_similar_style
Environment=SEARCH_CONFIG=config/search_config.10k_12g.json
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=TOKENIZERS_PARALLELISM=false
ExecStart=/home/ubuntu/workspaces/search_similar_style/.venv/bin/python -m uvicorn src.api_server:app --host 100.88.10.20 --port 9000 --workers 1 --timeout-keep-alive 75
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
SERVICE
```

启用并启动：
```bash
sudo systemctl daemon-reload
sudo systemctl enable search-similar-laptop
sudo systemctl restart search-similar-laptop
sudo systemctl status search-similar-laptop --no-pager -l
```

## 6.2 CVM 上 Nginx 自启动
```bash
sudo systemctl enable nginx
sudo systemctl restart nginx
```

## 七、连通性与接口测试

## 7.1 CVM 测笔记本内网连通
```bash
curl -s http://100.88.10.20:9000/health
curl -s http://100.88.10.20:9000/ready
```

## 7.2 外网测试
```bash
curl -i https://api.example.com/ready
curl -vS -X POST "https://api.example.com/search" \
  -H "X-API-Key: replace-with-real-key-a" \
  -H "Expect:" \
  -F "file=@/absolute/path/T03.jpg"
```

## 八、安全建议
1. CVM 只开放 22/80/443
2. 笔记本检索服务仅监听 Tailscale IP，不监听 `0.0.0.0`
3. 启用 API Key + Nginx 限流
4. 上 Cloudflare（WAF / Bot 防护 / Rate Limit）

## 九、常见问题
1. **请求偶发失败、重试成功**
   - 优先检查是否公网链路抖动
   - 客户端增加 `--retry --retry-all-errors`
2. **ready 正常但 search 偶发失败**
   - 检查上传文件大小、Nginx 超时与 `client_max_body_size`
3. **连接 Tailscale 失败**
   - 确认两端在同一 tailnet 且 ACL 放行
   - `tailscale status` 查看节点状态

## 十、可选增强
- 在 CVM 加一个轻量 API Gateway（只做鉴权与路由）
- 检索服务拆成独立进程（模型服务）以便灰度和热更新
- 增加 Prometheus + Grafana 监控请求时延与错误率
