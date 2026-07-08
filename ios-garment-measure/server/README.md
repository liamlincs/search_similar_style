# GarmentMeasure AI Preview Server

独立服务，用火山方舟 Seed3D 接口把「T 恤照片 + 尺寸」生成真实 3D 模型。不要把 `ARK_API_KEY` 放进 iOS App。

## 启动

```bash
cd ios-garment-measure/server
export ARK_API_KEY="你的火山方舟 API Key"
export PUBLIC_BASE_URL="https://api.openfire.cloud"
python -m uvicorn garment_ai_server:app --host 0.0.0.0 --port 8002
```

iPhone App 里服务地址填：

```text
https://api.openfire.cloud
```

健康检查：

```bash
curl http://127.0.0.1:8002/api/v1/health
curl https://api.openfire.cloud/api/v1/health
```

## systemd

在 Tailscale 节点 `100.64.0.2` 上，可以使用本目录下的 `garment-3d-laptop.service`：

```bash
sudo cp garment-3d-laptop.service /etc/systemd/system/garment-3d-laptop.service
sudo systemctl daemon-reload
sudo systemctl enable --now garment-3d-laptop.service
sudo systemctl status garment-3d-laptop.service --no-pager -l
```

服务文件里需要包含这些运行参数：

```ini
Environment=PUBLIC_BASE_URL=https://api.openfire.cloud
Environment=ARK_3D_MODEL=doubao-seed3d-2-0-260328
Environment=ARK_3D_SUBDIVISION=low
Environment=ARK_3D_FILE_FORMAT=usdz
```

Nginx 侧把 `/api/v1/garment/`、`/api/v1/health`、`/static-inputs/` 反代到：

```text
http://100.64.0.2:8002
```

## Seed3D 注意事项

Seed3D 的 `image_url.url` 需要火山云端能访问到，不能是 iPhone/Mac 局域网地址。服务会把上传图片保存到 `/static-inputs/...jpg`，并用 `PUBLIC_BASE_URL` 拼成图片 URL 传给火山。

默认参数：

```bash
export ARK_3D_MODEL="doubao-seed3d-2-0-260328"
export ARK_3D_SUBDIVISION="low"
export ARK_3D_FILE_FORMAT="usdz"
```

如果只在局域网运行，App 可以访问你的服务，但火山无法下载输入图片，Seed3D 任务会失败。当前推荐用 `https://api.openfire.cloud` 经 Nginx 反代到 Tailscale 节点 `100.64.0.2:8002`，不再依赖 Cloudflare Tunnel。
