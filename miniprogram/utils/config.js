const config = {
  // 与 curl 保持一致
  baseUrl: "https://api.openfire.cloud",
  // 拼图打印服务地址；不填则复用 baseUrl
  printBaseUrl: "https://api.openfire.cloud",
  apiKey: "replace-with-real-key-a",
  enableWechatSession: false,
  searchPath: "/search",
  wechatSessionPath: "/api/v1/wechat/session",
  imageUrlPath: "/api/v1/image-url",
  catalogPaths: {
    products: "/api/v1/catalog/products",
    tags: "/api/v1/catalog/tags",
    imports: "/api/v1/catalog/imports"
  },
  colorCardPaths: {
    libraries: "/api/v1/color-card/libraries",
    cards: "/api/v1/color-card/cards",
    match: "/api/v1/color-card/match"
  },
  printPaths: {
    templates: "/api/v1/templates",
    upload: "/api/v1/images/upload",
    render: "/api/v1/render"
  },
  recolorPath: "/recolor",
  recolorAiPath: "/recolor-ai",
  enableExperienceVersion: true,
  catalogH5Path: "/catalog",
  // H5 可单独切换，便于内网调测和公网调测分开验证。
  // 跳转参数支持 env=public/lan，或 h5_base_url=http%3A%2F%2F192.168.0.106%3A8000 临时覆盖。
  catalogH5BaseUrl: "",
  catalogH5BaseUrls: {
    public: "https://api.openfire.cloud",
    lan: ""
  },
  catalogH5Env: "public",
  includeImageBase64: false,
  requestTimeout: 15000,
  uploadTimeout: 300000,
  timeout: 300000,
  wechatSessionTimeout: 5000,
  recolorUpload: {
    maxSide: 1600,
    quality: 82
  },
  retry: {
    maxRetries: 4,
    baseDelayMs: 800,
    maxDelayMs: 5000,
    jitterRatio: 0.25
  },
  imageLoadRetry: {
    maxRetries: 2
  }
};

module.exports = config;
