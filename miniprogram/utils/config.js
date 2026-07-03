const config = {
  // 与 curl 保持一致
  baseUrl: "https://api.openfire.cloud",
  // 拼图打印服务地址；不填则复用 baseUrl
  printBaseUrl: "https://api.openfire.cloud",
  apiKey: "replace-with-real-key-a",
  searchPath: "/search",
  imageUrlPath: "/image-url",
  catalogPaths: {
    products: "/api/v1/catalog/products",
    tags: "/api/v1/catalog/tags"
  },
  colorCardPaths: {
    libraries: "/api/v1/color-card/libraries",
    match: "/api/v1/color-card/match"
  },
  printPaths: {
    templates: "/api/v1/templates",
    upload: "/api/v1/images/upload",
    render: "/api/v1/render"
  },
  recolorPath: "/recolor",
  recolorAiPath: "/recolor-ai",
  enableEnterpriseAiGeneration: true,
  includeImageBase64: false,
  timeout: 300000,
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
