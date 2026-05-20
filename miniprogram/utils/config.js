const config = {
  // 与 curl 保持一致
  baseUrl: "https://api.seekfire.cloud",
  // 拼图打印服务地址；不填则复用 baseUrl
  printBaseUrl: "https://api.seekfire.cloud",
  apiKey: "replace-with-real-key-a",
  searchPath: "/search",
  imageUrlPath: "/image-url",
  printPaths: {
    templates: "/api/v1/templates",
    upload: "/api/v1/images/upload",
    render: "/api/v1/render"
  },
  recolorPath: "/recolor",
  recolorAiPath: "/recolor-ai",
  includeImageBase64: false,
  timeout: 60000,
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
