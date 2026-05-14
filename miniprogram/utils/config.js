const config = {
  // 与 curl 保持一致
  baseUrl: "https://api.seekfire.cloud",
  apiKey: "replace-with-real-key-a",
  searchPath: "/search",
  imageUrlPath: "/image-url",
  includeImageBase64: false,
  timeout: 30000,
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
