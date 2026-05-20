const config = require("./config");

function buildSearchUrl() {
  const query = config.includeImageBase64 ? "?include_image_base64=true" : "";
  return `${config.baseUrl}${config.searchPath}${query}`;
}

function getPrintBaseUrl() {
  return (config.printBaseUrl || config.baseUrl || "").replace(/\/+$/, "");
}

function buildPrintUrl(path) {
  return `${getPrintBaseUrl()}${path}`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function computeBackoffDelay(attempt) {
  const retryCfg = config.retry || {};
  const base = Number(retryCfg.baseDelayMs || 800);
  const max = Number(retryCfg.maxDelayMs || 5000);
  const jitterRatio = Number(retryCfg.jitterRatio || 0.25);
  const exp = Math.min(max, base * Math.pow(2, Math.max(0, attempt - 1)));
  const jitter = exp * jitterRatio * (Math.random() * 2 - 1);
  return Math.max(0, Math.floor(exp + jitter));
}

function shouldRetryHttp(statusCode) {
  if (statusCode === 408 || statusCode === 429) return true;
  return statusCode >= 500;
}

function doUpload(filePath) {
  return new Promise((resolve, reject) => {
    wx.uploadFile({
      url: buildSearchUrl(),
      filePath,
      name: "file",
      timeout: config.timeout,
      header: {
        "X-API-Key": config.apiKey
      },
      success: (res) => {
        let parsed = {};
        try {
          parsed = JSON.parse(res.data || "{}");
        } catch (err) {
          reject(new Error("服务返回内容不是合法 JSON"));
          return;
        }

        if (res.statusCode === 200) {
          resolve(parsed);
          return;
        }

        const message = (parsed && parsed.detail) || `请求失败: HTTP ${res.statusCode}`;
        const err = new Error(message);
        err.statusCode = res.statusCode;
        reject(err);
      },
      fail: (err) => {
        const e = new Error(err.errMsg || "上传失败");
        e.isNetworkError = true;
        reject(e);
      }
    });
  });
}

function fetchSignedImageUrl(imageName) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: `${config.baseUrl}${config.imageUrlPath}`,
      method: "GET",
      timeout: config.timeout,
      data: { image_name: imageName },
      header: {
        "X-API-Key": config.apiKey
      },
      success: (res) => {
        const body = res.data || {};
        if (res.statusCode !== 200 || !body.image_url) {
          const msg = body.detail || `刷新图片地址失败: HTTP ${res.statusCode}`;
          reject(new Error(msg));
          return;
        }
        resolve(body);
      },
      fail: (err) => {
        reject(new Error(err.errMsg || "刷新图片地址失败"));
      }
    });
  });
}

function parseErrorMessage(res) {
  if (!res) return "请求失败";
  if (typeof res.data === "string") return res.data;
  if (res.data && typeof res.data.detail === "string") return res.data.detail;
  return `请求失败: HTTP ${res.statusCode || "unknown"}`;
}

function printRequest(path, method, data) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: buildPrintUrl(path),
      method: method || "GET",
      data: data || null,
      timeout: config.timeout,
      header: {
        "content-type": "application/json",
        "X-API-Key": config.apiKey
      },
      success: (res) => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve(res.data);
          return;
        }
        reject(new Error(parseErrorMessage(res)));
      },
      fail: (err) => reject(new Error((err && err.errMsg) || "网络错误"))
    });
  });
}

function printUpload(filePath) {
  const paths = config.printPaths || {};
  const uploadPath = paths.upload || "/api/v1/images/upload";
  return new Promise((resolve, reject) => {
    wx.uploadFile({
      url: buildPrintUrl(uploadPath),
      filePath,
      name: "file",
      timeout: config.timeout,
      header: {
        "X-API-Key": config.apiKey
      },
      success: (res) => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          try {
            resolve(JSON.parse(res.data || "{}"));
          } catch (err) {
            reject(new Error("上传返回解析失败"));
          }
          return;
        }
        reject(new Error(res.data || "上传失败"));
      },
      fail: (err) => reject(new Error((err && err.errMsg) || "上传失败"))
    });
  });
}

function fetchPrintTemplates() {
  const paths = config.printPaths || {};
  const templatesPath = paths.templates || "/api/v1/templates";
  return printRequest(templatesPath, "GET");
}

function renderPrintLayout(payload) {
  const paths = config.printPaths || {};
  const renderPath = paths.render || "/api/v1/render";
  return printRequest(renderPath, "POST", payload);
}

function toPrintAbsoluteUrl(pathOrUrl) {
  if (!pathOrUrl) return "";
  if (/^https?:\/\//.test(pathOrUrl)) return pathOrUrl;
  return `${getPrintBaseUrl()}${pathOrUrl}`;
}

async function uploadAndSearch(filePath) {
  const retryCfg = config.retry || {};
  const maxRetries = Number(retryCfg.maxRetries || 0);

  let lastError = null;
  for (let attempt = 1; attempt <= maxRetries + 1; attempt += 1) {
    try {
      return await doUpload(filePath);
    } catch (err) {
      lastError = err;
      const statusCode = Number(err.statusCode || 0);
      const retryable = err.isNetworkError || shouldRetryHttp(statusCode);
      const canRetry = retryable && attempt <= maxRetries;

      if (!canRetry) {
        throw err;
      }

      const delay = computeBackoffDelay(attempt);
      await sleep(delay);
    }
  }

  throw lastError || new Error("上传失败");
}

module.exports = {
  uploadAndSearch,
  fetchSignedImageUrl,
  fetchPrintTemplates,
  printUpload,
  renderPrintLayout,
  toPrintAbsoluteUrl
};
