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
  const finalUrl = buildPrintUrl(path);
  return new Promise((resolve, reject) => {
    wx.request({
      url: finalUrl,
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
      fail: (err) => {
        console.error("[printRequest:fail]", method || "GET", finalUrl, err);
        reject(new Error((err && err.errMsg) || "网络错误"));
      }
    });
  });
}

function printUpload(filePath) {
  const paths = config.printPaths || {};
  const uploadPath = paths.upload || "/api/v1/images/upload";
  const finalUrl = buildPrintUrl(uploadPath);
  return new Promise((resolve, reject) => {
    wx.uploadFile({
      url: finalUrl,
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
      fail: (err) => {
        console.error("[printUpload:fail]", finalUrl, err);
        reject(new Error((err && err.errMsg) || "上传失败"));
      }
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

function recolorUpload(filePath, options = {}) {
  const recolorPath = config.recolorPath || "/recolor";
  const finalUrl = `${config.baseUrl}${recolorPath}`;
  const formData = {
    target_hex: options.target_hex || "FF5500",
    x_ratio: String(options.x_ratio ?? 0.2),
    y_ratio: String(options.y_ratio ?? 0.2),
    w_ratio: String(options.w_ratio ?? 0.4),
    h_ratio: String(options.h_ratio ?? 0.4),
    strength: String(options.strength ?? 0.8),
    feather_ratio: String(options.feather_ratio ?? 0.02),
  };
  if (options.auto_mask !== undefined && options.auto_mask !== null) formData.auto_mask = String(options.auto_mask ? 1 : 0);
  return new Promise((resolve, reject) => {
    wx.uploadFile({
      url: finalUrl,
      filePath,
      name: "file",
      timeout: config.timeout,
      formData,
      header: {
        "X-API-Key": config.apiKey
      },
      success: (res) => {
        const raw = typeof res.data === "string" ? res.data : JSON.stringify(res.data || {});
        const blockedByCf = /Attention Required!|Sorry, you have been blocked|Cloudflare|Please enable cookies/i.test(raw);
        if (blockedByCf) {
          reject(new Error("请求被 Cloudflare 拦截：请放行 /recolor 与 /recolor-static 路径"));
          return;
        }
        if (res.statusCode >= 200 && res.statusCode < 300) {
          try {
            resolve(JSON.parse(res.data || "{}"));
          } catch (_err) {
            reject(new Error("改色返回解析失败"));
          }
          return;
        }
        reject(new Error(raw || "换色失败"));
      },
      fail: (err) => reject(new Error((err && err.errMsg) || "换色失败"))
    });
  });
}

function recolorAiUpload(filePath, options = {}) {
  const recolorPath = config.recolorAiPath || "/recolor-ai";
  const finalUrl = `${config.baseUrl}${recolorPath}`;
  const formData = {
    model: options.model || "Qwen/Qwen-Image-Edit-2509",
    target_hex: options.target_hex || "FF5500",
    x_ratio: String(options.x_ratio ?? 0.2),
    y_ratio: String(options.y_ratio ?? 0.2),
    w_ratio: String(options.w_ratio ?? 0.4),
    h_ratio: String(options.h_ratio ?? 0.4),
    strength: String(options.strength ?? 0.7),
  };
  if (options.prompt) formData.prompt = String(options.prompt);
  if (options.negative_prompt) formData.negative_prompt = String(options.negative_prompt);
  if (options.seed !== undefined && options.seed !== null) formData.seed = String(options.seed);
  if (options.cfg_scale !== undefined && options.cfg_scale !== null) formData.cfg_scale = String(options.cfg_scale);
  if (options.postprocess !== undefined && options.postprocess !== null) formData.postprocess = String(options.postprocess ? 1 : 0);
  if (options.num_inference_steps !== undefined && options.num_inference_steps !== null) formData.num_inference_steps = String(options.num_inference_steps);
  if (options.image2) formData.image2 = String(options.image2);
  if (options.image3) formData.image3 = String(options.image3);
  return new Promise((resolve, reject) => {
    wx.uploadFile({
      url: finalUrl,
      filePath,
      name: "file",
      timeout: config.timeout,
      formData,
      header: {
        "X-API-Key": config.apiKey
      },
      success: (res) => {
        const raw = typeof res.data === "string" ? res.data : JSON.stringify(res.data || {});
        const blockedByCf = /Attention Required!|Sorry, you have been blocked|Cloudflare|Please enable cookies/i.test(raw);
        if (blockedByCf) {
          reject(new Error("请求被 Cloudflare 拦截：请放行 /recolor-ai 与 /recolor-static 路径"));
          return;
        }
        if (res.statusCode >= 200 && res.statusCode < 300) {
          try {
            resolve(JSON.parse(res.data || "{}"));
          } catch (_err) {
            reject(new Error("AI改色返回解析失败"));
          }
          return;
        }
        reject(new Error(raw || "AI换色失败"));
      },
      fail: (err) => reject(new Error((err && err.errMsg) || "AI换色失败"))
    });
  });
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
  toPrintAbsoluteUrl,
  recolorUpload,
  recolorAiUpload
};
