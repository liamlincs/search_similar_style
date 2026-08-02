const config = require("../../utils/config");
const { ColorMeter } = require("../../utils/color_meter_bluetooth");
const { retry } = require("../../utils/color_meter_utils");
const METER_LAB_STORAGE_KEY = "catalog_meter_bridge_lab";

function scoreMeterDevice(device) {
  const name = `${device && (device.name || device.localName) || ""}`.toLowerCase();
  let score = 0;
  if (/color|colour|meter|spectro|colormeter|色差|测色|颜色/.test(name)) score += 100;
  if (/pt|rm|cr|wr|ys|nh|3nh|fru|bn/.test(name)) score += 20;
  return score + name.length;
}

Page({
  data: {
    url: "",
    meterConnecting: false,
    meterMeasuring: false
  },

  onLoad(options) {
    const url = this.buildCatalogUrl(options);
    const defaultTitle = options.type === "color" ? "色卡库" : "产品库";
    const title = options.type === "color" ? "色卡库" : decodeURIComponent(options.title || defaultTitle);
    wx.setNavigationBarTitle({ title });
    this.setData({ url });
  },

  onShow() {
    this.consumeMeterLab();
  },

  onWebMessage(event) {
    const list = event.detail && event.detail.data ? event.detail.data : [];
    const last = Array.isArray(list) ? list[list.length - 1] : list;
    const payload = last && last.data ? last.data : last;
    if (payload && payload.action) {
      this.handleWebAction(payload).catch((err) => {
        console.error("[catalog-webview:action:error]", err);
        wx.showToast({ title: err.message || "色差仪操作失败", icon: "none" });
      });
      return;
    }
    const title = payload && (payload.title || payload.pageTitle) ? String(payload.title || payload.pageTitle) : "";
    if (title) wx.setNavigationBarTitle({ title: title === "产品库" ? "产品库" : "色卡库" });
  },

  onWebLoad(event) {
    const url = event.detail && event.detail.src ? String(event.detail.src) : this.data.url;
    const action = this.getUrlParam(url, "native_action");
    if (!action) return;
    this.handleWebAction({ action }).catch((err) => {
      console.error("[catalog-webview:native-action:error]", err);
      wx.showToast({ title: err.message || "色差仪操作失败", icon: "none" });
      this.setData({ url: this.clearNativeAction(url) });
    });
  },

  async handleWebAction(payload) {
    const action = String(payload.action || "");
    if (action === "colorMeterConnect") {
      this.openMeterBridge("connect", payload);
      return;
    }
    if (action === "colorMeterMeasure") {
      this.openMeterBridge("measure", payload);
      return;
    }
    if (action === "colorMeterDisconnect") {
      this.openMeterBridge("disconnect", payload);
    }
  },

  openMeterBridge(action, payload) {
    const currentUrl = this.data.url || "";
    const measureMode = String(
      payload && payload.measure_mode ||
      payload && payload.measureMode ||
      this.getUrlParam(currentUrl, "measure_mode") ||
      "single"
    ) === "average" ? "average" : "single";
    const deviceId = String(
      payload && payload.device_id ||
      payload && payload.deviceId ||
      this.getUrlParam(currentUrl, "meter_device_id") ||
      ""
    );
    const query = [
      `action=${encodeURIComponent(action === "connect" ? "connect" : (action === "disconnect" ? "disconnect" : "measure"))}`,
      `measure_mode=${encodeURIComponent(measureMode)}`,
    ];
    if (deviceId) query.push(`device_id=${encodeURIComponent(deviceId)}`);
    wx.navigateTo({ url: `/pages/catalog_meter_bridge/index?${query.join("&")}` });
  },

  async ensureMeterConnected() {
    if (ColorMeter.connected) return;
    if (this.data.meterConnecting) throw new Error("正在连接色差仪");
    this.setData({ meterConnecting: true });
    let scanHandler = null;
    try {
      const devices = [];
      scanHandler = (res) => {
        (res.devices || []).forEach((device) => {
          if (!device.deviceId) return;
          if (!devices.some((item) => item.deviceId === device.deviceId)) {
            devices.push(Object.assign({}, device, { name: device.name || device.localName || "未知设备" }));
          }
        });
      };
      await ColorMeter.startScan(scanHandler, 3500);
      await new Promise((resolve) => setTimeout(resolve, 3600));
      await ColorMeter.stopScan(scanHandler).catch(() => null);
      scanHandler = null;
      devices.sort((a, b) => scoreMeterDevice(b) - scoreMeterDevice(a));
      if (!devices.length) throw new Error("未搜索到色差仪，请按一下设备顶部按钮唤醒后重试");
      let lastErr = null;
      for (let i = 0; i < devices.length; i += 1) {
        try {
          await ColorMeter.connect(devices[i]);
          await retry(() => ColorMeter.getDeviceInfo(), 1).catch(() => null);
          return;
        } catch (err) {
          lastErr = err;
        }
      }
      throw lastErr || new Error("连接色差仪失败，请唤醒设备后重试");
    } finally {
      if (scanHandler) await ColorMeter.stopScan(scanHandler).catch(() => null);
      this.setData({ meterConnecting: false });
    }
  },

  returnMeterLab(lab) {
    const normalized = this.normalizeLab(lab);
    if (!normalized) {
      wx.showToast({ title: "Lab 数据异常", icon: "none" });
      return;
    }
    const nextUrl = this.withMeterLab(this.data.url, normalized);
    console.log("[catalog-webview:meter-lab]", normalized, nextUrl);
    this.setData({ url: nextUrl });
  },

  returnMeterDevice(device) {
    const normalized = this.normalizeDevice(device);
    if (!normalized.deviceId && !normalized.deviceName) return;
    const nextUrl = this.withMeterDevice(this.data.url, normalized);
    console.log("[catalog-webview:meter-device]", normalized, nextUrl);
    this.setData({ url: nextUrl });
  },

  consumeMeterLab() {
    const lab = wx.getStorageSync(METER_LAB_STORAGE_KEY);
    if (!lab || !lab.ts) return;
    wx.removeStorageSync(METER_LAB_STORAGE_KEY);
    this.returnMeterLab(lab);
  },

  normalizeLab(raw) {
    const lab = {
      L: Number(raw && (raw.L !== undefined ? raw.L : raw.l)),
      a: Number(raw && raw.a),
      b: Number(raw && raw.b),
      deviceId: raw && raw.deviceId ? String(raw.deviceId) : "",
      deviceName: raw && raw.deviceName ? String(raw.deviceName) : ""
    };
    if (!Number.isFinite(lab.L) || !Number.isFinite(lab.a) || !Number.isFinite(lab.b)) return null;
    return lab;
  },

  normalizeDevice(raw) {
    return {
      deviceId: raw && raw.deviceId ? String(raw.deviceId) : "",
      deviceName: raw && (raw.name || raw.localName || raw.deviceName) ? String(raw.name || raw.localName || raw.deviceName) : ""
    };
  },

  withMeterLab(url, lab) {
    let clean = this.removeUrlParams(this.clearNativeAction(String(url || "")), [
      "meter_l",
      "meter_a",
      "meter_b",
      "meter_ts",
      "meter_device_id",
      "meter_device_name"
    ]);
    const sep = clean.includes("?") ? "&" : "?";
    const query = [
      `meter_l=${encodeURIComponent(Number(lab.L).toFixed(4))}`,
      `meter_a=${encodeURIComponent(Number(lab.a).toFixed(4))}`,
      `meter_b=${encodeURIComponent(Number(lab.b).toFixed(4))}`,
      `meter_device_id=${encodeURIComponent(lab.deviceId || "")}`,
      `meter_device_name=${encodeURIComponent(lab.deviceName || "")}`,
      `meter_ts=${Date.now()}`
    ].join("&");
    return `${clean}${sep}${query}`;
  },

  withMeterDevice(url, device) {
    const clean = this.removeUrlParams(this.clearNativeAction(String(url || "")), [
      "meter_device_id",
      "meter_device_name",
      "meter_device_ts"
    ]);
    const sep = clean.includes("?") ? "&" : "?";
    const query = [
      `meter_device_id=${encodeURIComponent(device.deviceId || "")}`,
      `meter_device_name=${encodeURIComponent(device.deviceName || "")}`,
      `meter_device_ts=${Date.now()}`
    ].join("&");
    return `${clean}${sep}${query}`;
  },

  getUrlParam(url, key) {
    const query = String(url || "").split("?")[1] || "";
    const pairs = query.split("&").filter(Boolean);
    for (let i = 0; i < pairs.length; i += 1) {
      const parts = pairs[i].split("=");
      if (decodeURIComponent(parts[0] || "") === key) return decodeURIComponent(parts.slice(1).join("=") || "");
    }
    return "";
  },

  clearNativeAction(url) {
    return this.removeUrlParams(url, ["native_action", "native_ts"]);
  },

  removeUrlParams(url, keys) {
    const text = String(url || "");
    const hashIndex = text.indexOf("#");
    const hash = hashIndex >= 0 ? text.slice(hashIndex) : "";
    const withoutHash = hashIndex >= 0 ? text.slice(0, hashIndex) : text;
    const [base, query = ""] = withoutHash.split("?");
    if (!query) return text;
    const removeKeys = new Set(keys || []);
    const next = query
      .split("&")
      .filter(Boolean)
      .filter((pair) => !removeKeys.has(decodeURIComponent((pair.split("=")[0] || ""))))
      .join("&");
    return (next ? `${base}?${next}` : base) + hash;
  },

  buildCatalogUrl(options) {
    const decodeOption = (value) => decodeURIComponent(String(value || "")).trim();
    const type = options.type === "color" ? "color" : (options.type === "product" ? "product" : "");
    if (type) {
      const env = decodeOption(options.env || "");
      const envBaseUrls = config.catalogH5BaseUrls || {};
      const baseUrl = (
        decodeOption(options.h5_base_url || options.base_url || "") ||
        String(env ? (envBaseUrls[env] || "") : "").trim() ||
        String(config.catalogH5BaseUrl || "").trim() ||
        String(config.baseUrl || "").trim()
      ).replace(/\/+$/, "");
      const path = config.catalogH5Path || "/catalog";
      const token = decodeOption(options.token || options.catalog_token || options.access_token || "");
      const query = [`type=${encodeURIComponent(type)}`];
      if (token) query.push(`token=${encodeURIComponent(token)}`);
      return `${baseUrl}${path}?${query.join("&")}`;
    }
    return decodeURIComponent(options.url || "").replace(/&amp;/g, "&");
  }
});
