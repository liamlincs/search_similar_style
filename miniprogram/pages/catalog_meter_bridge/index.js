const { ColorMeter } = require("../../utils/color_meter_bluetooth");
const { retry } = require("../../utils/color_meter_utils");
const config = require("../../utils/config");

const METER_LAB_STORAGE_KEY = "catalog_meter_bridge_lab";
const METER_DEVICE_STORAGE_KEY = "catalog_meter_devices";
const AVERAGE_SAMPLE_COUNT = 3;

function scoreMeterDevice(device) {
  const name = `${device && (device.name || device.localName) || ""}`.toLowerCase();
  let score = 0;
  if (/color|colour|meter|spectro|colormeter|色差|测色|颜色/.test(name)) score += 100;
  if (/pt|rm|cr|wr|ys|nh|3nh|fru|bn|cm/.test(name)) score += 20;
  return score + name.length;
}

function getMeterErrorMessage(err, fallback) {
  const message = ColorMeter.getErrorMessage ? ColorMeter.getErrorMessage(err) : "";
  if (message) return message;
  return (err && (err.message || err.errMsg)) || fallback || "色差仪操作失败";
}

function openBluetoothSetting() {
  if (wx.openAppAuthorizeSetting) {
    wx.openAppAuthorizeSetting({ fail: () => wx.openSetting && wx.openSetting() });
  } else if (wx.openSetting) {
    wx.openSetting();
  }
}

Page({
  data: {
    status: "正在准备色差仪...",
    devices: [],
    scanning: false,
    showDeviceList: false,
    connectingDeviceId: "",
    connectedDeviceId: "",
    connectedDeviceName: "",
    disconnecting: false,
    showConnectedActions: false,
  },

  onLoad(options) {
    const params = options || {};
    this.bridgeOptions = params;
    const action = params.action === "connect" ? "connect" : (params.action === "disconnect" ? "disconnect" : "measure");
    this.setData({ showConnectedActions: action === "connect" });
    this.run(action, params.device_id || "").catch((err) => {
      console.error("[catalog-meter-bridge:error]", err);
      const message = getMeterErrorMessage(err);
      this.setData({ status: message });
      if (ColorMeter.shouldOpenSetting && ColorMeter.shouldOpenSetting(err) && (wx.openAppAuthorizeSetting || wx.openSetting)) {
        wx.showModal({
          title: "无法使用蓝牙",
          content: message,
          confirmText: "去设置",
          cancelText: "返回",
          success: (res) => {
            if (res.confirm) openBluetoothSetting();
            else wx.navigateBack({ delta: 1 });
          },
        });
      } else {
        wx.showToast({ title: message, icon: "none" });
        setTimeout(() => wx.navigateBack({ delta: 1 }), 1800);
      }
    });
  },

  onUnload() {
    this.stopDeviceScan();
    this.stopButtonMeasurement();
    if (this.pickReject) {
      this.pickReject(new Error("已取消连接"));
      this.pickResolve = null;
      this.pickReject = null;
    }
  },

  async run(action, deviceId) {
    if (action === "disconnect") {
      this.syncConnectedDeviceState();
      await this.disconnectDevice();
      return;
    }
    await this.ensureMeterConnected(deviceId);
    if (action === "connect") {
      await this.publishDevice();
      this.syncConnectedDeviceState();
      this.setData({ status: "色差仪已连接，可断开设备或返回H5" });
      wx.showToast({ title: "色差仪已连接", icon: "none" });
      return;
    }
    this.syncConnectedDeviceState();
    const measureMode = this.safeDecode(this.bridgeOptions.measure_mode || "") === "average" ? "average" : "single";
    const lab = await this.waitForButtonLab(measureMode);
    await this.publishLab(lab);
    this.setData({ status: `测量完成 L:${lab.L.toFixed(2)} a:${lab.a.toFixed(2)} b:${lab.b.toFixed(2)}，正在返回...` });
    setTimeout(() => wx.navigateBack({ delta: 1 }), 300);
  },

  async waitForButtonLab(measureMode) {
    const targetCount = measureMode === "average" ? AVERAGE_SAMPLE_COUNT : 1;
    const samples = [];
    this.setData({
      status: targetCount > 1
        ? `请点击测量按钮并按设备按钮测量，需获取 ${targetCount} 次`
        : "请点击测量按钮并按设备按钮测量",
    });
    while (samples.length < targetCount) {
      const lab = await this.waitForSingleButtonLab(samples.length + 1, targetCount);
      samples.push(lab);
      await this.publishProgress(samples.length, targetCount).catch((err) => {
        console.warn("[catalog-meter-bridge:post-progress-fallback]", err);
      });
      if (samples.length === targetCount) {
        this.setData({ status: `已获取 ${samples.length}/${targetCount} 次，正在生成结果...` });
        await this.wait(650);
      }
      if (samples.length < targetCount) {
        this.setData({ status: `已获取 ${samples.length}/${targetCount} 次，请继续按设备按钮测量` });
      }
    }
    if (samples.length === 1) return samples[0];
    const sum = samples.reduce((acc, item) => ({
      L: acc.L + item.L,
      a: acc.a + item.a,
      b: acc.b + item.b,
    }), { L: 0, a: 0, b: 0 });
    return {
      L: sum.L / samples.length,
      a: sum.a / samples.length,
      b: sum.b / samples.length,
      ts: Date.now(),
      sampleCount: samples.length,
    };
  },

  waitForSingleButtonLab(index, total) {
    this.stopButtonMeasurement("已开始新的测量");
    return new Promise((resolve, reject) => {
      let settled = false;
      const cleanup = () => {
        if (this.buttonMeasureTimer) {
          clearTimeout(this.buttonMeasureTimer);
          this.buttonMeasureTimer = null;
        }
        if (this.buttonMeasureHandler) {
          ColorMeter.unsubscribe(this.buttonMeasureHandler);
          this.buttonMeasureHandler = null;
        }
        this.buttonMeasureReject = null;
      };
      this.buttonMeasureReject = (err) => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(err);
      };
      this.buttonMeasureHandler = async (event) => {
        if (!event || event.type !== "measure" || settled) return;
        settled = true;
        cleanup();
        try {
          this.setData({ status: total > 1 ? `正在读取第 ${index}/${total} 次 Lab...` : "正在读取 Lab..." });
          await this.wait(120);
          const lab = this.normalizeLab(await retry(() => ColorMeter.getLab(event.detail && event.detail.mode || 0), 2));
          resolve(lab);
        } catch (err) {
          reject(err);
        }
      };
      ColorMeter.subscribe(this.buttonMeasureHandler);
      this.buttonMeasureTimer = setTimeout(() => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(new Error("未收到设备按键测量结果，请确认色差仪已唤醒后重试"));
      }, 60000);
    });
  },

  stopButtonMeasurement(message) {
    if (this.buttonMeasureReject) {
      const reject = this.buttonMeasureReject;
      this.buttonMeasureReject = null;
      reject(new Error(message || "测量已取消"));
      return;
    }
    if (this.buttonMeasureTimer) {
      clearTimeout(this.buttonMeasureTimer);
      this.buttonMeasureTimer = null;
    }
    if (this.buttonMeasureHandler) {
      ColorMeter.unsubscribe(this.buttonMeasureHandler);
      this.buttonMeasureHandler = null;
    }
  },

  wait(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  },

  normalizeLab(raw) {
    const lab = {
      L: Number(raw && (raw.L !== undefined ? raw.L : raw.l)),
      a: Number(raw && raw.a),
      b: Number(raw && raw.b),
      ts: Date.now(),
    };
    if (!Number.isFinite(lab.L) || !Number.isFinite(lab.a) || !Number.isFinite(lab.b)) {
      throw new Error("色差仪返回 Lab 数据异常");
    }
    return lab;
  },

  async publishLab(lab) {
    const device = ColorMeter.connected || {};
    this.rememberDevice(device);
    const payload = Object.assign({}, lab, {
      deviceId: device.deviceId || "",
      deviceName: device.name || device.localName || "",
      sampleCount: lab.sampleCount || 1,
    });
    const posted = await this.postNativeReading(Object.assign({}, payload, { event: "measure" })).catch((err) => {
      console.warn("[catalog-meter-bridge:post-lab-fallback]", err);
      return false;
    });
    if (posted) {
      wx.removeStorageSync(METER_LAB_STORAGE_KEY);
      return;
    }
    wx.setStorageSync(METER_LAB_STORAGE_KEY, payload);
    const pages = getCurrentPages();
    const prevPage = pages.length >= 2 ? pages[pages.length - 2] : null;
    if (prevPage && typeof prevPage.returnMeterLab === "function") {
      wx.removeStorageSync(METER_LAB_STORAGE_KEY);
      prevPage.returnMeterLab(payload);
    }
  },

  async publishProgress(progressCount, totalCount) {
    const device = ColorMeter.connected || {};
    this.rememberDevice(device);
    await this.postNativeReading({
      event: "measure_progress",
      progressCount,
      totalCount,
      sampleCount: totalCount,
      deviceId: device.deviceId || "",
      deviceName: device.name || device.localName || "",
      ts: Date.now(),
    });
  },

  async publishDevice() {
    const device = ColorMeter.connected || {};
    const payload = {
      deviceId: device.deviceId || "",
      deviceName: device.name || device.localName || "",
    };
    this.rememberDevice(device);
    const posted = await this.postNativeReading(Object.assign({}, payload, { event: "connect" })).catch((err) => {
      console.warn("[catalog-meter-bridge:post-device-fallback]", err);
      return false;
    });
    if (posted) return;
    const pages = getCurrentPages();
    const prevPage = pages.length >= 2 ? pages[pages.length - 2] : null;
    if (prevPage && typeof prevPage.returnMeterDevice === "function") {
      prevPage.returnMeterDevice(payload);
    }
  },

  async publishDisconnect() {
    const deviceId = this.data.connectedDeviceId || "";
    const deviceName = this.data.connectedDeviceName || "";
    await this.postNativeReading({
      event: "disconnect",
      deviceId,
      deviceName,
      ts: Date.now(),
    }).catch((err) => {
      console.warn("[catalog-meter-bridge:post-disconnect-fallback]", err);
      return false;
    });
  },

  postNativeReading(payload) {
    const options = this.bridgeOptions || {};
    const apiBase = this.safeDecode(options.api_base || config.baseUrl || "").replace(/\/+$/, "");
    const userTag = this.safeDecode(options.user_tag || "");
    const token = this.safeDecode(options.token || "");
    if (!apiBase || !userTag) return Promise.resolve(false);
    const body = {
      user_tag: userTag,
      request_id: this.safeDecode(options.request_id || ""),
      event: payload.event || "measure",
      L: payload.L,
      a: payload.a,
      b: payload.b,
      sample_count: payload.sampleCount || 1,
      progress_count: payload.progressCount || 0,
      total_count: payload.totalCount || 0,
      device_id: payload.deviceId || "",
      device_name: payload.deviceName || "",
      ts: payload.ts || Date.now(),
    };
    return new Promise((resolve) => {
      wx.request({
        url: `${apiBase}/api/v1/color-card/native-meter/reading`,
        method: "POST",
        header: Object.assign(
          { "content-type": "application/json" },
          token ? { "X-Catalog-Token": token } : {}
        ),
        data: body,
        success: (res) => resolve(res.statusCode >= 200 && res.statusCode < 300),
        fail: () => resolve(false),
      });
    });
  },

  safeDecode(value) {
    const text = String(value || "");
    try {
      return decodeURIComponent(text);
    } catch (_) {
      return text;
    }
  },

  rememberDevice(device) {
    const clean = {
      deviceId: device && device.deviceId ? String(device.deviceId) : "",
      name: device && (device.name || device.localName) ? String(device.name || device.localName) : "",
      ts: Date.now(),
    };
    if (!clean.deviceId && !clean.name) return;
    const rows = wx.getStorageSync(METER_DEVICE_STORAGE_KEY) || [];
    const list = Array.isArray(rows) ? rows : [];
    const next = list.filter((item) => item.deviceId !== clean.deviceId && item.name !== clean.name);
    next.unshift(clean);
    wx.setStorageSync(METER_DEVICE_STORAGE_KEY, next.slice(0, 8));
  },

  syncConnectedDeviceState() {
    const device = ColorMeter.connected || {};
    this.setData({
      connectedDeviceId: device.deviceId || "",
      connectedDeviceName: device.name || device.localName || "",
      showDeviceList: false,
    });
  },

  async disconnectDevice() {
    if (this.data.disconnecting) return;
    this.setData({ disconnecting: true, status: "正在断开色差仪..." });
    try {
      this.stopDeviceScan();
      this.stopButtonMeasurement();
      await this.publishDisconnect();
      await ColorMeter.disconnect().catch(() => null);
      this.setData({
        status: "色差仪已断开，可重新扫描连接",
        connectedDeviceId: "",
        connectedDeviceName: "",
        connectingDeviceId: "",
        devices: [],
        showDeviceList: true,
      });
      wx.showToast({ title: "已断开", icon: "none" });
    } catch (err) {
      const message = getMeterErrorMessage(err, "断开设备失败");
      this.setData({ status: message });
      wx.showToast({ title: message, icon: "none" });
    } finally {
      this.setData({ disconnecting: false });
    }
  },

  backToH5() {
    this.stopButtonMeasurement("已返回H5");
    wx.navigateBack({ delta: 1 });
  },

  async ensureMeterConnected(deviceId) {
    if (ColorMeter.connected) return;
    this.setData({ status: "正在请求蓝牙权限..." });
    await ColorMeter.init();
    let knownDeviceId = this.safeDecode(deviceId || "");
    if (knownDeviceId) {
      const rows = wx.getStorageSync(METER_DEVICE_STORAGE_KEY) || [];
      const saved = Array.isArray(rows) ? rows.find((item) => item.deviceId === knownDeviceId) : null;
      const knownDevice = {
        deviceId: knownDeviceId,
        name: saved && saved.name ? saved.name : "色差仪",
      };
      this.setData({ status: `正在连接 ${knownDevice.name || "色差仪"}...` });
      try {
        await ColorMeter.connect(knownDevice);
        this.rememberDevice(knownDevice);
        this.syncConnectedDeviceState();
        await retry(() => ColorMeter.getDeviceInfo(), 1).catch(() => null);
        return;
      } catch (err) {
        console.warn("[catalog-meter-bridge:direct-connect-fallback]", err);
        await ColorMeter.disconnect().catch(() => null);
        this.setData({ status: "上次连接设备不可用，正在重新搜索附近设备..." });
      }
    }
    while (!ColorMeter.connected) {
      const picked = await this.pickMeterDevice(knownDeviceId);
      this.setData({ status: `正在连接 ${picked.name || "色差仪"}...`, connectingDeviceId: picked.deviceId });
      try {
        await ColorMeter.connect(picked);
        this.rememberDevice(picked);
        this.syncConnectedDeviceState();
        await retry(() => ColorMeter.getDeviceInfo(), 1).catch(() => null);
      } catch (err) {
        console.warn("[catalog-meter-bridge:selected-connect-failed]", picked, err);
        const message = getMeterErrorMessage(err, "连接失败，请选择其他设备或重新扫描");
        await ColorMeter.disconnect().catch(() => null);
        this.setData({
          status: message,
          connectingDeviceId: "",
          showDeviceList: true,
        });
        knownDeviceId = "";
        continue;
      } finally {
        this.setData({ connectingDeviceId: "" });
      }
    }
  },

  pickMeterDevice(preferredDeviceId) {
    this.stopDeviceScan();
    this.pickPreferredDeviceId = preferredDeviceId || "";
    this.setData({
      devices: [],
      showDeviceList: true,
      status: "正在搜索附近蓝牙设备，请按一下设备顶部按钮唤醒",
    });
    this.startDeviceScan();
    return new Promise((resolve, reject) => {
      this.pickResolve = resolve;
      this.pickReject = reject;
    });
  },

  startDeviceScan() {
    this.stopDeviceScan();
    this.setData({ scanning: true });
    this.scanHandler = (res) => {
      const next = (res.devices || [])
        .filter((device) => device && device.deviceId)
        .map((device) => this.normalizeScanDevice(device));
      if (!next.length) return;
      const byId = new Map((this.data.devices || []).map((item) => [item.deviceId, item]));
      next.forEach((device) => {
        const prev = byId.get(device.deviceId) || {};
        byId.set(device.deviceId, Object.assign({}, prev, device));
      });
      const devices = Array.from(byId.values()).sort((a, b) => scoreMeterDevice(b) - scoreMeterDevice(a));
      this.setData({ devices });
    };
    ColorMeter.startScan(this.scanHandler, 0).catch((err) => {
      const message = getMeterErrorMessage(err, "扫描蓝牙设备失败");
      this.setData({ status: message, scanning: false });
      if (this.pickReject) {
        this.pickReject(new Error(message));
        this.pickResolve = null;
        this.pickReject = null;
      }
    });
    this.scanTimer = setTimeout(() => {
      this.stopDeviceScan();
      if (!(this.data.devices || []).length) {
        this.setData({ status: "未搜索到设备，请按一下设备顶部按钮唤醒后重新扫描" });
      } else {
        this.setData({ status: "请选择要连接的设备" });
      }
    }, 8000);
  },

  stopDeviceScan() {
    if (this.scanTimer) {
      clearTimeout(this.scanTimer);
      this.scanTimer = null;
    }
    if (this.scanHandler) {
      try {
        ColorMeter.stopScan(this.scanHandler);
      } catch (_) {}
      this.scanHandler = null;
    }
    if (this.data && this.data.scanning) this.setData({ scanning: false });
  },

  normalizeScanDevice(device) {
    const name = device.name || device.localName || "未知设备";
    const rssi = Number(device.RSSI);
    return {
      deviceId: device.deviceId,
      name,
      localName: device.localName || "",
      RSSI: Number.isFinite(rssi) ? rssi : "",
      score: scoreMeterDevice(device),
      preferred: this.pickPreferredDeviceId && device.deviceId === this.pickPreferredDeviceId,
    };
  },

  chooseDevice(event) {
    if (this.data.connectingDeviceId) return;
    const deviceId = event.currentTarget.dataset.deviceId;
    const device = (this.data.devices || []).find((item) => item.deviceId === deviceId);
    if (!device || !this.pickResolve) return;
    this.stopDeviceScan();
    this.pickResolve(device);
    this.pickResolve = null;
    this.pickReject = null;
  },

  rescanDevices() {
    this.setData({ devices: [], status: "正在重新扫描附近蓝牙设备..." });
    this.startDeviceScan();
  },
});
