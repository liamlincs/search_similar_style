const { fetchColorCardLibraries, matchColorCards, saveColorCard } = require("../../utils/api");
const { ColorMeter } = require("../../utils/color_meter_bluetooth");
const { labToHex, retry } = require("../../utils/color_meter_utils");

function showBluetoothError(err) {
  const message = ColorMeter.getErrorMessage ? ColorMeter.getErrorMessage(err) : "蓝牙不可用或未授权";
  const openSetting = () => {
    if (wx.openAppAuthorizeSetting) {
      wx.openAppAuthorizeSetting({ fail: () => wx.openSetting && wx.openSetting() });
    } else if (wx.openSetting) {
      wx.openSetting();
    }
  };
  const canOpenSetting = ColorMeter.shouldOpenSetting ? ColorMeter.shouldOpenSetting(err) : ColorMeter.isPermissionError && ColorMeter.isPermissionError(err);
  if (canOpenSetting && (wx.openAppAuthorizeSetting || wx.openSetting)) {
    wx.showModal({
      title: "无法使用蓝牙",
      content: message,
      confirmText: "去设置",
      success: (res) => {
        if (res.confirm) openSetting();
      },
    });
  } else {
    wx.showToast({ title: message, icon: "none", duration: 3000 });
  }
  return message;
}

function scoreMeterDevice(device) {
  const name = String((device && (device.name || device.localName)) || "").toLowerCase();
  let score = 0;
  if (/color|colour|meter|spectro|colormeter|色差|测色|颜色/.test(name)) score += 100;
  if (/iphone|ipad|macbook|watch|airpods/.test(name)) score -= 50;
  const rssi = Number(device && device.RSSI);
  if (Number.isFinite(rssi)) score += Math.max(-20, Math.min(20, Math.round((rssi + 80) / 2)));
  return score;
}

function formatNumber(value, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "";
}

function textColorForLab(lValue) {
  return Number(lValue || 0) < 55 ? "#FFFFFF" : "#0F172A";
}

function normalizeHex(hex) {
  return String(hex || "CCCCCC").replace(/^#/, "").toUpperCase();
}

function firstKnownPrefix(name) {
  const text = String(name || "");
  if (text.indexOf("彩龙") >= 0) return "彩龙";
  if (text.indexOf("国彩") >= 0) return "国彩";
  if (text.indexOf("恩盛") >= 0) return "恩盛";
  return "";
}

function incrementNumericText(raw) {
  const text = String(raw || "").trim();
  if (!text) return "";
  const next = String(Number(text) + 1);
  return text.length > next.length ? next.padStart(text.length, "0") : next;
}

Page({
  data: {
    mode: "query",
    libraries: [],
    libraryPickerNames: ["全部色卡"],
    libraryManagePickerNames: [],
    selectedLibraryIndex: 0,
    selectedLibraryId: "",
    selectedLibraryName: "全部色卡",
    manageLibraryIndex: 0,
    manageLibraryId: "",
    manageLibraryName: "",
    newLibraryName: "",

    meterPanelOpen: false,
    meterScanning: false,
    meterConnecting: false,
    meterMeasuring: false,
    meterDevices: [],
    meterDeviceName: "",
    meterStatus: "未连接色差仪",

    lab: null,
    labLText: "",
    labAText: "",
    labBText: "",
    currentHex: "CCCCCC",
    currentColor: "#CCCCCC",
    currentTextColor: "#0F172A",
    manualL: "",
    manualA: "",
    manualB: "",

    namePrefix: "",
    nameNumber: "",
    nameSuffix: "",
    colorName: "",
    note: "",
    canSave: false,
    saving: false,

    colorMatching: false,
    colorMatches: [],
    colorMatchError: "",
  },

  onLoad(options) {
    const initialMode = options && options.mode === "manage" ? "manage" : "query";
    this.setData({ mode: initialMode });
    this._meterListener = (ev) => {
      if (ev.type === "disconnect") {
        this.setData({ meterStatus: "色差仪已断开", meterDeviceName: "" });
      }
      if (ev.type === "connected") {
        const name = (ColorMeter.connected && ColorMeter.connected.name) || "已连接设备";
        this.setData({ meterStatus: "色差仪已连接", meterDeviceName: name });
      }
    };
    ColorMeter.subscribe(this._meterListener);
    if (ColorMeter.connected) {
      this.setData({
        meterStatus: "色差仪已连接",
        meterDeviceName: ColorMeter.connected.name || "已连接设备",
      });
    }
    this.loadLibraries();
  },

  onUnload() {
    if (this._meterScanHandler) {
      ColorMeter.stopScan(this._meterScanHandler);
      this._meterScanHandler = null;
    }
    if (this._meterListener) {
      ColorMeter.unsubscribe(this._meterListener);
      this._meterListener = null;
    }
  },

  switchMode(e) {
    const mode = e.currentTarget.dataset.mode;
    if (mode === "query" || mode === "manage") this.setData({ mode });
  },

  async loadLibraries() {
    try {
      const res = await fetchColorCardLibraries();
      const currentManageId = String(this.data.manageLibraryId || "");
      const currentManageName = String(this.data.manageLibraryName || "");
      const libraries = (res.libraries || [])
        .map((item) => ({
          id: String(item.id || ""),
          name: String(item.name || ""),
          color_count: Number(item.color_count || 0),
        }))
        .filter((item) => item.id && item.name);
      const manageNames = libraries.length ? libraries.map((item) => `${item.name} (${item.color_count})`) : ["暂无色卡库"];
      let manageIndex = libraries.findIndex((item) => item.id === currentManageId);
      if (manageIndex < 0 && currentManageName) {
        manageIndex = libraries.findIndex((item) => item.name === currentManageName);
      }
      if (manageIndex < 0) manageIndex = 0;
      const first = libraries[manageIndex] || null;
      this.setData({
        libraries,
        libraryPickerNames: ["全部色卡"].concat(libraries.map((item) => `${item.name} (${item.color_count})`)),
        libraryManagePickerNames: manageNames,
        manageLibraryIndex: manageIndex,
        manageLibraryId: first ? first.id : "",
        manageLibraryName: first ? first.name : "",
        namePrefix: this.data.namePrefix || firstKnownPrefix(first ? first.name : ""),
      }, () => this.refreshColorName());
    } catch (err) {
      wx.showToast({ title: err.message || "加载色卡库失败", icon: "none" });
    }
  },

  onLibraryPickerChange(e) {
    const index = Number(e.detail.value || 0);
    const lib = index > 0 ? this.data.libraries[index - 1] : null;
    this.setData({
      selectedLibraryIndex: index,
      selectedLibraryId: lib ? lib.id : "",
      selectedLibraryName: lib ? lib.name : "全部色卡",
    });
    if (this.data.lab) this.matchLab(this.data.lab);
  },

  onManageLibraryChange(e) {
    const index = Number(e.detail.value || 0);
    const lib = this.data.libraries[index] || null;
    this.setData({
      manageLibraryIndex: index,
      manageLibraryId: lib ? lib.id : "",
      manageLibraryName: lib ? lib.name : "",
      namePrefix: this.data.namePrefix || firstKnownPrefix(lib ? lib.name : ""),
    }, () => {
      this.refreshColorName();
      this.refreshCanSave();
    });
  },

  onManualInput(e) {
    const key = e.currentTarget.dataset.key;
    if (!key) return;
    this.setData({ [key]: e.detail.value || "" });
  },

  onFieldInput(e) {
    const key = e.currentTarget.dataset.key;
    if (!key) return;
    this.setData({ [key]: e.detail.value || "" }, () => this.refreshCanSave());
  },

  onTemplateInput(e) {
    const key = e.currentTarget.dataset.key;
    if (!key) return;
    this.setData({ [key]: e.detail.value || "" }, () => this.refreshColorName());
  },

  refreshColorName() {
    const name = `${this.data.namePrefix || ""}${this.data.nameNumber || ""}${this.data.nameSuffix || ""}`.trim();
    this.setData({ colorName: name }, () => this.refreshCanSave());
  },

  refreshCanSave() {
    const lab = this.data.lab;
    const libraryName = String(this.data.newLibraryName || this.data.manageLibraryName || "").trim();
    const colorName = String(this.data.colorName || "").trim();
    this.setData({ canSave: !!(lab && libraryName && colorName) });
  },

  applyLab(lab) {
    const normalized = {
      L: Number(lab.L),
      a: Number(lab.a),
      b: Number(lab.b),
    };
    const hex = labToHex(normalized);
    this.setData({
      lab: normalized,
      labLText: formatNumber(normalized.L),
      labAText: formatNumber(normalized.a),
      labBText: formatNumber(normalized.b),
      currentHex: hex,
      currentColor: `#${hex}`,
      currentTextColor: textColorForLab(normalized.L),
      manualL: formatNumber(normalized.L),
      manualA: formatNumber(normalized.a),
      manualB: formatNumber(normalized.b),
    }, () => this.refreshCanSave());
    this.matchLab(normalized);
  },

  async matchManualColor() {
    const lab = {
      L: Number(this.data.manualL),
      a: Number(this.data.manualA),
      b: Number(this.data.manualB),
    };
    if (![lab.L, lab.a, lab.b].every(Number.isFinite)) {
      wx.showToast({ title: "请先输入 Lab 数值", icon: "none" });
      return;
    }
    this.applyLab(lab);
  },

  async matchLab(lab) {
    if (!lab) return;
    this.setData({ colorMatching: true, colorMatchError: "" });
    try {
      const res = await matchColorCards({
        L: lab.L,
        a: lab.a,
        b: lab.b,
        library_id: this.data.selectedLibraryId,
        limit: 12,
      });
      const matches = (res.matches || []).map((item) => {
        const lValue = Number(item.l || 0);
        const hex = normalizeHex(item.hex);
        const color = textColorForLab(lValue);
        return {
          id: item.id,
          name: String(item.name || ""),
          library_name: String(item.library_name || ""),
          hex,
          deltaText: Number(item.delta_e_00 || 0).toFixed(2),
          labText: `L ${Number(item.l).toFixed(1)} / a ${Number(item.a).toFixed(1)} / b ${Number(item.b).toFixed(1)}`,
          itemStyle: `background-color: #${hex}; color: ${color};`,
        };
      });
      this.setData({ colorMatches: matches });
    } catch (err) {
      this.setData({ colorMatchError: err.message || "色卡匹配失败", colorMatches: [] });
    } finally {
      this.setData({ colorMatching: false });
    }
  },

  toggleMeterPanel() {
    const nextOpen = !this.data.meterPanelOpen;
    this.setData({ meterPanelOpen: nextOpen });
    if (!nextOpen && this._meterScanHandler) {
      ColorMeter.stopScan(this._meterScanHandler);
      this._meterScanHandler = null;
      this.setData({ meterScanning: false });
    }
    if (nextOpen && !ColorMeter.connected && !this.data.meterScanning) this.startMeterScan();
  },

  openMeterPanelOnly() {
    if (!this.data.meterPanelOpen) {
      this.setData({ meterPanelOpen: true });
    }
  },

  async startMeterScan() {
    if (this.data.meterScanning) return;
    try {
      await ColorMeter.init();
      if (this._meterScanHandler) ColorMeter.stopScan(this._meterScanHandler);
      this._meterScanHandler = (res) => {
        const found = (res.devices || []).filter((device) => device.name || device.localName);
        if (!found.length) return;
        const merged = [...this.data.meterDevices];
        found.forEach((device) => {
          const normalized = Object.assign({}, device, { name: device.name || device.localName || "" });
          const idx = merged.findIndex((item) => item.deviceId === normalized.deviceId);
          if (idx >= 0) merged[idx] = normalized;
          else merged.push(normalized);
        });
        merged.sort((a, b) => scoreMeterDevice(b) - scoreMeterDevice(a));
        this.setData({ meterDevices: merged });
      };
      this.setData({ meterScanning: true, meterDevices: [], meterStatus: "正在扫描色差仪" });
      await ColorMeter.startScan(this._meterScanHandler, 10000);
      setTimeout(() => {
        if (this.data.meterScanning) {
          this.setData({ meterScanning: false, meterStatus: ColorMeter.connected ? "色差仪已连接" : "扫描完成" });
        }
      }, 10200);
    } catch (err) {
      console.error("[color-meter:scan:error]", err);
      const message = showBluetoothError(err);
      this.setData({ meterScanning: false, meterStatus: message });
    }
  },

  async connectMeter(e) {
    const device = e.currentTarget.dataset.device;
    if (!device || this.data.meterConnecting) return;
    this.setData({ meterConnecting: true, meterStatus: "正在连接色差仪" });
    try {
      if (this._meterScanHandler) {
        ColorMeter.stopScan(this._meterScanHandler);
        this._meterScanHandler = null;
      }
      await ColorMeter.connect(device);
      await retry(() => ColorMeter.getDeviceInfo(), 1).catch(() => null);
      this.setData({
        meterConnecting: false,
        meterScanning: false,
        meterDeviceName: device.name || "已连接设备",
        meterStatus: "色差仪已连接",
      });
      wx.showToast({ title: "色差仪已连接", icon: "none" });
    } catch (err) {
      await ColorMeter.disconnect().catch(() => null);
      this.setData({ meterConnecting: false, meterStatus: "连接失败" });
      wx.showToast({ title: "连接失败", icon: "none" });
    }
  },

  async disconnectMeter() {
    await ColorMeter.disconnect().catch(() => null);
    this.setData({ meterDeviceName: "", meterStatus: "未连接色差仪" });
  },

  async measureColor() {
    if (this.data.meterMeasuring) return;
    if (!ColorMeter.connected) {
      this.openMeterPanelOnly();
      wx.showToast({ title: "请先连接色差仪", icon: "none" });
      return;
    }
    this.setData({ meterMeasuring: true, meterStatus: "正在测量" });
    try {
      const lab = await retry(() => ColorMeter.measureAndGetLab(), 2);
      this.applyLab(lab);
      this.setData({ meterStatus: "测量完成" });
    } catch (err) {
      this.setData({ meterStatus: "测量失败" });
      wx.showToast({ title: "测量失败", icon: "none" });
    } finally {
      this.setData({ meterMeasuring: false });
    }
  },

  async saveCurrentColor() {
    if (!this.data.canSave || this.data.saving) return;
    const lab = this.data.lab;
    const libraryName = String(this.data.newLibraryName || this.data.manageLibraryName || "").trim();
    const name = String(this.data.colorName || "").trim();
    this.setData({ saving: true });
    try {
      const res = await saveColorCard({
        library_id: this.data.newLibraryName ? "" : this.data.manageLibraryId,
        library_name: libraryName,
        name,
        L: lab.L,
        a: lab.a,
        b: lab.b,
        hex: this.data.currentHex,
        note: this.data.note,
      });
      wx.showToast({ title: `已保存：${name}`, icon: "none" });
      const nextNumber = incrementNumericText(this.data.nameNumber);
      const savedCard = (res && res.card) || {};
      this.setData({
        nameNumber: nextNumber,
        nameSuffix: "",
        note: "",
        newLibraryName: "",
        manageLibraryId: savedCard.library_id || this.data.manageLibraryId,
        manageLibraryName: savedCard.library_name || libraryName,
      }, () => this.refreshColorName());
      await this.loadLibraries();
      await this.matchLab(lab);
    } catch (err) {
      wx.showToast({ title: err.message || "保存失败", icon: "none" });
    } finally {
      this.setData({ saving: false });
    }
  },
});
