const { recolorUpload, recolorAiUpload, fetchColorCardLibraries, matchColorCards } = require("../../utils/api");
const config = require("../../utils/config");
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

function toAbsolute(pathOrUrl) {
  if (!pathOrUrl) return "";
  if (/^https?:\/\//.test(pathOrUrl)) return pathOrUrl;
  return `${config.baseUrl}${pathOrUrl}`;
}

function getTouchPoint(t) {
  if (!t) return null;
  const x = Number(t.clientX ?? t.pageX ?? t.x);
  const y = Number(t.clientY ?? t.pageY ?? t.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y };
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function toHex2(n) {
  return clamp(Number(n) || 0, 0, 255).toString(16).padStart(2, "0");
}

function rgbToHex(r, g, b) {
  return `${toHex2(r)}${toHex2(g)}${toHex2(b)}`.toUpperCase();
}

function hsvToRgb(h, s, v) {
  const hh = ((h % 360) + 360) % 360;
  const ss = clamp(s, 0, 1);
  const vv = clamp(v, 0, 1);
  const c = vv * ss;
  const x = c * (1 - Math.abs(((hh / 60) % 2) - 1));
  const m = vv - c;
  let r1 = 0;
  let g1 = 0;
  let b1 = 0;
  if (hh < 60) [r1, g1, b1] = [c, x, 0];
  else if (hh < 120) [r1, g1, b1] = [x, c, 0];
  else if (hh < 180) [r1, g1, b1] = [0, c, x];
  else if (hh < 240) [r1, g1, b1] = [0, x, c];
  else if (hh < 300) [r1, g1, b1] = [x, 0, c];
  else [r1, g1, b1] = [c, 0, x];
  return {
    r: Math.round((r1 + m) * 255),
    g: Math.round((g1 + m) * 255),
    b: Math.round((b1 + m) * 255),
  };
}

function rgbToHsv(r, g, b) {
  const rr = clamp(Number(r) || 0, 0, 255) / 255;
  const gg = clamp(Number(g) || 0, 0, 255) / 255;
  const bb = clamp(Number(b) || 0, 0, 255) / 255;
  const max = Math.max(rr, gg, bb);
  const min = Math.min(rr, gg, bb);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === rr) h = ((gg - bb) / d) % 6;
    else if (max === gg) h = (bb - rr) / d + 2;
    else h = (rr - gg) / d + 4;
    h *= 60;
  }
  if (h < 0) h += 360;
  return {
    h,
    s: max === 0 ? 0 : d / max,
    v: max,
  };
}

function hexToRgb(hex) {
  const raw = String(hex || "").replace(/^#/, "");
  if (!/^[0-9a-fA-F]{6}$/.test(raw)) return null;
  return {
    r: parseInt(raw.slice(0, 2), 16),
    g: parseInt(raw.slice(2, 4), 16),
    b: parseInt(raw.slice(4, 6), 16),
  };
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

function downloadToLocal(url) {
  return new Promise((resolve) => {
    if (!url) return resolve("");
    wx.downloadFile({
      url,
      success: (res) => resolve(res.tempFilePath || ""),
      fail: () => resolve("")
    });
  });
}

function compressForUpload(filePath) {
  return new Promise((resolve) => {
    if (!filePath) return resolve(filePath);
    wx.getImageInfo({
      src: filePath,
      success: (info) => {
        const cfg = config.recolorUpload || {};
        const maxSide = Number(cfg.maxSide || 1600);
        const quality = Number(cfg.quality || 82);
        const w = Number(info.width || 0);
        const h = Number(info.height || 0);
        if (!w || !h || Math.max(w, h) <= maxSide) {
          resolve(filePath);
          return;
        }
        const scale = maxSide / Math.max(w, h);
        const targetW = Math.max(1, Math.round(w * scale));
        const targetH = Math.max(1, Math.round(h * scale));
        const ext = filePath.toLowerCase().split(".").pop() || "";
        const forceJpg = ext !== "jpg" && ext !== "jpeg";
        wx.compressImage({
          src: filePath,
          quality: clamp(quality, 30, 100),
          compressedWidth: targetW,
          compressedHeight: targetH,
          compressedFormat: forceJpg ? "jpg" : "none",
          success: (res) => resolve(res.tempFilePath || filePath),
          fail: () => resolve(filePath),
        });
      },
      fail: () => resolve(filePath),
    });
  });
}

function filePathToDataUrl(filePath) {
  return new Promise((resolve, reject) => {
    if (!filePath) return resolve("");
    const fs = wx.getFileSystemManager();
    fs.readFile({
      filePath,
      encoding: "base64",
      success: (res) => {
        const ext = String(filePath).toLowerCase().split(".").pop() || "jpg";
        const mime = ext === "png" ? "image/png" : (ext === "webp" ? "image/webp" : "image/jpeg");
        resolve(`data:${mime};base64,${res.data || ""}`);
      },
      fail: (err) => reject(new Error((err && err.errMsg) || "读取部件图失败")),
    });
  });
}

function buildAiGenerationPrompt(userPrompt, hasImage2, hasImage3, targetHex) {
  const raw = String(userPrompt || "").trim();
  let prompt = raw
    .replace(/部件图\s*2|部件图二|参考图\s*2|参考图二|图\s*3|图三/g, "image 3")
    .replace(/部件图\s*1|部件图一|部件图|部件|参考图\s*1|参考图一|图\s*2|图二/g, "image 2")
    .replace(/主图|原图|图\s*1|图一/g, "image 1");
  if (!hasImage2 && !hasImage3) {
    prompt += `\n目标色：#${String(targetHex || "").toUpperCase()}。保持主体、材质、光影和背景自然。`;
  }
  return prompt;
}

Page({
  data: {
    mode: "fast", // fast | ai
    enterpriseAiEnabled: true,
    localImage: "",
    referenceImage2: "",
    referenceImage3: "",
    recoloredUrl: "",
    recoloredLocalUrl: "",
    recolorMaskBackend: "",
    recolorMaskMode: "",
    aiUsedParamsText: "",
    processing: false,
    processingAi: false,

    hsvH: 30,
    hsvS: 1,
    hsvV: 1,
    targetHex: "FB8C00",
    targetColor: "#FB8C00",
    wheelSize: 220,
    wheelRadius: 110,
    wheelCenterX: 110,
    wheelCenterY: 110,
    pickX: 190,
    pickY: 55,

    stageLeft: 0,
    stageTop: 0,
    imgRect: null,
    selRect: null,
    refStageLeft: 0,
    refStageTop: 0,
    refImgRect: null,
    componentRect: null,
    dragMode: "",
    dragStartX: 0,
    dragStartY: 0,
    dragSelStart: null,
    refDragMode: "",
    refDragStartX: 0,
    refDragStartY: 0,
    refDragSelStart: null,

    strength: 80,
    feather: 2,
    fastParamsOpen: false,
    aiPrompt: "",
    aiPromptPlaceholder: "融合：将部件图的衣领合并到主图上\n改色：把主图衣服改成目标色",

    meterPanelOpen: false,
    meterScanning: false,
    meterConnecting: false,
    meterMeasuring: false,
    meterDevices: [],
    meterDeviceName: "",
    meterStatus: "未连接色差仪",
    meterLastLab: null,
    meterLastLabText: "",
    colorLibraries: [],
    selectedColorLibraryId: "",
    selectedColorLibraryName: "全部色卡",
    colorMatching: false,
    colorMatches: [],
    colorMatchError: "",
    experienceVersionEnabled: !!config.enableExperienceVersion,
  },

  onLoad() {
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

  goSearchPage() {
    wx.navigateBack({ fail: () => wx.reLaunch({ url: "/pages/index/index" }) });
  },

  goPrintPage() {
    wx.navigateTo({ url: "/pages/print/index" });
  },

  goRecolorPage() {},

  openColorLibraryNative() {
    wx.navigateTo({
      url: "/pages/color_card/index"
    });
  },

  switchMode(e) {
    const mode = e.currentTarget.dataset.mode;
    if (!mode || (mode !== "fast" && mode !== "ai")) return;
    if (mode === "ai" && !this.data.enterpriseAiEnabled) return;
    this.setData({ mode });
  },

  chooseImage() {
    wx.chooseMedia({
      count: 1,
      mediaType: ["image"],
      success: async (res) => {
        const file = (res.tempFiles || [])[0];
        if (!file || !file.tempFilePath) return;
        const uploadPath = await compressForUpload(file.tempFilePath);
        this.setData(
          {
            localImage: uploadPath,
            recoloredUrl: "",
            recoloredLocalUrl: "",
            selRect: null,
            imgRect: null,
          },
          () => this.setupStageAndImageRect()
        );
      },
      fail: () => wx.showToast({ title: "未选择图片", icon: "none" })
    });
  },

  chooseReferenceImage(e) {
    const slot = Number((e.currentTarget && e.currentTarget.dataset && e.currentTarget.dataset.slot) || 2);
    wx.chooseMedia({
      count: 1,
      mediaType: ["image"],
      success: async (res) => {
        const file = (res.tempFiles || [])[0];
        if (!file || !file.tempFilePath) return;
        const uploadPath = await compressForUpload(file.tempFilePath);
        const key = slot === 3 ? "referenceImage3" : "referenceImage2";
        this.setData(
          { [key]: uploadPath, recoloredUrl: "", recoloredLocalUrl: "", componentRect: null, refImgRect: null },
          () => this.setupRefImageRect()
        );
      },
      fail: () => wx.showToast({ title: "未选择图片", icon: "none" })
    });
  },

  removeReferenceImage(e) {
    const slot = Number((e.currentTarget && e.currentTarget.dataset && e.currentTarget.dataset.slot) || 2);
    const key = slot === 3 ? "referenceImage3" : "referenceImage2";
    this.setData({ [key]: "" });
  },

  setupStageAndImageRect() {
    const filePath = this.data.localImage;
    if (!filePath) return;
    wx.getImageInfo({
      src: filePath,
      success: (imgInfo) => {
        const q = wx.createSelectorQuery();
        q.select(".stage").boundingClientRect();
        q.exec((res) => {
          const box = (res && res[0]) || null;
          if (!box || !box.width || !box.height) return;
          const stageW = box.width;
          const stageH = box.height;
          const stageLeft = box.left || 0;
          const stageTop = box.top || 0;
          const iw = imgInfo.width || stageW;
          const ih = imgInfo.height || stageH;
          const scale = Math.min(stageW / iw, stageH / ih);
          const drawW = iw * scale;
          const drawH = ih * scale;
          const imgRect = {
            x: (stageW - drawW) / 2,
            y: (stageH - drawH) / 2,
            w: drawW,
            h: drawH,
          };
          this.setData({ stageLeft, stageTop, imgRect });
        });
      }
    });
  },

  setupRefImageRect() {
    const filePath = this.data.referenceImage2;
    if (!filePath) return;
    wx.getImageInfo({
      src: filePath,
      success: (imgInfo) => {
        const q = wx.createSelectorQuery();
        q.select(".ref-stage").boundingClientRect();
        q.exec((res) => {
          const box = (res && res[0]) || null;
          if (!box || !box.width || !box.height) return;
          const stageW = box.width;
          const stageH = box.height;
          const stageLeft = box.left || 0;
          const stageTop = box.top || 0;
          const iw = imgInfo.width || stageW;
          const ih = imgInfo.height || stageH;
          const scale = Math.min(stageW / iw, stageH / ih);
          const drawW = iw * scale;
          const drawH = ih * scale;
          const refImgRect = {
            x: (stageW - drawW) / 2,
            y: (stageH - drawH) / 2,
            w: drawW,
            h: drawH,
          };
          this.setData({ refStageLeft: stageLeft, refStageTop: stageTop, refImgRect });
        });
      }
    });
  },

  refreshStageOffset(cb) {
    const q = wx.createSelectorQuery();
    q.select(".stage").boundingClientRect();
    q.exec((res) => {
      const box = (res && res[0]) || null;
      if (!box) return typeof cb === "function" ? cb() : null;
      this.setData({ stageLeft: box.left || 0, stageTop: box.top || 0 }, () => {
        if (typeof cb === "function") cb();
      });
    });
  },

  _pointInSel(x, y, sel) {
    if (!sel) return false;
    return x >= sel.x && x <= sel.x + sel.w && y >= sel.y && y <= sel.y + sel.h;
  },

  _clampSelToImgRect(sel, imgRect) {
    if (!sel || !imgRect) return sel;
    const minSize = 12;
    let x = clamp(sel.x, imgRect.x, imgRect.x + imgRect.w - minSize);
    let y = clamp(sel.y, imgRect.y, imgRect.y + imgRect.h - minSize);
    let w = Math.max(minSize, sel.w);
    let h = Math.max(minSize, sel.h);
    if (x + w > imgRect.x + imgRect.w) w = imgRect.x + imgRect.w - x;
    if (y + h > imgRect.y + imgRect.h) h = imgRect.y + imgRect.h - y;
    return { x, y, w, h };
  },

  handleStageTouchStart(e) {
    if (!(this.data.enterpriseAiEnabled && this.data.mode === "ai")) return;
    this.refreshStageOffset(() => {
      const t = (e.touches || [])[0];
      const p = getTouchPoint(t);
      if (!p || !this.data.imgRect) return;
      const x = p.x - this.data.stageLeft;
      const y = p.y - this.data.stageTop;
      const imgRect = this.data.imgRect;
      if (x < imgRect.x || x > imgRect.x + imgRect.w || y < imgRect.y || y > imgRect.y + imgRect.h) return;
      const sel = this.data.selRect;
      if (this._pointInSel(x, y, sel)) {
        this.setData({ dragMode: "move", dragStartX: x, dragStartY: y, dragSelStart: { ...sel } });
      } else {
        this.setData({ dragMode: "create", dragStartX: x, dragStartY: y, dragSelStart: null, selRect: { x, y, w: 1, h: 1 } });
      }
    });
  },

  handleStageTouchMove(e) {
    if (!(this.data.enterpriseAiEnabled && this.data.mode === "ai")) return;
    const t = (e.touches || [])[0];
    const p = getTouchPoint(t);
    if (!p || !this.data.imgRect || !this.data.dragMode) return;
    const x = p.x - this.data.stageLeft;
    const y = p.y - this.data.stageTop;
    const imgRect = this.data.imgRect;

    if (this.data.dragMode === "create") {
      const x0 = this.data.dragStartX;
      const y0 = this.data.dragStartY;
      const sel = this._clampSelToImgRect(
        { x: Math.min(x0, x), y: Math.min(y0, y), w: Math.abs(x - x0), h: Math.abs(y - y0) },
        imgRect
      );
      this.setData({ selRect: sel });
      return;
    }

    if (this.data.dragMode === "move" && this.data.dragSelStart) {
      const dx = x - this.data.dragStartX;
      const dy = y - this.data.dragStartY;
      const s0 = this.data.dragSelStart;
      const sel = this._clampSelToImgRect({ x: s0.x + dx, y: s0.y + dy, w: s0.w, h: s0.h }, imgRect);
      this.setData({ selRect: sel });
    }
  },

  handleStageTouchEnd() {
    this.setData({ dragMode: "", dragSelStart: null });
  },

  refreshRefStageOffset(cb) {
    const q = wx.createSelectorQuery();
    q.select(".ref-stage").boundingClientRect();
    q.exec((res) => {
      const box = (res && res[0]) || null;
      if (!box) return typeof cb === "function" ? cb() : null;
      this.setData({ refStageLeft: box.left || 0, refStageTop: box.top || 0 }, () => {
        if (typeof cb === "function") cb();
      });
    });
  },

  handleRefTouchStart(e) {
    if (!(this.data.enterpriseAiEnabled && this.data.mode === "ai")) return;
    this.refreshRefStageOffset(() => {
      const t = (e.touches || [])[0];
      const p = getTouchPoint(t);
      if (!p || !this.data.refImgRect) return;
      const x = p.x - this.data.refStageLeft;
      const y = p.y - this.data.refStageTop;
      const imgRect = this.data.refImgRect;
      if (x < imgRect.x || x > imgRect.x + imgRect.w || y < imgRect.y || y > imgRect.y + imgRect.h) return;
      const sel = this.data.componentRect;
      if (this._pointInSel(x, y, sel)) {
        this.setData({ refDragMode: "move", refDragStartX: x, refDragStartY: y, refDragSelStart: { ...sel } });
      } else {
        this.setData({ refDragMode: "create", refDragStartX: x, refDragStartY: y, refDragSelStart: null, componentRect: { x, y, w: 1, h: 1 } });
      }
    });
  },

  handleRefTouchMove(e) {
    if (!(this.data.enterpriseAiEnabled && this.data.mode === "ai")) return;
    const t = (e.touches || [])[0];
    const p = getTouchPoint(t);
    if (!p || !this.data.refImgRect || !this.data.refDragMode) return;
    const x = p.x - this.data.refStageLeft;
    const y = p.y - this.data.refStageTop;
    const imgRect = this.data.refImgRect;

    if (this.data.refDragMode === "create") {
      const x0 = this.data.refDragStartX;
      const y0 = this.data.refDragStartY;
      const sel = this._clampSelToImgRect(
        { x: Math.min(x0, x), y: Math.min(y0, y), w: Math.abs(x - x0), h: Math.abs(y - y0) },
        imgRect
      );
      this.setData({ componentRect: sel });
      return;
    }

    if (this.data.refDragMode === "move" && this.data.refDragSelStart) {
      const dx = x - this.data.refDragStartX;
      const dy = y - this.data.refDragStartY;
      const s0 = this.data.refDragSelStart;
      const sel = this._clampSelToImgRect({ x: s0.x + dx, y: s0.y + dy, w: s0.w, h: s0.h }, imgRect);
      this.setData({ componentRect: sel });
    }
  },

  handleRefTouchEnd() {
    this.setData({ refDragMode: "", refDragSelStart: null });
  },

  pickColorFromWheel(clientX, clientY) {
    const query = wx.createSelectorQuery();
    query.select(".wheel-touch").boundingClientRect();
    query.exec((res) => {
      const box = (res && res[0]) || null;
      if (!box) return;
      const localX = clientX - box.left;
      const localY = clientY - box.top;
      const cx = this.data.wheelCenterX;
      const cy = this.data.wheelCenterY;
      const r = this.data.wheelRadius;
      const dx = localX - cx;
      const dy = localY - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const sat = clamp(dist / r, 0, 1);
      // CSS conic-gradient 默认 0deg 在正上方；atan2 计算的 0deg 在正右方，需 +90 对齐。
      const hue = (Math.atan2(dy, dx) * 180) / Math.PI + 90;
      const v = this.data.hsvV;
      const rgb = hsvToRgb(hue, sat, v);
      let px = localX;
      let py = localY;
      if (dist > r) {
        const k = r / dist;
        px = cx + dx * k;
        py = cy + dy * k;
      }
      const targetHex = rgbToHex(rgb.r, rgb.g, rgb.b);
      this.setData({
        hsvH: hue,
        hsvS: sat,
        pickX: px,
        pickY: py,
        targetHex,
        targetColor: `#${targetHex}`,
      });
    });
  },

  onWheelTouchStart(e) {
    const t = (e.touches || [])[0];
    const p = getTouchPoint(t);
    if (!p) return;
    this.pickColorFromWheel(p.x, p.y);
  },

  onWheelTouchMove(e) {
    const t = (e.touches || [])[0];
    const p = getTouchPoint(t);
    if (!p) return;
    this.pickColorFromWheel(p.x, p.y);
  },

  onValueChange(e) {
    const v = clamp(Number(e.detail.value) / 100, 0, 1);
    const { hsvH, hsvS } = this.data;
    const rgb = hsvToRgb(hsvH, hsvS, v);
    const targetHex = rgbToHex(rgb.r, rgb.g, rgb.b);
    this.setData({ hsvV: v, targetHex, targetColor: `#${targetHex}` });
  },

  onStrengthChange(e) {
    this.setData({ strength: Number(e.detail.value) });
  },

  onFeatherChange(e) {
    this.setData({ feather: Number(e.detail.value) });
  },

  onAiPromptInput(e) {
    this.setData({ aiPrompt: e.detail.value || "" });
  },

  applyTargetHex(targetHex, extra) {
    const rgb = hexToRgb(targetHex);
    if (!rgb) return;
    const hsv = rgbToHsv(rgb.r, rgb.g, rgb.b);
    const radius = this.data.wheelRadius;
    const angle = ((hsv.h - 90) * Math.PI) / 180;
    const dist = hsv.s * radius;
    this.setData({
      hsvH: hsv.h,
      hsvS: hsv.s,
      hsvV: hsv.v,
      pickX: this.data.wheelCenterX + Math.cos(angle) * dist,
      pickY: this.data.wheelCenterY + Math.sin(angle) * dist,
      targetHex,
      targetColor: `#${targetHex}`,
      ...(extra || {}),
    });
  },

  async loadColorLibraries() {
    try {
      const res = await fetchColorCardLibraries();
      const libs = (res.libraries || []).map((item) => ({
        id: String(item.id || ""),
        name: String(item.name || ""),
        color_count: Number(item.color_count || 0),
      })).filter((item) => item.id && item.name);
      this.setData({ colorLibraries: libs });
    } catch (err) {
      console.warn("[color-card:libraries:error]", err);
    }
  },

  async selectColorLibrary(e) {
    const id = String((e.currentTarget.dataset && e.currentTarget.dataset.id) || "");
    const name = String((e.currentTarget.dataset && e.currentTarget.dataset.name) || "全部色卡");
    this.setData({
      selectedColorLibraryId: id,
      selectedColorLibraryName: name || "全部色卡",
    });
    if (this.data.meterLastLab) {
      await this.matchMeasuredColor(this.data.meterLastLab);
    }
  },

  async matchMeasuredColor(lab) {
    if (!lab) return;
    this.setData({ colorMatching: true, colorMatchError: "" });
    try {
      const res = await matchColorCards({
        L: lab.L,
        a: lab.a,
        b: lab.b,
        library_id: this.data.selectedColorLibraryId,
        limit: 12,
      });
      const matches = (res.matches || []).map((item) => {
        const lValue = Number(item.l || 0);
        const delta = Number(item.delta_e_00 || 0);
        return {
          id: item.id,
          name: String(item.name || ""),
          library_name: String(item.library_name || ""),
          hex: String(item.hex || "CCCCCC").replace(/^#/, "").toUpperCase(),
          deltaText: delta.toFixed(2),
          labText: `L ${Number(item.l).toFixed(1)} / a ${Number(item.a).toFixed(1)} / b ${Number(item.b).toFixed(1)}`,
          textColor: lValue < 55 ? "#FFFFFF" : "#0F172A",
          itemStyle: `background-color: #${String(item.hex || "CCCCCC").replace(/^#/, "").toUpperCase()}; color: ${lValue < 55 ? "#FFFFFF" : "#0F172A"};`,
        };
      });
      this.setData({ colorMatches: matches });
    } catch (err) {
      console.error("[color-card:match:error]", err);
      this.setData({ colorMatchError: err.message || "色卡匹配失败", colorMatches: [] });
    } finally {
      this.setData({ colorMatching: false });
    }
  },

  toggleMeterPanel() {
    const nextOpen = !this.data.meterPanelOpen;
    this.setData({ meterPanelOpen: nextOpen });
    if (nextOpen && !this.data.colorLibraries.length) {
      this.loadColorLibraries();
    }
    if (nextOpen && !ColorMeter.connected && !this.data.meterScanning) {
      this.startMeterScan();
    }
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
          const name = device.name || device.localName || "";
          const normalized = { ...device, name };
          const idx = merged.findIndex((i) => i.deviceId === normalized.deviceId);
          if (idx >= 0) merged[idx] = normalized;
          else merged.push(normalized);
        });
        merged.sort((a, b) => scoreMeterDevice(b) - scoreMeterDevice(a));
        this.setData({ meterDevices: merged });
      };
      this.setData({ meterScanning: true, meterDevices: [], meterStatus: "正在扫描色差仪" });
      await ColorMeter.startScan(this._meterScanHandler, 10000);
      setTimeout(() => {
        if (this.data.meterScanning) this.setData({ meterScanning: false, meterStatus: ColorMeter.connected ? "色差仪已连接" : "扫描完成" });
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
      console.error("[color-meter:connect:error]", err);
      await ColorMeter.disconnect().catch(() => null);
      this.setData({ meterConnecting: false, meterStatus: "连接失败" });
      wx.showToast({ title: "连接失败", icon: "none" });
    }
  },

  async disconnectMeter() {
    await ColorMeter.disconnect().catch(() => null);
    this.setData({ meterDeviceName: "", meterStatus: "未连接色差仪" });
  },

  async measureTargetColor() {
    if (this.data.meterMeasuring) return;
    if (!ColorMeter.connected) {
      this.openMeterPanelOnly();
      wx.showToast({ title: "请先连接色差仪", icon: "none" });
      return;
    }
    this.setData({ meterMeasuring: true, meterStatus: "正在测量目标色" });
    try {
      const lab = await retry(() => ColorMeter.measureAndGetLab(), 2);
      const targetHex = labToHex(lab);
      this.applyTargetHex(targetHex, {
        meterLastLab: lab,
        meterLastLabText: `Lab：L ${lab.L.toFixed(2)} / a ${lab.a.toFixed(2)} / b ${lab.b.toFixed(2)}`,
        meterStatus: `测量完成 #${targetHex}`,
      });
      await this.matchMeasuredColor(lab);
      wx.showToast({ title: "已设为目标色", icon: "none" });
    } catch (err) {
      console.error("[color-meter:measure:error]", err);
      this.setData({ meterStatus: "测量失败" });
      wx.showToast({ title: "测量失败", icon: "none" });
    } finally {
      this.setData({ meterMeasuring: false });
    }
  },

  toggleFastParams() {
    this.setData({ fastParamsOpen: !this.data.fastParamsOpen });
  },

  buildRecolorPayload(useFullImage = false) {
    const img = this.data.imgRect;
    if (!img) {
      return {
        target_hex: this.data.targetHex,
        x_ratio: 0,
        y_ratio: 0,
        w_ratio: 1,
        h_ratio: 1,
        strength: this.data.strength / 100,
        feather_ratio: this.data.feather / 100,
      };
    }
    const sel = useFullImage || !this.data.selRect
      ? { x: img.x, y: img.y, w: img.w, h: img.h }
      : this.data.selRect;
    return {
      target_hex: this.data.targetHex,
      x_ratio: (sel.x - img.x) / img.w,
      y_ratio: (sel.y - img.y) / img.h,
      w_ratio: sel.w / img.w,
      h_ratio: sel.h / img.h,
      strength: this.data.strength / 100,
      feather_ratio: this.data.feather / 100,
    };
  },

  buildRectPayload(sel, img) {
    if (!sel || !img) return null;
    return {
      x: clamp((sel.x - img.x) / img.w, 0, 1),
      y: clamp((sel.y - img.y) / img.h, 0, 1),
      w: clamp(sel.w / img.w, 0.01, 1),
      h: clamp(sel.h / img.h, 0.01, 1),
    };
  },

  expandRectPayload(rect, xScale = 2.0, yScale = 2.4) {
    if (!rect) return null;
    const cx = rect.x + rect.w / 2;
    const cy = rect.y + rect.h / 2;
    const w = clamp(rect.w * xScale, 0.01, 1);
    const h = clamp(rect.h * yScale, 0.01, 1);
    return {
      x: clamp(cx - w / 2, 0, 1 - w),
      y: clamp(cy - h / 2, 0, 1 - h),
      w,
      h,
    };
  },

  async runRecolor() {
    if (!this.data.localImage || this.data.processing || this.data.processingAi) {
      wx.showToast({ title: "请先选择图片", icon: "none" });
      return;
    }
    this.setData({ processing: true });
    try {
      const payload = this.buildRecolorPayload(true);
      payload.auto_mask = true;
      const res = await recolorUpload(this.data.localImage, payload);
      const remoteUrl = toAbsolute(res.recolored_url);
      const localUrl = await downloadToLocal(`${remoteUrl}${remoteUrl.includes("?") ? "&" : "?"}t=${Date.now()}`);
      this.setData({
        recoloredUrl: remoteUrl,
        recoloredLocalUrl: localUrl,
        recolorMaskBackend: String(res.mask_backend || ""),
        recolorMaskMode: String(res.mask_mode || ""),
        aiUsedParamsText: "",
      });
      wx.showToast({ title: "抠图换色完成", icon: "none" });
    } catch (err) {
      console.error("[recolor:error]", err);
      wx.showToast({ title: err.message || "抠图换色失败", icon: "none" });
    } finally {
      this.setData({ processing: false });
    }
  },

  async runAiRecolor() {
    if (!this.data.enterpriseAiEnabled) {
      wx.showToast({ title: "当前版本未开放该功能", icon: "none" });
      return;
    }
    if (!this.data.localImage || this.data.processing || this.data.processingAi) {
      wx.showToast({ title: "请先选择图片", icon: "none" });
      return;
    }
    this.setData({ processingAi: true });
    try {
      const payload = this.buildRecolorPayload(true);
      const userPrompt = (this.data.aiPrompt || "").trim();
      if (!userPrompt) {
        wx.showToast({ title: "请先输入融合说明", icon: "none" });
        return;
      }
      const hasImage2 = !!this.data.referenceImage2;
      const hasImage3 = !!this.data.referenceImage3;
      payload.prompt = buildAiGenerationPrompt(userPrompt, hasImage2, hasImage3, this.data.targetHex);
      if (this.data.referenceImage2) payload.image2 = await filePathToDataUrl(this.data.referenceImage2);
      if (this.data.referenceImage3) payload.image3 = await filePathToDataUrl(this.data.referenceImage3);
      const res = await recolorAiUpload(this.data.localImage, payload);
      const remoteUrl = toAbsolute(res.recolored_url);
      const localUrl = await downloadToLocal(`${remoteUrl}${remoteUrl.includes("?") ? "&" : "?"}t=${Date.now()}`);
      const used = res.used_params || {};
      const image2Status = used.has_image2 === undefined ? "unknown" : (used.has_image2 ? "yes" : "no");
      const image3Status = used.has_image3 === undefined ? "unknown" : (used.has_image3 ? "yes" : "no");
      const usedText = [
        `prompt: ${used.prompt || payload.prompt || ""}`,
        `image2: ${image2Status}`,
        `image3: ${image3Status}`,
      ].join("\n");
      this.setData({ recoloredUrl: remoteUrl, recoloredLocalUrl: localUrl, recolorMaskBackend: "", recolorMaskMode: "", aiUsedParamsText: usedText });
      wx.showToast({ title: "预览完成", icon: "none" });
    } catch (err) {
      console.error("[recolor:ai:error]", err);
      wx.showToast({ title: err.message || "预览失败", icon: "none" });
    } finally {
      this.setData({ processingAi: false });
    }
  },

  previewResult() {
    const current = this.data.recoloredLocalUrl || this.data.recoloredUrl;
    if (!current) return;
    wx.previewImage({ current, urls: [current] });
  },

  onResultImageError(e) {
    console.error("[recolor:resultImageError]", e, this.data.recoloredUrl, this.data.recoloredLocalUrl);
    wx.showToast({ title: "结果图加载失败", icon: "none" });
  },
});
