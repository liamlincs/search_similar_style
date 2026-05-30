const { recolorUpload, recolorAiUpload } = require("../../utils/api");
const config = require("../../utils/config");

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

Page({
  data: {
    mode: "fast", // fast | ai
    localImage: "",
    recoloredUrl: "",
    recoloredLocalUrl: "",
    recolorMaskBackend: "",
    recolorMaskMode: "",
    processing: false,
    processingAi: false,

    hsvH: 30,
    hsvS: 1,
    hsvV: 1,
    targetHex: "FB8C00",
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
    dragMode: "",
    dragStartX: 0,
    dragStartY: 0,
    dragSelStart: null,

    strength: 80,
    feather: 2,
    fastParamsOpen: false,
    aiPrompt: "",
    aiRawMode: true,
  },

  goSearchPage() {
    wx.navigateBack({ fail: () => wx.reLaunch({ url: "/pages/index/index" }) });
  },

  goPrintPage() {
    wx.navigateTo({ url: "/pages/print/index" });
  },

  goRecolorPage() {},

  switchMode(e) {
    const mode = e.currentTarget.dataset.mode;
    if (!mode || (mode !== "fast" && mode !== "ai")) return;
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
      this.setData({
        hsvH: hue,
        hsvS: sat,
        pickX: px,
        pickY: py,
        targetHex: rgbToHex(rgb.r, rgb.g, rgb.b),
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
    this.setData({ hsvV: v, targetHex: rgbToHex(rgb.r, rgb.g, rgb.b) });
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

  toggleFastParams() {
    this.setData({ fastParamsOpen: !this.data.fastParamsOpen });
  },

  onAiRawModeChange(e) {
    this.setData({ aiRawMode: !!e.detail.value });
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
      });
      wx.showToast({ title: "标准换色完成", icon: "none" });
    } catch (err) {
      console.error("[recolor:error]", err);
      wx.showToast({ title: err.message || "标准换色失败", icon: "none" });
    } finally {
      this.setData({ processing: false });
    }
  },

  async runAiRecolor() {
    if (!this.data.localImage || this.data.processing || this.data.processingAi) {
      wx.showToast({ title: "请先选择图片", icon: "none" });
      return;
    }
    this.setData({ processingAi: true });
    try {
      const payload = this.buildRecolorPayload(true);
      const userPrompt = (this.data.aiPrompt || "").trim();
      const colorHint = `目标色必须为 #${this.data.targetHex}。`;
      payload.prompt = userPrompt
        ? `${userPrompt}\n${colorHint}\n请严格按照目标色调整，不要改成其他颜色。`
        : `将整张图主色调调整为 #${this.data.targetHex}。请严格按目标色处理，保持文字清晰和纹理自然。`;
      payload.model = "Qwen/Qwen-Image-Edit-2509";
      payload.num_inference_steps = 20;
      payload.postprocess = !this.data.aiRawMode;
      const res = await recolorAiUpload(this.data.localImage, payload);
      const remoteUrl = toAbsolute(res.recolored_url);
      const localUrl = await downloadToLocal(`${remoteUrl}${remoteUrl.includes("?") ? "&" : "?"}t=${Date.now()}`);
      this.setData({ recoloredUrl: remoteUrl, recoloredLocalUrl: localUrl, recolorMaskBackend: "", recolorMaskMode: "" });
      wx.showToast({ title: "AI换色完成", icon: "none" });
    } catch (err) {
      console.error("[recolor:ai:error]", err);
      wx.showToast({ title: err.message || "AI换色失败", icon: "none" });
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
