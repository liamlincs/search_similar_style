const { recolorUpload } = require("../../utils/api");
const config = require("../../utils/config");

function toAbsolute(pathOrUrl) {
  if (!pathOrUrl) return "";
  if (/^https?:\/\//.test(pathOrUrl)) return pathOrUrl;
  return `${config.baseUrl}${pathOrUrl}`;
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

function getTouchPoint(t) {
  if (!t) return null;
  const x = Number(t.clientX ?? t.pageX ?? t.x);
  const y = Number(t.clientY ?? t.pageY ?? t.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y };
}

Page({
  data: {
    localImage: "",
    recoloredUrl: "",
    recoloredLocalUrl: "",
    processing: false,
    palette: [
      { name: "红", hex: "E53935" },
      { name: "橙", hex: "FB8C00" },
      { name: "黄", hex: "FDD835" },
      { name: "绿", hex: "43A047" },
      { name: "青", hex: "00ACC1" },
      { name: "蓝", hex: "1E88E5" },
      { name: "紫", hex: "8E24AA" },
      { name: "黑", hex: "212121" },
      { name: "白", hex: "F5F5F5" }
    ],
    selectedColorIndex: 1,
    stageW: 0,
    stageH: 0,
    stageLeft: 0,
    stageTop: 0,
    imgRect: null,
    selRect: null,
    strength: 80,
    feather: 2,
    dragMode: "",
    dragStartX: 0,
    dragStartY: 0,
    dragSelStart: null,
  },

  goSearchPage() {
    wx.navigateBack({ fail: () => wx.reLaunch({ url: "/pages/index/index" }) });
  },

  goPrintPage() {
    wx.navigateTo({ url: "/pages/print/index" });
  },

  goRecolorPage() {},

  chooseImage() {
    wx.chooseMedia({
      count: 1,
      mediaType: ["image"],
      success: (res) => {
        const file = (res.tempFiles || [])[0];
        if (!file || !file.tempFilePath) return;
        this.setData({ localImage: file.tempFilePath, recoloredUrl: "", recoloredLocalUrl: "", selRect: null, imgRect: null }, () => {
          this.setupStageAndImageRect();
        });
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
          this.setData({ stageW, stageH, stageLeft, stageTop, imgRect });
        });
      }
    });
  },

  refreshStageOffset(cb) {
    const q = wx.createSelectorQuery();
    q.select(".stage").boundingClientRect();
    q.exec((res) => {
      const box = (res && res[0]) || null;
      if (!box) {
        if (typeof cb === "function") cb();
        return;
      }
      this.setData(
        {
          stageLeft: box.left || 0,
          stageTop: box.top || 0,
        },
        () => {
          if (typeof cb === "function") cb();
        }
      );
    });
  },

  _pointInSel(x, y, sel) {
    if (!sel) return false;
    return x >= sel.x && x <= sel.x + sel.w && y >= sel.y && y <= sel.y + sel.h;
  },

  _clampSelToImgRect(sel, imgRect) {
    if (!sel || !imgRect) return sel;
    const minSize = 12;
    let x = Math.max(imgRect.x, Math.min(sel.x, imgRect.x + imgRect.w - minSize));
    let y = Math.max(imgRect.y, Math.min(sel.y, imgRect.y + imgRect.h - minSize));
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
      const x = p.x - (this.data.stageLeft || 0);
      const y = p.y - (this.data.stageTop || 0);
      const imgRect = this.data.imgRect;
      if (x < imgRect.x || x > imgRect.x + imgRect.w || y < imgRect.y || y > imgRect.y + imgRect.h) return;

      const sel = this.data.selRect;
      if (this._pointInSel(x, y, sel)) {
        this.setData({
          dragMode: "move",
          dragStartX: x,
          dragStartY: y,
          dragSelStart: { ...sel }
        });
      } else {
        this.setData({
          dragMode: "create",
          dragStartX: x,
          dragStartY: y,
          dragSelStart: null,
          selRect: { x, y, w: 1, h: 1 }
        });
      }
    });
  },

  handleStageTouchMove(e) {
    const t = (e.touches || [])[0];
    const p = getTouchPoint(t);
    if (!p || !this.data.imgRect || !this.data.dragMode) return;
    const x = p.x - (this.data.stageLeft || 0);
    const y = p.y - (this.data.stageTop || 0);
    const mode = this.data.dragMode;
    const imgRect = this.data.imgRect;

    if (mode === "create") {
      const x0 = this.data.dragStartX;
      const y0 = this.data.dragStartY;
      const sx = Math.min(x0, x);
      const sy = Math.min(y0, y);
      const sw = Math.abs(x - x0);
      const sh = Math.abs(y - y0);
      const sel = this._clampSelToImgRect({ x: sx, y: sy, w: sw, h: sh }, imgRect);
      this.setData({ selRect: sel });
      return;
    }

    if (mode === "move" && this.data.dragSelStart) {
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

  selectColor(e) {
    const idx = Number(e.currentTarget.dataset.index);
    if (!Number.isInteger(idx) || idx < 0) return;
    this.setData({ selectedColorIndex: idx });
  },
  onStrengthChange(e) { this.setData({ strength: Number(e.detail.value) }); },
  onFeatherChange(e) { this.setData({ feather: Number(e.detail.value) }); },

  async runRecolor() {
    if (!this.data.localImage || this.data.processing) {
      wx.showToast({ title: "请先选择图片", icon: "none" });
      return;
    }
    if (!this.data.selRect || !this.data.imgRect) {
      wx.showToast({ title: "请先在图片上框选改色区域", icon: "none" });
      return;
    }
    this.setData({ processing: true });
    try {
      const sel = this.data.selRect;
      const img = this.data.imgRect;
      const payload = {
        target_hex: (this.data.palette[this.data.selectedColorIndex] || {}).hex || "FF5500",
        x_ratio: (sel.x - img.x) / img.w,
        y_ratio: (sel.y - img.y) / img.h,
        w_ratio: sel.w / img.w,
        h_ratio: sel.h / img.h,
        strength: this.data.strength / 100,
        feather_ratio: this.data.feather / 100,
      };
      const res = await recolorUpload(this.data.localImage, payload);
      const remoteUrl = toAbsolute(res.recolored_url);
      const localUrl = await downloadToLocal(`${remoteUrl}${remoteUrl.includes("?") ? "&" : "?"}t=${Date.now()}`);
      this.setData({ recoloredUrl: remoteUrl, recoloredLocalUrl: localUrl });
      console.log("[recolor:result]", { remoteUrl, localUrl });
      wx.showToast({ title: "改色完成", icon: "none" });
    } catch (err) {
      console.error("[recolor:error]", err);
      wx.showToast({ title: err.message || "改色失败", icon: "none" });
    } finally {
      this.setData({ processing: false });
    }
  },

  previewResult() {
    const current = this.data.recoloredLocalUrl || this.data.recoloredUrl;
    if (!current) return;
    wx.previewImage({ current, urls: [current] });
  },

  clearSelection() {
    this.setData({ selRect: null });
  },

  onResultImageError(e) {
    console.error("[recolor:resultImageError]", e, this.data.recoloredUrl, this.data.recoloredLocalUrl);
    wx.showToast({ title: "结果图加载失败", icon: "none" });
  },
});
