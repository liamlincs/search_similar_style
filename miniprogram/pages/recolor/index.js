const { recolorUpload } = require("../../utils/api");
const config = require("../../utils/config");

function toAbsolute(pathOrUrl) {
  if (!pathOrUrl) return "";
  if (/^https?:\/\//.test(pathOrUrl)) return pathOrUrl;
  return `${config.baseUrl}${pathOrUrl}`;
}

Page({
  data: {
    localImage: "",
    recoloredUrl: "",
    processing: false,
    targetHex: "FF5500",
    xRatio: 20,
    yRatio: 20,
    wRatio: 40,
    hRatio: 40,
    strength: 80,
    feather: 2,
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
        this.setData({ localImage: file.tempFilePath, recoloredUrl: "" });
      },
      fail: () => wx.showToast({ title: "未选择图片", icon: "none" })
    });
  },

  onHexInput(e) { this.setData({ targetHex: (e.detail.value || "").replace(/[^0-9a-fA-F]/g, "").slice(0, 6) }); },
  onXChange(e) { this.setData({ xRatio: Number(e.detail.value) }); },
  onYChange(e) { this.setData({ yRatio: Number(e.detail.value) }); },
  onWChange(e) { this.setData({ wRatio: Number(e.detail.value) }); },
  onHChange(e) { this.setData({ hRatio: Number(e.detail.value) }); },
  onStrengthChange(e) { this.setData({ strength: Number(e.detail.value) }); },
  onFeatherChange(e) { this.setData({ feather: Number(e.detail.value) }); },

  async runRecolor() {
    if (!this.data.localImage || this.data.processing) {
      wx.showToast({ title: "请先选择图片", icon: "none" });
      return;
    }
    this.setData({ processing: true });
    try {
      const payload = {
        target_hex: this.data.targetHex || "FF5500",
        x_ratio: this.data.xRatio / 100,
        y_ratio: this.data.yRatio / 100,
        w_ratio: this.data.wRatio / 100,
        h_ratio: this.data.hRatio / 100,
        strength: this.data.strength / 100,
        feather_ratio: this.data.feather / 100,
      };
      const res = await recolorUpload(this.data.localImage, payload);
      this.setData({ recoloredUrl: toAbsolute(res.recolored_url) });
      wx.showToast({ title: "改色完成", icon: "none" });
    } catch (err) {
      console.error("[recolor:error]", err);
      wx.showToast({ title: err.message || "改色失败", icon: "none" });
    } finally {
      this.setData({ processing: false });
    }
  },

  previewResult() {
    if (!this.data.recoloredUrl) return;
    wx.previewImage({ current: this.data.recoloredUrl, urls: [this.data.recoloredUrl] });
  }
});
