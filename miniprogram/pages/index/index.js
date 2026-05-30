const { uploadAndSearch, fetchSignedImageUrl } = require("../../utils/api");
const config = require("../../utils/config");

Page({
  data: {
    localImage: "",
    searching: false,
    hasSearched: false,
    errorMessage: "",
    results: [],
    isAmbiguous: false,
    confidenceBand: "low"
  },

  chooseFromAlbum() {
    this.pickImage(["album"]);
  },

  takePhoto() {
    this.pickImage(["camera"]);
  },

  retrySearch() {
    const filePath = this.data.localImage;
    if (!filePath || this.data.searching) return;
    this.search(filePath);
  },

  goPrintPage() {
    wx.navigateTo({ url: "/pages/print/index" });
  },

  goRecolorPage() {
    wx.navigateTo({ url: "/pages/recolor/index" });
  },

  goSearchPage() {},

  pickImage(sourceType) {
    wx.chooseMedia({
      count: 1,
      mediaType: ["image"],
      sourceType,
      success: (res) => {
        const file = res.tempFiles && res.tempFiles[0];
        if (!file || !file.tempFilePath) {
          wx.showToast({ title: "未获取到图片", icon: "none" });
          return;
        }
        this.setData({
          localImage: file.tempFilePath,
          hasSearched: false,
          errorMessage: "",
          results: [],
          isAmbiguous: false,
          confidenceBand: "low"
        });
        this.search(file.tempFilePath);
      },
      fail: () => {
        wx.showToast({ title: "已取消选择", icon: "none" });
      }
    });
  },

  async search(filePath) {
    this.setData({ searching: true, errorMessage: "" });
    try {
      const resp = await uploadAndSearch(filePath);
      const topCodes = resp.topk_style_codes || [];
      const byImage = {};
      topCodes.forEach((item, idx) => {
        const key = item.best_standard_image || "";
        if (key) byImage[key] = { item, idx };
      });
      const srcList = (resp.similar_images && resp.similar_images.length)
        ? resp.similar_images
        : topCodes.map((item) => ({
            image_name: item.best_standard_image || "",
            image_url: item.best_standard_image_url || "",
            rank_score: Number(item.rank_score || 0)
          }));

      const list = srcList.map((row, idx) => {
        const imageName = row.image_name || row.best_standard_image || "";
        const meta = byImage[imageName] || null;
        const scoreNum = Number(row.score || 0);
        return {
          rank: idx + 1,
          styleCode:
            (meta && meta.item && meta.item.style_code) ||
            row.style_code ||
            (imageName ? imageName.replace(/\.[^.]+$/, "").replace(/_[^_]+$/, "") : "-"),
          imageName,
          imageUrl: row.image_url || row.best_standard_image_url || "",
          imageRetryCount: 0,
          score: scoreNum,
          scoreText: `${(scoreNum * 100).toFixed(2)}%`,
          rankScore: Number(row.rank_score || 0)
        };
      }).sort((a, b) => Number(b.score || 0) - Number(a.score || 0))
        .map((item, idx) => ({ ...item, rank: idx + 1 }));

      this.setData({
        hasSearched: true,
        results: list,
        isAmbiguous: !!resp.is_ambiguous,
        confidenceBand: resp.confidence_band || "low",
        errorMessage: list.length ? "" : "没有找到相似款，请更换图片重试。"
      });
    } catch (err) {
      this.setData({
        hasSearched: true,
        results: [],
        isAmbiguous: false,
        confidenceBand: "low",
        errorMessage: err.message || "检索失败，请稍后重试"
      });
    } finally {
      this.setData({ searching: false });
    }
  },

  previewResult(e) {
    const current = e.currentTarget.dataset.url;
    if (!current) return;
    const urls = this.data.results.map((x) => x.imageUrl).filter(Boolean);
    wx.previewImage({ current, urls });
  },

  async onResultImageError(e) {
    const idx = Number(e.currentTarget.dataset.index);
    if (!Number.isInteger(idx) || idx < 0) return;
    const current = this.data.results[idx];
    if (!current || !current.imageName) return;

    const maxRetries = Number((config.imageLoadRetry || {}).maxRetries || 0);
    const tried = Number(current.imageRetryCount || 0);
    if (tried >= maxRetries) return;

    try {
      const refreshed = await fetchSignedImageUrl(current.imageName);
      const keyUrl = `results[${idx}].imageUrl`;
      const keyRetry = `results[${idx}].imageRetryCount`;
      this.setData({
        [keyUrl]: refreshed.image_url || current.imageUrl,
        [keyRetry]: tried + 1
      });
    } catch (err) {
      const keyRetry = `results[${idx}].imageRetryCount`;
      this.setData({ [keyRetry]: tried + 1 });
    }
  }
});
