const { uploadAndSearch, fetchSignedImageUrl } = require("../../utils/api");
const config = require("../../utils/config");

Page({
  data: {
    localImage: "",
    searching: false,
    hasSearched: false,
    errorMessage: "",
    results: []
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
          results: []
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
      const list = (resp.topk_style_codes || []).map((item, idx) => {
        const scoreNum = Number(item.score || 0);
        return {
          rank: idx + 1,
          styleCode: item.style_code || "-",
          imageName: item.best_standard_image || "",
          imageUrl: item.best_standard_image_url || "",
          imageRetryCount: 0,
          score: scoreNum,
          scoreText: `${(scoreNum * 100).toFixed(2)}%`
        };
      });

      this.setData({
        hasSearched: true,
        results: list,
        errorMessage: list.length ? "" : "没有找到相似款，请更换图片重试。"
      });
    } catch (err) {
      this.setData({
        hasSearched: true,
        results: [],
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
