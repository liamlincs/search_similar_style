Page({
  data: {
    styleCode: "",
    images: [],
    currentIndex: 0,
    storageKey: "",
  },

  onLoad(options) {
    const key = decodeURIComponent((options && options.key) || "");
    const payload = key ? wx.getStorageSync(key) : null;
    const images = (payload && payload.images) || [];
    this.setData({
      storageKey: key,
      styleCode: (payload && payload.styleCode) || "",
      images,
      currentIndex: Number((payload && payload.current) || 0),
    });
  },

  onUnload() {
    if (this.data.storageKey) wx.removeStorageSync(this.data.storageKey);
  },

  onSwiperChange(e) {
    this.setData({ currentIndex: Number(e.detail.current || 0) });
  },

  goBack() {
    wx.navigateBack();
  },

  viewOriginal() {
    const item = (this.data.images || [])[this.data.currentIndex];
    const url = (item && (item.originalUrl || item.url)) || "";
    if (!url) return;
    wx.previewImage({ current: url, urls: [url] });
  },
});
