Page({
  data: {
    url: ""
  },

  onLoad(options) {
    const url = decodeURIComponent(options.url || "");
    const title = decodeURIComponent(options.title || "资料库");
    wx.setNavigationBarTitle({ title });
    this.setData({ url });
  }
});
