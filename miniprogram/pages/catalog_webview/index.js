const config = require("../../utils/config");

Page({
  data: {
    url: ""
  },

  onLoad(options) {
    const url = this.buildCatalogUrl(options);
    const title = decodeURIComponent(options.title || "资料库");
    wx.setNavigationBarTitle({ title });
    this.setData({ url });
  },

  buildCatalogUrl(options) {
    const type = options.type === "color" ? "color" : (options.type === "product" ? "product" : "");
    if (type) {
      const baseUrl = String(config.baseUrl || "").replace(/\/+$/, "");
      const path = config.catalogH5Path || "/catalog";
      const token = config.catalogH5Token || config.apiKey || "";
      return `${baseUrl}${path}?type=${type}&token=${encodeURIComponent(token)}`;
    }
    return decodeURIComponent(options.url || "").replace(/&amp;/g, "&");
  }
});
