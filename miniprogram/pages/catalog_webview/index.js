const config = require("../../utils/config");

Page({
  data: {
    url: ""
  },

  onLoad(options) {
    const url = this.buildCatalogUrl(options);
    const title = decodeURIComponent(options.title || "产品库");
    wx.setNavigationBarTitle({ title });
    this.setData({ url });
  },

  buildCatalogUrl(options) {
    const decodeOption = (value) => decodeURIComponent(String(value || "")).trim();
    const type = options.type === "color" ? "color" : (options.type === "product" ? "product" : "");
    if (type) {
      const env = decodeOption(options.env || "");
      const envBaseUrls = config.catalogH5BaseUrls || {};
      const baseUrl = (
        decodeOption(options.h5_base_url || options.base_url || "") ||
        String(env ? (envBaseUrls[env] || "") : "").trim() ||
        String(config.catalogH5BaseUrl || "").trim() ||
        String(config.baseUrl || "").trim()
      ).replace(/\/+$/, "");
      const path = config.catalogH5Path || "/catalog";
      const token = decodeOption(options.token || options.catalog_token || options.access_token || "");
      const query = [`type=${encodeURIComponent(type)}`];
      if (token) query.push(`token=${encodeURIComponent(token)}`);
      return `${baseUrl}${path}?${query.join("&")}`;
    }
    return decodeURIComponent(options.url || "").replace(/&amp;/g, "&");
  }
});
