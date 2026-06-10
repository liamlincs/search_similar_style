const { uploadAndSearch, fetchSignedImageUrl, fetchCatalogTags, fetchCatalogProducts, fetchCatalogProductDetail } = require("../../utils/api");
const config = require("../../utils/config");

function uniqTags(tags) {
  return Array.from(new Set((tags || []).filter(Boolean)));
}

function buildCatalogTagItems(allTags, selectedTags) {
  const selected = new Set(selectedTags || []);
  return (allTags || []).map((tag) => ({
    name: tag,
    active: selected.has(tag)
  }));
}

function buildPreviewUrls(product) {
  const imageUrls = (product.images || [])
    .map((item) => item.image_url || item.imageUrl || "")
    .filter(Boolean);
  if (imageUrls.length) return uniqTags(imageUrls);
  const cover = product.coverImageUrl || product.cover_image_url || product.imageUrl || "";
  return cover ? [cover] : [];
}

Page({
  data: {
    pageMode: "image",
    localImage: "",
    searching: false,
    hasSearched: false,
    errorMessage: "",
    results: [],
    isAmbiguous: false,
    confidenceBand: "low",
    catalogQuery: "",
    catalogTags: [],
    catalogTagItems: [],
    selectedCatalogTags: [],
    catalogLoading: false,
    catalogLoadingMore: false,
    catalogHasMore: true,
    catalogLimit: 20,
    catalogOffset: 0,
    catalogErrorMessage: "",
    catalogResults: []
  },

  onLoad() {
    this.loadCatalogFilters();
    this.searchCatalog(true);
  },

  onReachBottom() {
    if (this.data.pageMode !== "catalog") return;
    if (this.data.catalogLoading || this.data.catalogLoadingMore || !this.data.catalogHasMore) return;
    this.searchCatalog(false);
  },

  switchMode(e) {
    const mode = e.currentTarget.dataset.mode || "image";
    if (mode === this.data.pageMode) return;
    this.setData({ pageMode: mode });
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

  onCatalogQueryInput(e) {
    this.setData({ catalogQuery: e.detail.value || "" });
  },

  toggleCatalogTag(e) {
    const tag = e.currentTarget.dataset.tag || "";
    if (!tag) return;
    const selected = this.data.selectedCatalogTags || [];
    const next = selected.includes(tag)
      ? selected.filter((item) => item !== tag)
      : uniqTags(selected.concat(tag));
    this.setData({
      selectedCatalogTags: next,
      catalogTagItems: buildCatalogTagItems(this.data.catalogTags, next)
    });
    this.searchCatalog(true);
  },

  clearCatalogFilters() {
    this.setData({
      catalogQuery: "",
      selectedCatalogTags: [],
      catalogTagItems: buildCatalogTagItems(this.data.catalogTags, []),
      catalogErrorMessage: ""
    });
    this.searchCatalog(true);
  },

  async loadCatalogFilters() {
    try {
      const resp = await fetchCatalogTags();
      const tags = resp.tags || [];
      this.setData({
        catalogTags: tags,
        catalogTagItems: buildCatalogTagItems(tags, this.data.selectedCatalogTags || [])
      });
    } catch (_err) {}
  },

  async searchCatalog(reset = true) {
    if (this.data.catalogLoading || this.data.catalogLoadingMore) return;
    const offset = reset ? 0 : Number(this.data.catalogOffset || 0);
    const limit = Number(this.data.catalogLimit || 20);
    this.setData({
      catalogLoading: reset,
      catalogLoadingMore: !reset,
      catalogErrorMessage: reset ? "" : this.data.catalogErrorMessage
    });
    try {
      const resp = await fetchCatalogProducts({
        style_code: this.data.catalogQuery,
        tags: this.data.selectedCatalogTags,
        limit,
        offset
      });
      const list = (resp.products || []).map((item) => ({
        styleCode: item.style_code || "",
        coverImage: item.cover_image || "",
        coverImageUrl: item.cover_image_url || "",
        imageRetryCount: 0,
        tags: item.tags || [],
        images: item.images || [],
        imageCount: (item.images || []).length,
        previewUrls: buildPreviewUrls(item)
      }));
      const merged = reset ? list : (this.data.catalogResults || []).concat(list);
      this.setData({
        catalogResults: merged,
        catalogOffset: merged.length,
        catalogHasMore: list.length >= limit
      });
    } catch (err) {
      this.setData({
        catalogResults: reset ? [] : this.data.catalogResults,
        catalogErrorMessage: err.message || "产品库检索失败"
      });
    } finally {
      this.setData({
        catalogLoading: false,
        catalogLoadingMore: false
      });
    }
  },

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
          tags: (meta && meta.item && meta.item.tags) || row.tags || [],
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
    const idx = Number(e.currentTarget.dataset.index);
    if (!Number.isInteger(idx) || idx < 0) return;
    const item = this.data.results[idx];
    if (!item || !item.styleCode) return;
    this.previewStyleImages(item.styleCode, item.imageUrl);
  },

  previewCatalogProduct(e) {
    const idx = Number(e.currentTarget.dataset.index);
    if (!Number.isInteger(idx) || idx < 0) return;
    const item = this.data.catalogResults[idx];
    if (!item || !item.styleCode) return;
    this.previewStyleImages(item.styleCode, item.coverImageUrl, idx);
  },

  async previewStyleImages(styleCode, fallbackUrl, catalogIndex = -1) {
    try {
      const detail = await fetchCatalogProductDetail(styleCode);
      const urls = buildPreviewUrls(detail);
      const current = urls[0] || fallbackUrl || "";
      if (!current) return;
      if (catalogIndex >= 0 && urls.length) {
        this.setData({
          [`catalogResults[${catalogIndex}].images`]: detail.images || [],
          [`catalogResults[${catalogIndex}].previewUrls`]: urls
        });
      }
      wx.previewImage({ current, urls: urls.length ? urls : [current] });
    } catch (_err) {
      if (!fallbackUrl) return;
      wx.previewImage({ current: fallbackUrl, urls: [fallbackUrl] });
    }
  },

  async onResultImageError(e) {
    const idx = Number(e.currentTarget.dataset.index);
    const listType = e.currentTarget.dataset.list || "results";
    if (!Number.isInteger(idx) || idx < 0) return;
    const list = this.data[listType] || [];
    const current = list[idx];
    if (!current || !current.imageName) return;

    const maxRetries = Number((config.imageLoadRetry || {}).maxRetries || 0);
    const tried = Number(current.imageRetryCount || 0);
    if (tried >= maxRetries) return;

    try {
      const refreshed = await fetchSignedImageUrl(current.imageName);
      const keyUrl = `${listType}[${idx}].imageUrl`;
      const keyRetry = `${listType}[${idx}].imageRetryCount`;
      this.setData({
        [keyUrl]: refreshed.image_url || current.imageUrl,
        [keyRetry]: tried + 1
      });
    } catch (err) {
      const keyRetry = `${listType}[${idx}].imageRetryCount`;
      this.setData({ [keyRetry]: tried + 1 });
    }
  },

  async onCatalogImageError(e) {
    const idx = Number(e.currentTarget.dataset.index);
    if (!Number.isInteger(idx) || idx < 0) return;
    const current = this.data.catalogResults[idx];
    if (!current || !current.coverImage) return;

    const maxRetries = Number((config.imageLoadRetry || {}).maxRetries || 0);
    const tried = Number(current.imageRetryCount || 0);
    if (tried >= maxRetries) return;

    try {
      const refreshed = await fetchSignedImageUrl(current.coverImage);
      this.setData({
        [`catalogResults[${idx}].coverImageUrl`]: refreshed.image_url || current.coverImageUrl,
        [`catalogResults[${idx}].imageRetryCount`]: tried + 1
      });
    } catch (_err) {
      this.setData({ [`catalogResults[${idx}].imageRetryCount`]: tried + 1 });
    }
  }
});
