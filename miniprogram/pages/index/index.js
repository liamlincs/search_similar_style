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

function buildCatalogTagSections(tagGroups, selectedGroups) {
  const groups = tagGroups || {};
  const selected = selectedGroups || {};
  return [
    { key: "year", title: "年份", tags: groups.year || [], selected: selected.year || [] },
    { key: "category", title: "类别", tags: groups.category || [], selected: selected.category || [] },
    { key: "subcategory", title: "细类", tags: groups.subcategory || [], selected: selected.subcategory || [] }
  ].map((section) => ({
    key: section.key,
    title: section.title,
    items: buildCatalogTagItems(section.tags, section.selected)
  })).filter((section) => section.items.length);
}

function buildPreviewUrls(product) {
  const imageUrls = (product.images || [])
    .map((item) => item.image_url || item.imageUrl || "")
    .filter(Boolean);
  if (imageUrls.length) return uniqTags(imageUrls);
  const cover = product.coverImageUrl || product.cover_image_url || product.imageUrl || "";
  return cover ? [cover] : [];
}

function buildCatalogFilterKey(query, tags, tagGroups) {
  const q = String(query || "").trim();
  const tagKey = (tags || []).map((tag) => String(tag || "").trim()).filter(Boolean).sort().join("|");
  const groups = tagGroups || {};
  const groupKey = ["year", "category", "subcategory"]
    .map((key) => (groups[key] || []).map((tag) => String(tag || "").trim()).filter(Boolean).sort().join("|"))
    .join("::");
  return `${q}::${tagKey}::${groupKey}`;
}

Page({
  data: {
    pageMode: "image",
    localImage: "",
    imageInfo: null,
    regionMode: false,
    regionBox: null,
    cropRect: null,
    cropStart: null,
    searching: false,
    hasSearched: false,
    errorMessage: "",
    results: [],
    isAmbiguous: false,
    confidenceBand: "low",
    catalogQuery: "",
    catalogTags: [],
    catalogTagItems: [],
    catalogTagGroups: { year: [], category: [], subcategory: [] },
    catalogTagSections: [],
    selectedCatalogTagGroups: { year: [], category: [], subcategory: [] },
    selectedCatalogTags: [],
    catalogLoading: false,
    catalogLoadingMore: false,
    catalogHasMore: true,
    catalogLimit: 9,
    catalogOffset: 0,
    catalogRequestSeq: 0,
    catalogActiveFilterKey: "",
    catalogErrorMessage: "",
    catalogResults: [],
    experienceVersionEnabled: !!config.enableExperienceVersion
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
    if (mode === "catalog" && !this.data.catalogResults.length && !this.data.catalogLoading) {
      this.searchCatalog(true);
    }
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

  searchFullImage() {
    const filePath = this.data.localImage;
    if (!filePath || this.data.searching) return;
    this.setData({ regionMode: false, cropStart: null });
    this.search(filePath);
  },

  loadLocalImageInfo(filePath) {
    wx.getImageInfo({
      src: filePath,
      success: (info) => {
        this.setData({ imageInfo: { width: Number(info.width || 0), height: Number(info.height || 0) } });
      },
      fail: () => {
        this.setData({ imageInfo: null });
      }
    });
  },

  enableRegionSelect() {
    if (!this.data.localImage || this.data.searching) return;
    this.ensureRegionBox(() => {
      const box = this.data.regionBox || {};
      const w = Number(box.width || 0);
      const h = Number(box.height || 0);
      if (!w || !h) return;
      const defaultRect = this.data.cropRect || {
        left: Math.round(w * 0.25),
        top: Math.round(h * 0.25),
        width: Math.round(w * 0.5),
        height: Math.round(h * 0.5)
      };
      this.setData({ regionMode: true, cropRect: defaultRect });
    });
  },

  clearRegionSelect() {
    this.setData({ regionMode: false, cropRect: null, cropStart: null });
  },

  ensureRegionBox(callback) {
    wx.createSelectorQuery()
      .in(this)
      .select(".query-wrap")
      .boundingClientRect((rect) => {
        if (rect) {
          this.setData({
            regionBox: {
              left: Number(rect.left || 0),
              top: Number(rect.top || 0),
              width: Number(rect.width || 0),
              height: Number(rect.height || 0)
            }
          }, callback);
        } else if (callback) {
          callback();
        }
      })
      .exec();
  },

  pointInRegionBox(touch) {
    const box = this.data.regionBox || {};
    const x = Number(touch.clientX || 0) - Number(box.left || 0);
    const y = Number(touch.clientY || 0) - Number(box.top || 0);
    return {
      x: Math.max(0, Math.min(Number(box.width || 0), x)),
      y: Math.max(0, Math.min(Number(box.height || 0), y))
    };
  },

  onRegionTouchStart(e) {
    if (!this.data.regionMode || this.data.searching) return;
    const touch = e.touches && e.touches[0];
    if (!touch) return;
    this.ensureRegionBox(() => {
      const p = this.pointInRegionBox(touch);
      this.setData({
        cropStart: p,
        cropRect: { left: p.x, top: p.y, width: 1, height: 1 }
      });
    });
  },

  onRegionTouchMove(e) {
    if (!this.data.regionMode || !this.data.cropStart) return;
    const touch = e.touches && e.touches[0];
    if (!touch) return;
    const start = this.data.cropStart;
    const p = this.pointInRegionBox(touch);
    this.setData({
      cropRect: {
        left: Math.min(start.x, p.x),
        top: Math.min(start.y, p.y),
        width: Math.abs(p.x - start.x),
        height: Math.abs(p.y - start.y)
      }
    });
  },

  onRegionTouchEnd() {
    if (!this.data.regionMode) return;
    this.setData({ cropStart: null });
  },

  buildCropRatio() {
    const rect = this.data.cropRect;
    const box = this.data.regionBox;
    const info = this.data.imageInfo;
    if (!rect || !box || !info || !info.width || !info.height) return null;
    if (Number(rect.width || 0) < 20 || Number(rect.height || 0) < 20) return null;

    const scale = Math.min(box.width / info.width, box.height / info.height);
    const drawW = info.width * scale;
    const drawH = info.height * scale;
    const offsetX = (box.width - drawW) / 2;
    const offsetY = (box.height - drawH) / 2;
    const left = Math.max(rect.left, offsetX);
    const top = Math.max(rect.top, offsetY);
    const right = Math.min(rect.left + rect.width, offsetX + drawW);
    const bottom = Math.min(rect.top + rect.height, offsetY + drawH);
    if (right - left < 20 || bottom - top < 20) return null;
    return {
      x: Math.max(0, Math.min(1, (left - offsetX) / drawW)),
      y: Math.max(0, Math.min(1, (top - offsetY) / drawH)),
      w: Math.max(0, Math.min(1, (right - left) / drawW)),
      h: Math.max(0, Math.min(1, (bottom - top) / drawH))
    };
  },

  searchSelectedRegion() {
    if (!this.data.localImage || this.data.searching) return;
    this.ensureRegionBox(() => {
      const crop = this.buildCropRatio();
      if (!crop) {
        wx.showToast({ title: "请框选更大的区域", icon: "none" });
        return;
      }
      this.search(this.data.localImage, { crop });
    });
  },

  goPrintPage() {
    wx.navigateTo({ url: "/pages/print/index" });
  },

  goRecolorPage() {
    wx.navigateTo({ url: "/pages/recolor/index" });
  },

  goSearchPage() {},

  openProductLibraryH5() {
    const baseUrl = String(config.baseUrl || "").replace(/\/+$/, "");
    const path = config.catalogH5Path || "/catalog";
    const token = config.catalogH5Token || config.apiKey || "";
    if (!baseUrl || !token) {
      wx.showToast({ title: "请先配置产品库体验 token", icon: "none" });
      return;
    }
    const query = `type=product&token=${encodeURIComponent(token)}`;
    const url = `${baseUrl}${path}?${query}`;
    wx.navigateTo({
      url: `/pages/catalog_webview/index?title=${encodeURIComponent("产品库")}&url=${encodeURIComponent(url)}`
    });
  },

  onCatalogQueryInput(e) {
    this.setData({ catalogQuery: e.detail.value || "" });
  },

  toggleCatalogTag(e) {
    const tag = e.currentTarget.dataset.tag || "";
    const group = e.currentTarget.dataset.group || "";
    if (!tag) return;
    if (group) {
      const selectedGroups = Object.assign({ year: [], category: [], subcategory: [] }, this.data.selectedCatalogTagGroups || {});
      const current = selectedGroups[group] || [];
      selectedGroups[group] = current.includes(tag)
        ? current.filter((item) => item !== tag)
        : uniqTags(current.concat(tag));
      this.setData({
        selectedCatalogTagGroups: selectedGroups,
        catalogTagSections: buildCatalogTagSections(this.data.catalogTagGroups, selectedGroups)
      });
      this.searchCatalog(true);
      return;
    }
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
    const emptyGroups = { year: [], category: [], subcategory: [] };
    this.setData({
      catalogQuery: "",
      selectedCatalogTagGroups: emptyGroups,
      selectedCatalogTags: [],
      catalogTagSections: buildCatalogTagSections(this.data.catalogTagGroups, emptyGroups),
      catalogTagItems: buildCatalogTagItems(this.data.catalogTags, []),
      catalogErrorMessage: ""
    });
    this.searchCatalog(true);
  },

  clearCatalogTags() {
    const emptyGroups = { year: [], category: [], subcategory: [] };
    this.setData({
      selectedCatalogTagGroups: emptyGroups,
      selectedCatalogTags: [],
      catalogTagSections: buildCatalogTagSections(this.data.catalogTagGroups, emptyGroups),
      catalogTagItems: buildCatalogTagItems(this.data.catalogTags, []),
      catalogErrorMessage: ""
    });
    this.searchCatalog(true);
  },

  async loadCatalogFilters() {
    try {
      const resp = await fetchCatalogTags();
      const tags = resp.tags || [];
      const tagGroups = resp.tag_groups || { year: [], category: [], subcategory: [] };
      this.setData({
        catalogTags: tags,
        catalogTagGroups: tagGroups,
        catalogTagSections: buildCatalogTagSections(tagGroups, this.data.selectedCatalogTagGroups || {}),
        catalogTagItems: buildCatalogTagItems(tags, this.data.selectedCatalogTags || [])
      });
    } catch (_err) {}
  },

  async searchCatalog(reset = true) {
    if (!reset && (this.data.catalogLoading || this.data.catalogLoadingMore)) return;
    const offset = reset ? 0 : Number(this.data.catalogOffset || 0);
    const limit = Number(this.data.catalogLimit || 20);
    const query = this.data.catalogQuery;
    const tags = [...(this.data.selectedCatalogTags || [])];
    const selectedGroups = Object.assign({ year: [], category: [], subcategory: [] }, this.data.selectedCatalogTagGroups || {});
    const filterKey = buildCatalogFilterKey(query, tags, selectedGroups);
    const requestSeq = Number(this.data.catalogRequestSeq || 0) + 1;
    this.setData({
      catalogLoading: reset,
      catalogLoadingMore: !reset,
      catalogRequestSeq: requestSeq,
      catalogActiveFilterKey: filterKey,
      catalogResults: reset ? [] : this.data.catalogResults,
      catalogOffset: reset ? 0 : this.data.catalogOffset,
      catalogHasMore: reset ? true : this.data.catalogHasMore,
      catalogErrorMessage: reset ? "" : this.data.catalogErrorMessage
    });
    try {
      const resp = await fetchCatalogProducts({
        style_code: query,
        tags,
        year_tags: selectedGroups.year,
        category_tags: selectedGroups.category,
        subcategory_tags: selectedGroups.subcategory,
        limit,
        offset
      });
      if (
        requestSeq !== Number(this.data.catalogRequestSeq || 0) ||
        filterKey !== buildCatalogFilterKey(this.data.catalogQuery, this.data.selectedCatalogTags || [], this.data.selectedCatalogTagGroups || {})
      ) {
        return;
      }
      const list = (resp.products || []).map((item) => ({
        styleCode: item.style_code || "",
        coverImage: item.cover_image || "",
        coverImageUrl: item.cover_image_url || "",
        imageRetryCount: 0,
        tags: item.tags || [],
        tagGroups: item.tag_groups || {},
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
      if (
        requestSeq !== Number(this.data.catalogRequestSeq || 0) ||
        filterKey !== buildCatalogFilterKey(this.data.catalogQuery, this.data.selectedCatalogTags || [], this.data.selectedCatalogTagGroups || {})
      ) {
        return;
      }
      this.setData({
        catalogResults: reset ? [] : this.data.catalogResults,
        catalogErrorMessage: err.message || "款库检索失败"
      });
    } finally {
      if (requestSeq === Number(this.data.catalogRequestSeq || 0)) {
        this.setData({
          catalogLoading: false,
          catalogLoadingMore: false
        });
      }
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
          imageInfo: null,
          regionMode: false,
          regionBox: null,
          cropRect: null,
          cropStart: null,
          hasSearched: false,
          errorMessage: "",
          results: [],
          isAmbiguous: false,
          confidenceBand: "low"
        });
        this.loadLocalImageInfo(file.tempFilePath);
      },
      fail: () => {
        wx.showToast({ title: "已取消选择", icon: "none" });
      }
    });
  },

  async search(filePath, options = {}) {
    this.setData({ searching: true, errorMessage: "" });
    try {
      const resp = await uploadAndSearch(filePath, options);
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

      const rawList = srcList.map((row, idx) => {
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
      });
      const seenStyleCodes = new Set();
      const list = [];
      rawList.forEach((item) => {
        const key = String(item.styleCode || "").trim().toUpperCase();
        if (!key || seenStyleCodes.has(key) || list.length >= 9) return;
        seenStyleCodes.add(key);
        list.push({
          ...item,
          rank: list.length + 1
        });
      });

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
      const refreshed = await fetchSignedImageUrl(current.coverImage, { kind: "catalog" });
      this.setData({
        [`catalogResults[${idx}].coverImageUrl`]: refreshed.image_url || current.coverImageUrl,
        [`catalogResults[${idx}].imageRetryCount`]: tried + 1
      });
    } catch (_err) {
      this.setData({ [`catalogResults[${idx}].imageRetryCount`]: tried + 1 });
    }
  }
});
