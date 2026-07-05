const {
  fetchCatalogTags,
  fetchCatalogProducts,
  replaceCatalogProductTags,
  uploadCatalogImportFiles,
  fetchCatalogImportJob,
  commitCatalogImport,
} = require("../../utils/api");

function typedTag(kind, value) {
  const clean = String(value || "").trim();
  return clean ? `${kind}:${clean}` : "";
}

function statusText(status) {
  return status === "ok" ? "已识别" : "需确认";
}

function displayTags(tags) {
  return (tags || []).map((tag) => String(tag || "").trim()).filter(Boolean).join("，");
}

function firstOf(list) {
  return Array.isArray(list) && list.length ? String(list[0] || "").trim() : "";
}

function deriveYearFromFilename(filename) {
  const raw = String(filename || "").trim();
  const stem = raw.replace(/\.[^.]+$/, "");
  const styleCode = stem.indexOf("_") >= 0 ? stem.slice(0, stem.lastIndexOf("_")) : stem;
  const prefix = styleCode.split("-", 1)[0] || "";
  const match = prefix.match(/(\d{2})$/);
  return match ? `20${match[1]}` : "";
}

function uniq(list) {
  const seen = new Set();
  const out = [];
  (list || []).forEach((item) => {
    const value = String(item || "").trim();
    if (!value || seen.has(value)) return;
    seen.add(value);
    out.push(value);
  });
  return out;
}

function compressImage(filePath) {
  return new Promise((resolve) => {
    if (!filePath) return resolve(filePath);
    wx.compressImage({
      src: filePath,
      quality: 78,
      success: (res) => resolve(res.tempFilePath || filePath),
      fail: () => resolve(filePath),
    });
  });
}

Page({
  data: {
    mode: "tags",

    query: "",
    catalogTagGroups: { year: [], category: [], subcategory: [] },
    categoryOptions: [],
    subcategoryOptions: [],
    products: [],
    productLimit: 9,
    productOffset: 0,
    productHasMore: true,
    productLoading: false,
    productLoadingMore: false,
    productMessage: "",
    editingStyleCode: "",
    editModalOpen: false,
    editYear: "",
    editCategory: "",
    editSubcategory: "",
    savingTags: false,
    tagMessage: "",

    selectedFiles: [],
    jobId: "",
    jobStatus: "",
    jobMessage: "",
    items: [],
    batchCategory: "",
    batchSubcategory: "暂无",
    uploading: false,
    committing: false,
    canCommit: false,
    commitMessage: "",
  },

  onUnload() {
    this.stopPolling();
  },

  onLoad() {
    this.loadTagOptions();
    this.loadProducts(true);
  },

  onReachBottom() {
    if (this.data.mode !== "tags") return;
    if (this.data.productLoading || this.data.productLoadingMore || !this.data.productHasMore) return;
    this.loadProducts(false);
  },

  switchMode(e) {
    const mode = e.currentTarget.dataset.mode;
    if (mode === "tags" || mode === "import") this.setData({ mode });
  },

  onQueryInput(e) {
    this.setData({ query: e.detail.value || "" });
  },

  async loadTagOptions() {
    try {
      const res = await fetchCatalogTags();
      const groups = res.tag_groups || {};
      this.setData({
        catalogTagGroups: groups,
        categoryOptions: uniq((groups.category || []).concat(["单品", "罗纹", "毛织配件", "布匹"])),
        subcategoryOptions: uniq((groups.subcategory || []).concat(["暂无"])),
      });
    } catch (err) {
      console.warn("[catalog-tags:error]", err);
    }
  },

  normalizeProducts(rawProducts) {
    return (rawProducts || []).map((item) => ({
      styleCode: item.style_code || item.styleCode || "",
      coverImageUrl: item.cover_image_url || item.coverImageUrl || "",
      imageCount: (item.images || []).length,
      tags: item.tags || [],
      tagGroups: item.tag_groups || item.tagGroups || {},
      tagsText: displayTags(item.tags || []),
    })).filter((item) => item.styleCode);
  },

  async loadProducts(reset) {
    if (reset ? this.data.productLoading : this.data.productLoadingMore) return;
    const offset = reset ? 0 : Number(this.data.productOffset || 0);
    const limit = Number(this.data.productLimit || 9);
    this.setData({
      productLoading: reset,
      productLoadingMore: !reset,
      productMessage: reset ? "" : this.data.productMessage,
      products: reset ? [] : this.data.products,
      productOffset: reset ? 0 : this.data.productOffset,
      productHasMore: reset ? true : this.data.productHasMore,
      editingStyleCode: reset ? "" : this.data.editingStyleCode,
      editModalOpen: reset ? false : this.data.editModalOpen,
    });
    try {
      const res = await fetchCatalogProducts({
        style_code: this.data.query,
        limit,
        offset,
      });
      const list = this.normalizeProducts(res.products || []);
      const merged = reset ? list : (this.data.products || []).concat(list);
      this.setData({
        products: merged,
        productOffset: merged.length,
        productHasMore: list.length >= limit,
        productMessage: merged.length ? "" : (this.data.query ? "没有找到款号" : "暂无款图"),
      });
    } catch (err) {
      this.setData({ productMessage: err.message || "查询失败" });
    } finally {
      this.setData({ productLoading: false, productLoadingMore: false });
    }
  },

  searchProducts() {
    this.loadProducts(true);
  },

  loadMoreProducts() {
    this.loadProducts(false);
  },

  editProductTags(e) {
    const index = Number(e.currentTarget.dataset.index);
    const item = this.data.products[index];
    if (!item) return;
    const groups = item.tagGroups || {};
    this.setData({
      editingStyleCode: item.styleCode,
      editModalOpen: true,
      editYear: firstOf(groups.year),
      editCategory: firstOf(groups.category),
      editSubcategory: firstOf(groups.subcategory) || "暂无",
      tagMessage: "",
    });
  },

  onEditFieldInput(e) {
    const key = e.currentTarget.dataset.key;
    if (!key) return;
    this.setData({ [key]: e.detail.value || "" });
  },

  selectEditTag(e) {
    const kind = e.currentTarget.dataset.kind;
    const value = String(e.currentTarget.dataset.value || "");
    if (kind === "category") this.setData({ editCategory: value });
    if (kind === "subcategory") this.setData({ editSubcategory: value });
  },

  cancelEditTags() {
    this.setData({
      editingStyleCode: "",
      editModalOpen: false,
      editYear: "",
      editCategory: "",
      editSubcategory: "",
      tagMessage: "",
    });
  },

  noop() {},

  buildEditTags() {
    return [
      String(this.data.editYear || "").trim(),
      typedTag("category", this.data.editCategory),
      typedTag("subcategory", this.data.editSubcategory),
    ].filter(Boolean);
  },

  async saveTags() {
    if (!this.data.editingStyleCode || this.data.savingTags) return;
    this.setData({ savingTags: true, tagMessage: "正在保存标签..." });
    try {
      const tags = this.buildEditTags();
      const res = await replaceCatalogProductTags(this.data.editingStyleCode, tags);
      const products = (this.data.products || []).map((item) => {
        if (item.styleCode !== this.data.editingStyleCode) return item;
        return Object.assign({}, item, {
          tags: res.tags || tags,
          tagGroups: res.tag_groups || item.tagGroups || {},
          tagsText: displayTags(res.tags || tags),
        });
      });
      this.setData({ products, savingTags: false, tagMessage: "标签已保存", editModalOpen: false });
      wx.showToast({ title: "标签已保存", icon: "none" });
    } catch (err) {
      this.setData({ savingTags: false, tagMessage: err.message || "保存失败" });
    }
  },

  chooseImportImages() {
    if (this.data.uploading) return;
    wx.chooseMedia({
      count: 9,
      mediaType: ["image"],
      success: async (res) => {
        const files = (res.tempFiles || []).filter((item) => item && item.tempFilePath);
        if (!files.length) return;
        const compressed = [];
        for (let i = 0; i < files.length; i += 1) {
          compressed.push({
            tempFilePath: await compressImage(files[i].tempFilePath),
            name: `upload_${i + 1}.jpg`,
          });
        }
        this.setData({
          selectedFiles: compressed,
        });
        await this.uploadSelectedFiles(compressed);
      },
      fail: () => wx.showToast({ title: "未选择图片", icon: "none" }),
    });
  },

  async uploadSelectedFiles(files) {
    this.stopPolling();
    this.setData({
      uploading: true,
      jobId: "",
      jobStatus: "",
      jobMessage: `正在上传 ${files.length} 张图片...`,
      items: [],
      commitMessage: "",
      canCommit: false,
    });
    try {
      const payload = files.map((item, index) => ({
        tempFilePath: item.tempFilePath,
        name: `upload_${index + 1}.jpg`,
      }));
      const res = await uploadCatalogImportFiles(payload);
      const jobId = String(res.job_id || "");
      this.setData({ jobId, uploading: false, jobMessage: "正在识别款号..." });
      if (jobId) this.pollJob();
    } catch (err) {
      this.setData({ uploading: false, jobMessage: err.message || "上传失败" });
    }
  },

  stopPolling() {
    if (this._pollTimer) {
      clearTimeout(this._pollTimer);
      this._pollTimer = null;
    }
  },

  async pollJob() {
    const jobId = this.data.jobId;
    if (!jobId) return;
    try {
      const job = await fetchCatalogImportJob(jobId);
      this.applyJob(job);
      if (job.status === "pending" || job.status === "running") {
        this._pollTimer = setTimeout(() => this.pollJob(), 900);
      }
    } catch (err) {
      this.setData({ jobMessage: err.message || "导入进度查询失败" });
    }
  },

  applyJob(job) {
    const items = (job.items || []).map((item) => ({
      source_rel_path: String(item.source_rel_path || ""),
      source_name: String(item.source_name || ""),
      proposed_style_code: String(item.proposed_style_code || ""),
      target_filename: String(item.target_filename || item.proposed_filename || ""),
      year_tag: String(item.year_tag || item.proposed_year_tag || ""),
      selected: item.selected !== false,
      status: String(item.status || ""),
      statusText: statusText(item.status),
      error: String(item.error || ""),
    }));
    const total = Number(job.total || 0);
    const processed = Number(job.processed || 0);
    const suffix = total ? ` ${processed}/${total}` : "";
    this.setData({
      jobStatus: String(job.status || ""),
      jobMessage: `${job.message || ""}${suffix}`,
      items,
      canCommit: job.status === "completed" && !job.committed && items.some((item) => item.selected),
    });
  },

  onBatchInput(e) {
    const key = e.currentTarget.dataset.key;
    if (!key) return;
    this.setData({ [key]: e.detail.value || "" });
  },

  selectBatchTag(e) {
    const kind = e.currentTarget.dataset.kind;
    const value = String(e.currentTarget.dataset.value || "");
    if (kind === "category") this.setData({ batchCategory: value });
    if (kind === "subcategory") this.setData({ batchSubcategory: value });
  },

  toggleItemSelected(e) {
    const index = Number(e.currentTarget.dataset.index);
    if (!Number.isFinite(index)) return;
    this.setData({ [`items[${index}].selected`]: !this.data.items[index].selected }, () => this.refreshCanCommit());
  },

  onItemInput(e) {
    const index = Number(e.currentTarget.dataset.index);
    const key = e.currentTarget.dataset.key;
    if (!Number.isFinite(index) || !key) return;
    const value = e.detail.value || "";
    const updates = { [`items[${index}].${key}`]: value };
    if (key === "target_filename") {
      const year = deriveYearFromFilename(value);
      if (year) updates[`items[${index}].year_tag`] = year;
    }
    this.setData(updates);
  },

  refreshCanCommit() {
    this.setData({
      canCommit: this.data.jobStatus === "completed" && this.data.items.some((item) => item.selected),
    });
  },

  buildCommitItems() {
    const tags = [
      typedTag("category", this.data.batchCategory),
      typedTag("subcategory", this.data.batchSubcategory),
    ].filter(Boolean);
    return (this.data.items || []).map((item) => ({
      source_rel_path: item.source_rel_path,
      selected: !!item.selected,
      target_filename: String(item.target_filename || "").trim(),
      year_tag: String(item.year_tag || "").trim(),
      tags,
    }));
  },

  async commitImport() {
    if (!this.data.canCommit || this.data.committing) return;
    this.setData({ committing: true, commitMessage: "正在导入..." });
    try {
      const res = await commitCatalogImport(this.data.jobId, this.buildCommitItems());
      const sync = res.sync || {};
      this.setData({
        committing: false,
        canCommit: false,
        commitMessage: `已导入 ${res.imported || 0} 张；新增款 ${sync.products_added || 0}，新增/更新图 ${sync.images_added_or_updated || 0}`,
      });
      wx.showToast({ title: "导入完成", icon: "none" });
    } catch (err) {
      this.setData({ committing: false, commitMessage: err.message || "导入失败" });
    }
  },
});
