const {
  fetchPrintTemplates,
  printUpload,
  renderPrintLayout,
  toPrintAbsoluteUrl
} = require("../../utils/api");

function downloadToLocal(url) {
  return new Promise((resolve) => {
    if (!url) return resolve("");
    wx.downloadFile({
      url,
      success: (res) => resolve(res.tempFilePath || url),
      fail: () => resolve(url)
    });
  });
}

Page({
  data: {
    printUploading: false,
    printRendering: false,
    paperOptions: ["A4", "A5", "4R"],
    paperIndex: 0,
    templates: [],
    templateNames: [],
    templateIndex: 0,
    printUploadedImages: [],
    printImageIds: [],
    printPreviewUrl: "",
    printPreviewLocalUrl: "",
    printPdfUrl: ""
  },

  async onLoad() {
    await this.loadPrintTemplates();
  },

  goSearchPage() {
    wx.navigateBack({
      fail: () => wx.reLaunch({ url: "/pages/index/index" })
    });
  },

  goPrintPage() {},

  async loadPrintTemplates() {
    try {
      const templates = await fetchPrintTemplates();
      const templateNames = templates.map((item) => `${item.name}（${item.slots}）`);
      const defaultIdx = templates.findIndex((item) => item.template_id === "single_full");
      this.setData({
        templates,
        templateNames,
        templateIndex: defaultIdx >= 0 ? defaultIdx : 0
      });
    } catch (err) {
      wx.showToast({ title: err.message || "加载模板失败", icon: "none" });
    }
  },

  onPaperChange(e) {
    this.setData({ paperIndex: Number(e.detail.value) });
  },

  onTemplateChange(e) {
    this.setData({ templateIndex: Number(e.detail.value) });
  },

  pickPrintImages() {
    wx.chooseMedia({
      count: 9,
      mediaType: ["image"],
      success: async (res) => {
        await this.uploadPrintTempFiles(res.tempFiles || []);
      },
      fail: () => wx.showToast({ title: "未选择图片", icon: "none" })
    });
  },

  async uploadPrintTempFiles(tempFiles) {
    if (!tempFiles.length) return;

    this.setData({ printUploading: true });
    try {
      const uploadedBatch = [];
      const ids = [...this.data.printImageIds];

      for (const file of tempFiles) {
        const payload = await printUpload(file.tempFilePath);
        const processedUrl = toPrintAbsoluteUrl(payload.processed_url);
        const localUrl = await downloadToLocal(processedUrl);
        uploadedBatch.push({
          image_id: payload.image_id,
          processed_url: processedUrl,
          display_url: localUrl
        });
        ids.push(payload.image_id);
      }

      this.setData({
        printUploadedImages: [...this.data.printUploadedImages, ...uploadedBatch],
        printImageIds: ids
      });
      wx.showToast({ title: `已上传 ${uploadedBatch.length} 张`, icon: "none" });
    } catch (err) {
      wx.showToast({ title: err.message || "上传失败", icon: "none" });
    } finally {
      this.setData({ printUploading: false });
    }
  },

  removePrintImage(e) {
    const imageId = e.currentTarget.dataset.id;
    const printUploadedImages = this.data.printUploadedImages.filter((item) => item.image_id !== imageId);
    const printImageIds = this.data.printImageIds.filter((id) => id !== imageId);
    this.setData({ printUploadedImages, printImageIds });
  },

  async renderPrintCollage() {
    const { printImageIds, paperOptions, paperIndex, templates, templateIndex, printRendering } = this.data;
    if (printRendering) return;
    if (printImageIds.length === 0 || templates.length === 0) {
      wx.showToast({ title: "请先上传图片", icon: "none" });
      return;
    }

    this.setData({ printRendering: true });
    try {
      const payload = {
        paper_size: paperOptions[paperIndex],
        template_id: templates[templateIndex].template_id,
        placements: printImageIds.map((id, idx) => ({ image_id: id, slot_index: idx })),
        auto_fill: true
      };

      const res = await renderPrintLayout(payload);
      const previewUrl = toPrintAbsoluteUrl(res.preview_url);
      const pdfUrl = toPrintAbsoluteUrl(res.pdf_url);
      const previewLocalUrl = await downloadToLocal(previewUrl);
      this.setData({
        printPreviewUrl: previewUrl,
        printPreviewLocalUrl: previewLocalUrl,
        printPdfUrl: pdfUrl
      });
      wx.showToast({ title: "已生成拼图预览", icon: "none" });
    } catch (err) {
      wx.showToast({ title: err.message || "生成失败", icon: "none" });
    } finally {
      this.setData({ printRendering: false });
    }
  },

  openPrintPdf() {
    if (!this.data.printPdfUrl) return;
    wx.showLoading({ title: "下载中..." });
    wx.downloadFile({
      url: this.data.printPdfUrl,
      success: (res) => {
        wx.hideLoading();
        wx.openDocument({
          filePath: res.tempFilePath,
          fileType: "pdf",
          showMenu: true,
          success: () => {
            const sys = (wx.getSystemInfoSync().system || "").toLowerCase();
            if (sys.includes("ios")) {
              wx.showToast({ title: "右上角菜单可打印/用其他应用打开", icon: "none", duration: 2500 });
            }
          }
        });
      },
      fail: () => {
        wx.hideLoading();
        wx.showToast({ title: "打开 PDF 失败", icon: "none" });
      }
    });
  },

  copyPrintPdfLink() {
    if (!this.data.printPdfUrl) return;
    wx.setClipboardData({
      data: this.data.printPdfUrl,
      success: () => wx.showToast({ title: "PDF 链接已复制", icon: "none" })
    });
  },

  onPrintPreviewError() {
    wx.showToast({ title: "预览加载失败，请检查配置", icon: "none" });
  }
});
