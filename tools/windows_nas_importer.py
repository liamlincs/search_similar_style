import argparse
import json
import logging
import re
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from catalog_store import filename_to_style_code

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _clean_style_code(value: str) -> str:
    code = str(value or "").strip().upper()
    if code.endswith("#"):
        code = code[:-1]
    return SAFE_STEM_RE.sub("_", code).strip("_")


def _is_valid_style_code(value: str) -> bool:
    code = _clean_style_code(value)
    return bool(code) and bool(re.match(r"^[A-Za-z]", code))


def _sanitize_filename(filename: str, fallback_suffix: str = ".jpg") -> str:
    raw = Path(str(filename or "").replace("\\", "/")).name
    stem = SAFE_STEM_RE.sub("_", Path(raw).stem).strip("_")
    suffix = Path(raw).suffix.lower() or fallback_suffix.lower()
    if suffix not in IMAGE_EXTS:
        suffix = fallback_suffix.lower()
    if not stem:
        stem = "UNKNOWN"
    return f"{stem}{suffix}"


def _split_tags(value: Any) -> list[str]:
    out: list[str] = []
    seen = set()
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[、,，\s]+", str(value or ""))
    for item in raw_items:
        tag = str(item or "").strip()
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


def _next_target_name(prefix: str, suffix: str, used_names: set[str], next_seq: dict[str, int]) -> str:
    clean = SAFE_STEM_RE.sub("_", str(prefix or "").strip()).strip("_") or "UNKNOWN"
    seq = int(next_seq.get(clean, 0))
    suffix = suffix.lower() if suffix.lower() in IMAGE_EXTS else ".jpg"
    while seq < 100000:
        candidate = f"{clean}_{seq:03d}{suffix}"
        key = candidate.lower()
        if key not in used_names:
            used_names.add(key)
            next_seq[clean] = seq + 1
            return candidate
        seq += 1
    raise RuntimeError(f"cannot allocate filename for {clean}")


def _code_to_filename_prefix(code: str) -> str:
    core = str(code or "")
    if core.endswith("#"):
        core = core[:-1]
    core = SAFE_STEM_RE.sub("_", core).strip("_")
    return core if core else "UNKNOWN"


def _build_name_allocator(target_dir: Path) -> tuple[set[str], dict[str, int]]:
    used = set()
    next_seq: dict[str, int] = {}
    if target_dir.exists():
        for path in sorted(target_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            used.add(path.name.lower())
            stem = path.stem
            if "_" not in stem:
                continue
            prefix, seq_text = stem.rsplit("_", 1)
            if seq_text.isdigit():
                next_seq[prefix] = max(next_seq.get(prefix, 0), int(seq_text) + 1)
    return used, next_seq


def _scan_images(source_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def _extract_style(path: Path, tesseract_bin: str | None) -> str:
    from extract_style_codes import build_header_crops, try_extract_code_from_image

    for crop in build_header_crops(path):
        code = str(try_extract_code_from_image(crop, tesseract_bin) or "").strip()
        if code:
            return code
    return ""


@dataclass
class ImportJob:
    job_id: str
    source_dir: Path
    target_dir: Path
    status: str = "pending"
    message: str = "任务已创建"
    total: int = 0
    processed: int = 0
    items: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False
    committed: bool = False
    result: dict[str, Any] | None = None


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, ImportJob] = {}

    def create(self, source_dir: Path, target_dir: Path) -> ImportJob:
        job = ImportJob(uuid.uuid4().hex, source_dir, target_dir)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> ImportJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in changes.items():
                setattr(job, key, value)

    def snapshot(self, job: ImportJob) -> dict[str, Any]:
        with self._lock:
            return {
                "job_id": job.job_id,
                "source_dir": str(job.source_dir),
                "target_dir": str(job.target_dir),
                "status": job.status,
                "message": job.message,
                "total": job.total,
                "processed": job.processed,
                "items": list(job.items),
                "committed": job.committed,
                "result": job.result or {},
            }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NAS 款图批量入库</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f6f7f9; }
    header { position: sticky; top: 0; z-index: 5; background: #fff; border-bottom: 1px solid #e5e7eb; padding: 14px 18px; }
    h1 { margin: 0; font-size: 20px; }
    .title-row { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
    .top-progress { display: grid; gap: 6px; min-width: min(360px, 45vw); }
    .progress-text { color: #64748b; font-size: 14px; font-weight: 900; text-align: right; }
    .progress-track { height: 8px; border-radius: 999px; background: #e5e7eb; overflow: hidden; }
    .progress-bar { width: 0%; height: 100%; border-radius: inherit; background: #2563eb; transition: width .18s ease; }
    .path-grid { display: grid; grid-template-columns: 1fr 1fr auto; gap: 10px; align-items: end; }
    .scan-actions { display: flex; gap: 8px; align-items: center; }
    label { display: grid; gap: 5px; font-size: 12px; font-weight: 700; color: #4b5563; }
    input[type="text"] { width: 100%; height: 42px; border: 1px solid #cfd5df; border-radius: 6px; padding: 9px 12px; font-size: 14px; background: #fff; }
    button { border: 0; border-radius: 6px; padding: 10px 14px; background: #111827; color: #fff; font-weight: 800; cursor: pointer; min-height: 40px; }
    button.secondary { background: #e5e7eb; color: #111827; }
    button:disabled { opacity: .52; cursor: not-allowed; }
    main { padding: 16px 18px 40px; }
    .status { margin: 0 0 12px; color: #64748b; font-weight: 700; }
    .status:empty { display: none; }
    .status.err { color: #b91c1c; }
    .toolbar { display: flex; gap: 10px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
    .batch-tags { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
    .batch-note { color: #475569; font-weight: 800; margin-bottom: 10px; }
    .batch-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; align-items: start; }
    .batch-field { min-width: 0; }
    .quick-picks { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; align-content: flex-start; min-height: 78px; }
    .quick-pick { min-height: 32px; padding: 5px 12px; border: 1px solid #d7dee8; border-radius: 999px; background: #f8fafc; color: #334155; font-weight: 800; white-space: nowrap; }
    .table-wrap { overflow: auto; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; min-width: 1080px; }
    th, td { border-bottom: 1px solid #e5e7eb; padding: 10px; text-align: left; vertical-align: middle; }
    th { position: sticky; top: 0; background: #f9fafb; z-index: 1; font-size: 13px; }
    tr.warn { background: #fff7ed; }
    tr.err { background: #fef2f2; }
    .source { max-width: 220px; word-break: break-all; color: #374151; font-weight: 700; }
    .source button { appearance: none; border: 0; background: transparent; color: #2563eb; padding: 0; min-height: 0; font: inherit; font-weight: 800; text-align: left; cursor: pointer; }
    .muted { color: #6b7280; font-size: 12px; margin-top: 4px; }
    .pill { display: inline-block; border-radius: 999px; padding: 4px 9px; font-size: 12px; font-weight: 800; background: #e5f3ff; color: #075985; }
    .pill.err { background: #fee2e2; color: #b91c1c; }
    .pill.warn { background: #ffedd5; color: #c2410c; }
    .modal { position: fixed; inset: 0; display: none; place-items: center; background: rgba(17,24,39,.48); z-index: 20; padding: 20px; }
    .modal.open { display: grid; }
    .modal-box { width: min(520px, 100%); border-radius: 8px; background: #fff; box-shadow: 0 18px 50px rgba(0,0,0,.22); padding: 18px; }
    .modal-box.image-box { width: min(920px, 100%); }
    .modal-message { white-space: pre-line; line-height: 1.55; font-weight: 700; color: #111827; }
    .modal-image { display: none; width: 100%; max-height: 76vh; object-fit: contain; background: #f3f4f6; border-radius: 6px; margin-bottom: 12px; }
    .modal-actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 16px; }
    @media (max-width: 900px) {
      .path-grid { grid-template-columns: 1fr; }
      .batch-grid { grid-template-columns: 1fr; }
      .quick-picks { min-height: 0; }
      .title-row { display: grid; }
      .top-progress { min-width: 0; width: 100%; }
      .progress-text { text-align: left; }
      header { position: static; }
    }
  </style>
</head>
<body>
  <header>
    <div class="title-row">
      <h1>NAS 款图批量入库</h1>
      <div class="top-progress" aria-label="扫描进度">
        <div id="progressText" class="progress-text">已处理 0/0</div>
        <div class="progress-track"><div id="progressBar" class="progress-bar"></div></div>
      </div>
    </div>
    <div class="path-grid">
      <label>源目录
        <input id="sourceDir" type="text" placeholder="例如 Z:\2018\成衣" />
      </label>
      <label>目标目录
        <input id="targetDir" type="text" />
      </label>
      <div class="scan-actions">
        <button id="scanBtn" type="button">扫描识别</button>
        <button id="cancelScanBtn" type="button" class="secondary" disabled>停止扫描</button>
      </div>
    </div>
  </header>
  <main>
    <div id="status" class="status"></div>
    <div class="batch-tags">
      <div class="batch-note">批量标签：年份、类别、细类会统一加到本次勾选导入的图片所属款号。</div>
      <div class="batch-grid">
        <label class="batch-field">年份
          <input id="batchYear" type="text" placeholder="如 2024" />
          <div id="batchYearPicks" class="quick-picks"></div>
        </label>
        <label class="batch-field">类别
          <input id="batchCategory" type="text" placeholder="如 单品、罗纹、毛织配件、布匹" />
          <div id="batchCategoryPicks" class="quick-picks"></div>
        </label>
        <label class="batch-field">细类
          <input id="batchSubcategory" type="text" placeholder="如 暂无，或输入新增细类" />
          <div id="batchSubcategoryPicks" class="quick-picks"></div>
        </label>
      </div>
    </div>
    <div class="toolbar">
      <button id="toggleBtn" type="button" class="secondary">全选/反选</button>
      <button id="commitBtn" type="button">确认复制入库</button>
      <span class="muted" id="countText"></span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>导入</th>
            <th>源文件</th>
            <th>款号</th>
            <th>导入后文件名</th>
            <th>状态</th>
          </tr>
        </thead>
        <tbody id="rows">
          <tr><td colspan="5" class="muted">填写源目录后开始扫描。</td></tr>
        </tbody>
      </table>
    </div>
  </main>
  <div id="modal" class="modal">
    <div id="modalBox" class="modal-box">
      <img id="modalImage" class="modal-image" alt="图片预览" />
      <div id="modalMessage" class="modal-message"></div>
      <div class="modal-actions">
        <button id="modalOk" type="button">确定</button>
      </div>
    </div>
  </div>
  <script>
    const DEFAULT_SOURCE = "__DEFAULT_SOURCE__";
    const DEFAULT_TARGET = "__DEFAULT_TARGET__";
    const $ = (id) => document.getElementById(id);
    let currentJob = null;
    let pollTimer = null;

    const PATH_MEMORY_KEYS = {
      source: "nas_importer_source_dir",
      target: "nas_importer_target_dir",
    };
    $("sourceDir").value = localStorage.getItem(PATH_MEMORY_KEYS.source) || DEFAULT_SOURCE;
    $("targetDir").value = localStorage.getItem(PATH_MEMORY_KEYS.target) || DEFAULT_TARGET;
    const QUICK_YEARS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"];
    const QUICK_CATEGORIES = ["单品", "罗纹", "毛织配件", "布匹"];
    const QUICK_SUBCATEGORIES = ["屁帘", "章仔"];
    const TAG_MEMORY_KEYS = {
      year: "nas_importer_year_tags",
      category: "nas_importer_category_tags",
      subcategory: "nas_importer_subcategory_tags",
    };

    function splitInput(value) {
      return String(value || "").split(/[、,，\s]+/).map(x => x.trim()).filter(Boolean);
    }
    function rememberPaths() {
      const source = $("sourceDir").value.trim();
      const target = $("targetDir").value.trim();
      if (source) localStorage.setItem(PATH_MEMORY_KEYS.source, source);
      if (target) localStorage.setItem(PATH_MEMORY_KEYS.target, target);
    }
    function uniqList(values) {
      const out = [];
      const seen = new Set();
      (values || []).forEach((value) => {
        const clean = String(value || "").trim();
        if (!clean || seen.has(clean)) return;
        seen.add(clean);
        out.push(clean);
      });
      return out;
    }
    function readTagMemory(kind) {
      try {
        const raw = JSON.parse(localStorage.getItem(TAG_MEMORY_KEYS[kind]) || "[]");
        return Array.isArray(raw) ? raw.map(String) : [];
      } catch (_) {
        return [];
      }
    }
    function writeTagMemory(kind, values) {
      const list = uniqList(values);
      localStorage.setItem(TAG_MEMORY_KEYS[kind], JSON.stringify(list));
      return list;
    }
    function rememberTagValues(kind, values) {
      const current = readTagMemory(kind);
      return writeTagMemory(kind, current.concat(values || []));
    }
    function setupQuickPicks(boxId, inputId, values, kind) {
      const box = $(boxId);
      if (!box) return;
      const merged = uniqList((values || []).concat(readTagMemory(kind)));
      box.innerHTML = merged.map(value => `<button type="button" class="quick-pick" data-value="${escapeHtml(value)}">${escapeHtml(value)}</button>`).join("");
      box.querySelectorAll("button").forEach((button) => {
        button.addEventListener("click", () => {
          const input = $(inputId);
          const value = button.dataset.value || "";
          const parts = splitInput(input.value);
          if (!parts.includes(value)) parts.push(value);
          input.value = parts.join("、");
          rememberTagValues(kind, parts);
        });
      });
    }
    function rememberBatchTags() {
      rememberTagValues("year", splitInput($("batchYear").value));
      rememberTagValues("category", splitInput($("batchCategory").value));
      rememberTagValues("subcategory", splitInput($("batchSubcategory").value));
      setupQuickPicks("batchYearPicks", "batchYear", QUICK_YEARS, "year");
      setupQuickPicks("batchCategoryPicks", "batchCategory", QUICK_CATEGORIES, "category");
      setupQuickPicks("batchSubcategoryPicks", "batchSubcategory", QUICK_SUBCATEGORIES, "subcategory");
    }

    function alertBox(message, imageUrl = "") {
      return new Promise((resolve) => {
        $("modalBox").classList.toggle("image-box", !!imageUrl);
        $("modalImage").style.display = imageUrl ? "block" : "none";
        $("modalImage").src = imageUrl || "";
        $("modalMessage").textContent = String(message || "");
        $("modal").classList.add("open");
        $("modalOk").onclick = () => {
          $("modal").classList.remove("open");
          $("modalImage").removeAttribute("src");
          $("modalOk").onclick = null;
          resolve();
        };
      });
    }
    function setStatus(message, err = false) {
      $("status").textContent = message || "";
      $("status").className = err ? "status err" : "status";
    }
    function setTopProgressStatus(message = "", processed = 0, total = 0) {
      const done = Math.max(0, Number(processed || 0));
      const all = Math.max(0, Number(total || 0));
      const prefix = String(message || "已处理").trim();
      $("progressText").textContent = all > 0 ? `${prefix} ${done}/${all}` : prefix;
      const pct = all > 0 ? Math.max(0, Math.min(100, (done / all) * 100)) : 0;
      $("progressBar").style.width = pct.toFixed(1) + "%";
    }
    function setScanning(active) {
      $("scanBtn").disabled = !!active;
      $("scanBtn").textContent = active ? "扫描中..." : "扫描识别";
      $("cancelScanBtn").disabled = !active;
      $("sourceDir").disabled = !!active;
      $("targetDir").disabled = !!active;
    }
    async function api(path, options = {}) {
      const resp = await fetch(path, options);
      if (!resp.ok) {
        let message = await resp.text();
        try { message = JSON.parse(message).detail || message; } catch (_) {}
        throw new Error(message);
      }
      return resp.json();
    }
    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
    }
    function cleanCode(value) {
      return String(value || "").trim().toUpperCase().replace(/#$/, "").replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "");
    }
    function updateFilename(row) {
      const codeInput = row.querySelector('[data-role="style"]');
      const filenameInput = row.querySelector('[data-role="filename"]');
      const code = cleanCode(codeInput.value);
      if (code) filenameInput.value = code + "_000" + (row.dataset.suffix || ".jpg");
    }
    function rowPayload(row) {
      return {
        selected: row.querySelector('[data-role="selected"]').checked,
        source_rel_path: row.dataset.relPath || "",
        style_code: row.querySelector('[data-role="style"]').value.trim(),
        year_tag: $("batchYear").value.trim(),
        category: $("batchCategory").value.trim(),
        subcategory: $("batchSubcategory").value.trim(),
        target_filename: row.querySelector('[data-role="filename"]').value.trim(),
      };
    }
    function previewSource(jobId, relPath, name) {
      const qs = new URLSearchParams({ job_id: jobId, source_rel_path: relPath || "", max_edge: "900" });
      return alertBox(name || "图片预览", "/api/source-image?" + qs.toString());
    }
    function renderJob(job) {
      currentJob = job;
      $("countText").textContent = job.total ? `${job.processed}/${job.total}` : "";
      const status = String(job.status || "");
      const label = status === "canceled" ? "已停止扫描" : (status === "failed" ? "扫描失败" : (status === "completed" ? "预处理完成" : "已处理"));
      setTopProgressStatus(label, job.processed || 0, job.total || 0);
      setStatus("");
      if (!job.items || !job.items.length) {
        $("rows").innerHTML = '<tr><td colspan="5" class="muted">暂无待确认图片。</td></tr>';
        return;
      }
      $("rows").innerHTML = job.items.map((item, index) => {
        const statusClass = item.status === "ok" ? "" : (item.status === "ocr_failed" || item.status === "invalid_style_code" ? "err" : "warn");
        const pillClass = item.status === "ok" ? "pill" : (statusClass === "err" ? "pill err" : "pill warn");
        const statusText = item.error || (item.status === "ok" ? "识别成功" : item.status);
        return `
          <tr class="${statusClass}" data-rel-path="${escapeHtml(item.source_rel_path || "")}" data-suffix="${escapeHtml(item.suffix || ".jpg")}">
            <td><input data-role="selected" type="checkbox" checked></td>
            <td><div class="source"><button type="button" data-role="previewSource">${escapeHtml(item.source_name || "")}</button></div><div class="muted">${escapeHtml(item.source_rel_path || "")}</div></td>
            <td><input data-role="style" type="text" value="${escapeHtml(item.proposed_style_code || "")}"></td>
            <td><input data-role="filename" type="text" value="${escapeHtml(item.proposed_filename || "")}"></td>
            <td><span class="${pillClass}">${escapeHtml(statusText)}</span></td>
          </tr>`;
      }).join("");
      $("rows").querySelectorAll('[data-role="style"]').forEach((input) => {
        input.addEventListener("input", () => updateFilename(input.closest("tr")));
      });
      $("rows").querySelectorAll('[data-role="previewSource"]').forEach((button) => {
        button.addEventListener("click", () => {
          const row = button.closest("tr");
          previewSource(job.job_id, row.dataset.relPath || "", button.textContent || "图片预览");
        });
      });
    }
    async function startScan() {
      const source_dir = $("sourceDir").value.trim();
      const target_dir = $("targetDir").value.trim();
      if (!source_dir || !target_dir) return alertBox("请填写源目录和目标目录");
      rememberPaths();
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      currentJob = null;
      setTopProgressStatus("正在创建扫描任务", 0, 0);
      setScanning(true);
      setStatus("正在创建扫描任务...");
      try {
        const job = await api("/api/jobs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source_dir, target_dir }),
        });
        renderJob(job);
        pollTimer = setInterval(pollJob, 1200);
      } catch (err) {
        setStatus(err.message || "扫描失败", true);
        await alertBox("扫描失败：" + (err.message || "未知错误"));
      } finally {
        if (!currentJob || ["completed", "failed", "canceled"].includes(currentJob.status)) setScanning(false);
      }
    }
    async function pollJob() {
      if (!currentJob) return;
      try {
        const job = await api("/api/jobs/" + encodeURIComponent(currentJob.job_id));
        renderJob(job);
        if (["completed", "failed", "canceled"].includes(job.status)) {
          clearInterval(pollTimer);
          pollTimer = null;
          setScanning(false);
          if (job.status === "failed") await alertBox("扫描失败：" + (job.message || ""));
        }
      } catch (err) {
        clearInterval(pollTimer);
        pollTimer = null;
        setScanning(false);
        setStatus(err.message || "读取任务失败", true);
      }
    }
    async function cancelScan() {
      if (!currentJob || !["pending", "running"].includes(currentJob.status)) return;
      $("cancelScanBtn").disabled = true;
      setTopProgressStatus("正在停止扫描", currentJob.processed || 0, currentJob.total || 0);
      setStatus("正在停止扫描...");
      try {
        const job = await api("/api/jobs/" + encodeURIComponent(currentJob.job_id) + "/cancel", { method: "POST" });
        renderJob(job);
        if (["completed", "failed", "canceled"].includes(job.status)) {
          if (pollTimer) clearInterval(pollTimer);
          pollTimer = null;
          setScanning(false);
        }
      } catch (err) {
        setStatus(err.message || "停止扫描失败", true);
        $("cancelScanBtn").disabled = false;
      }
    }
    async function commitJob() {
      if (!currentJob || currentJob.status !== "completed") return alertBox("请先等待扫描完成");
      rememberBatchTags();
      const items = Array.from($("rows").querySelectorAll("tr[data-rel-path]")).map(rowPayload);
      if (!items.some((item) => item.selected)) return alertBox("请至少选择一张要导入的图片");
      $("commitBtn").disabled = true;
      setStatus("正在复制到目标目录...");
      try {
        const result = await api("/api/jobs/" + encodeURIComponent(currentJob.job_id) + "/commit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ items }),
        });
        const failed = result.failed || [];
        const message = `导入完成：成功 ${result.imported || 0} 张，失败 ${failed.length} 张` +
          (failed.length ? "\\n\\n" + failed.slice(0, 20).map(x => `${x.source_rel_path || ""}：${x.error || "失败"}`).join("\\n") : "");
        setStatus(message, failed.length > 0);
        await alertBox(message);
      } catch (err) {
        setStatus(err.message || "导入失败", true);
        await alertBox("导入失败：" + (err.message || "未知错误"));
      } finally {
        $("commitBtn").disabled = false;
      }
    }
    $("scanBtn").addEventListener("click", startScan);
    $("cancelScanBtn").addEventListener("click", cancelScan);
    $("commitBtn").addEventListener("click", commitJob);
    $("toggleBtn").addEventListener("click", () => {
      const boxes = Array.from(document.querySelectorAll('[data-role="selected"]'));
      const should = boxes.some(box => !box.checked);
      boxes.forEach(box => { box.checked = should; });
    });
    ["batchYear", "batchCategory", "batchSubcategory"].forEach((id) => {
      $(id).addEventListener("blur", rememberBatchTags);
    });
    ["sourceDir", "targetDir"].forEach((id) => {
      $(id).addEventListener("blur", rememberPaths);
    });
    setupQuickPicks("batchYearPicks", "batchYear", QUICK_YEARS, "year");
    setupQuickPicks("batchCategoryPicks", "batchCategory", QUICK_CATEGORIES, "category");
    setupQuickPicks("batchSubcategoryPicks", "batchSubcategory", QUICK_SUBCATEGORIES, "subcategory");
  </script>
</body>
</html>
"""


class ImportHandler(BaseHTTPRequestHandler):
    server: "ImportServer"

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, content: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: int, payload: Any) -> None:
        self._send(status, _json_bytes(payload))

    def _error(self, status: int, detail: str) -> None:
        self._send_json(status, {"detail": detail})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                html = HTML.replace("__DEFAULT_SOURCE__", self.server.default_source.replace("\\", "\\\\")).replace(
                    "__DEFAULT_TARGET__", self.server.default_target.replace("\\", "\\\\")
                )
                self._send(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path.startswith("/api/jobs/"):
                job_id = unquote(parsed.path.rsplit("/", 1)[-1])
                job = self.server.jobs.get(job_id)
                if not job:
                    self._error(HTTPStatus.NOT_FOUND, "任务不存在")
                    return
                self._send_json(HTTPStatus.OK, self.server.jobs.snapshot(job))
                return
            if parsed.path == "/api/source-image":
                self._serve_source_image(parsed)
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            logging.exception("GET failed")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/jobs":
                payload = _read_json(self)
                source_dir = Path(str(payload.get("source_dir") or "").strip())
                target_dir = Path(str(payload.get("target_dir") or "").strip())
                if not source_dir.exists() or not source_dir.is_dir():
                    self._error(HTTPStatus.BAD_REQUEST, "源目录不存在")
                    return
                target_dir.mkdir(parents=True, exist_ok=True)
                job = self.server.jobs.create(source_dir, target_dir)
                threading.Thread(target=self.server.prepare_job, args=(job.job_id,), daemon=True).start()
                self._send_json(HTTPStatus.OK, self.server.jobs.snapshot(job))
                return
            if parsed.path.endswith("/cancel") and parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.split("/")[3]
                job = self.server.jobs.get(job_id)
                if not job:
                    self._error(HTTPStatus.NOT_FOUND, "任务不存在")
                    return
                if job.status in {"completed", "failed", "canceled"}:
                    self._send_json(HTTPStatus.OK, self.server.jobs.snapshot(job))
                    return
                self.server.jobs.update(job_id, cancel_requested=True, message="正在停止扫描...")
                self._send_json(HTTPStatus.OK, self.server.jobs.snapshot(job))
                return
            if parsed.path.endswith("/commit") and parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.split("/")[3]
                payload = _read_json(self)
                result = self.server.commit_job(job_id, payload.get("items") or [])
                self._send_json(HTTPStatus.OK, result)
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            logging.exception("POST failed")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _serve_source_image(self, parsed: Any) -> None:
        qs = parse_qs(parsed.query)
        job_id = str((qs.get("job_id") or [""])[0])
        rel = str((qs.get("source_rel_path") or [""])[0])
        max_edge = max(128, min(2048, int((qs.get("max_edge") or ["360"])[0] or "360")))
        job = self.server.jobs.get(job_id)
        if not job:
            self._error(HTTPStatus.NOT_FOUND, "任务不存在")
            return
        fp = (job.source_dir / rel).resolve()
        try:
            fp.relative_to(job.source_dir.resolve())
        except Exception:
            self._error(HTTPStatus.BAD_REQUEST, "图片路径无效")
            return
        if not fp.exists() or not fp.is_file():
            self._error(HTTPStatus.NOT_FOUND, "图片不存在")
            return
        with Image.open(fp) as im0:
            im = im0.convert("RGB")
            im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            import io

            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=72)
        self._send(HTTPStatus.OK, buf.getvalue(), "image/jpeg")


class ImportServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        default_source: str,
        default_target: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.jobs = JobStore()
        self.default_source = default_source
        self.default_target = default_target
        self.tesseract_bin = shutil.which("tesseract")

    def prepare_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        try:
            files = _scan_images(job.source_dir)
            used_names, next_seq = _build_name_allocator(job.target_dir)
            self.jobs.update(job_id, status="running", total=len(files), message=f"发现 {len(files)} 张图片")
            items: list[dict[str, Any]] = []
            for index, path in enumerate(files, start=1):
                latest = self.jobs.get(job_id)
                if not latest or latest.cancel_requested:
                    self.jobs.update(job_id, status="canceled", message=f"已停止扫描，已处理 {index - 1}/{len(files)}")
                    return
                code = ""
                error = ""
                try:
                    code = _extract_style(path, self.tesseract_bin)
                except Exception as exc:
                    logging.warning("OCR failed: %s", path, exc_info=True)
                    error = f"OCR 失败：{exc}"
                style_code = _clean_style_code(code)
                valid = _is_valid_style_code(style_code)
                if not code and not error:
                    error = "OCR 未识别到款号"
                elif code and not valid:
                    error = "识别款号必须以字母开头"
                prefix = _code_to_filename_prefix(code) if code else SAFE_STEM_RE.sub("_", path.stem).strip("_") or "UNKNOWN"
                filename = _next_target_name(prefix, path.suffix.lower(), used_names, next_seq)
                rel = str(path.relative_to(job.source_dir)).replace("\\", "/")
                latest = self.jobs.get(job_id)
                if not latest or latest.cancel_requested:
                    self.jobs.update(job_id, status="canceled", message=f"已停止扫描，已处理 {index - 1}/{len(files)}")
                    return
                items.append(
                    {
                        "source_rel_path": rel,
                        "source_name": path.name,
                        "suffix": path.suffix.lower() if path.suffix.lower() in IMAGE_EXTS else ".jpg",
                        "proposed_style_code": style_code,
                        "proposed_filename": filename,
                        "status": "ok" if valid else ("invalid_style_code" if code else "ocr_failed"),
                        "error": error,
                    }
                )
                self.jobs.update(job_id, processed=index, items=list(items), message=f"已处理 {index}/{len(files)}")
            self.jobs.update(job_id, status="completed", message=f"预处理完成，共 {len(files)} 张", items=items)
        except Exception as exc:
            logging.exception("prepare job failed")
            self.jobs.update(job_id, status="failed", message=str(exc))

    def commit_job(self, job_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            raise RuntimeError("任务不存在")
        if job.status != "completed":
            raise RuntimeError("任务尚未完成识别")
        if job.committed:
            raise RuntimeError("该批次已导入")
        prepared = {str(item.get("source_rel_path")): item for item in job.items}
        used_names, next_seq = _build_name_allocator(job.target_dir)
        imported = 0
        failed: list[dict[str, str]] = []
        manifest_rows: list[dict[str, Any]] = []
        for item in items:
            if not item.get("selected"):
                continue
            rel = str(item.get("source_rel_path") or "")
            source = prepared.get(rel)
            if not source:
                failed.append({"source_rel_path": rel, "error": "源记录不存在"})
                continue
            src = (job.source_dir / rel).resolve()
            try:
                src.relative_to(job.source_dir.resolve())
            except Exception:
                failed.append({"source_rel_path": rel, "error": "源路径无效"})
                continue
            if not src.exists() or not src.is_file():
                failed.append({"source_rel_path": rel, "error": "源文件不存在"})
                continue
            style_code = _clean_style_code(str(item.get("style_code") or ""))
            raw_filename = str(item.get("target_filename") or "").strip()
            if not raw_filename and style_code:
                raw_filename = f"{style_code}_000{src.suffix.lower()}"
            if not _is_valid_style_code(style_code):
                failed.append({"source_rel_path": rel, "error": "款号必须以字母开头"})
                continue
            try:
                first = _sanitize_filename(raw_filename, src.suffix.lower())
                if first.lower() in used_names:
                    stem = re.sub(r"_\d+$", "", Path(first).stem).strip() or style_code
                    target_name = _next_target_name(stem, Path(first).suffix, used_names, next_seq)
                else:
                    target_name = first
                    used_names.add(first.lower())
                shutil.copy2(src, job.target_dir / target_name)
                imported += 1
                final_style_code = filename_to_style_code(target_name).strip()
                tags: list[str] = []
                year_tag = str(item.get("year_tag") or "").strip()
                if year_tag:
                    tags.append(f"year:{year_tag}")
                for category in _split_tags(item.get("category")):
                    tags.append(f"category:{category}")
                for subcategory in _split_tags(item.get("subcategory")):
                    tags.append(f"subcategory:{subcategory}")
                manifest_rows.append(
                    {
                        "source_rel_path": rel,
                        "image_name": target_name,
                        "style_code": final_style_code,
                        "tags": tags,
                        "imported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
            except Exception as exc:
                failed.append({"source_rel_path": rel, "error": str(exc)})
        if manifest_rows:
            manifest_path = job.target_dir / "_nas_import_manifest.jsonl"
            with manifest_path.open("a", encoding="utf-8") as fh:
                for row in manifest_rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        result = {"imported": imported, "failed": failed, "target_dir": str(job.target_dir)}
        self.jobs.update(job_id, committed=True, result=result, message=f"导入完成：成功 {imported} 张，失败 {len(failed)} 张")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Windows NAS product image batch importer")
    parser.add_argument("--source", default="", help=r"源目录，例如 Z:\2018\成衣")
    parser.add_argument("--target", default=r"Z:\products\standard_samples", help=r"目标目录，默认 Z:\products\standard_samples")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    server = ImportServer(
        (args.host, args.port),
        ImportHandler,
        default_source=args.source,
        default_target=args.target,
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"NAS 款图批量入库工具已启动：{url}")
    print("按 Ctrl+C 退出。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
