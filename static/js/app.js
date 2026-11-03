/* Citation-first clause RAG UI on chat-pdf dual-pane shell.
 * Product: ask → curated citations → left PDF + evidence strip.
 * P1: stream end-only PDF/citations; refuse card styles.
 * P2: PDF.js highlight, chip docs, chunk debug.
 * P3: multi-turn left-pane stickiness.
 * Untrusted answer/source text always goes through escapeHtml before innerHTML.
 */
(function () {
  const pdfSelect = document.getElementById("pdfSelect");
  const pdfEmpty = document.getElementById("pdfEmpty");
  const pdfFrame = document.getElementById("pdfFrame");
  const pdfJsView = document.getElementById("pdfJsView");
  const pdfCanvas = document.getElementById("pdfCanvas");
  const pdfTextLayer = document.getElementById("pdfTextLayer");
  const pdfHighlightLayer = document.getElementById("pdfHighlightLayer");
  const pdfTitle = document.getElementById("pdfTitle");
  const pdfSubtitle = document.getElementById("pdfSubtitle");
  const pageBadge = document.getElementById("pageBadge");
  const evidenceStrip = document.getElementById("evidenceStrip");
  const evidenceCite = document.getElementById("evidenceCite");
  const evidenceExcerpt = document.getElementById("evidenceExcerpt");
  const corpusMeta = document.getElementById("corpusMeta");
  const embedMeta = document.getElementById("embedMeta");
  const messagesEl = document.getElementById("messages");
  const emptyChat = document.getElementById("emptyChat");
  const form = document.getElementById("askForm");
  const input = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const resetBtn = document.getElementById("resetBtn");
  const stopBtn = document.getElementById("stopBtn");
  const debugToggle = document.getElementById("debugToggle");
  const streamToggle = document.getElementById("streamToggle");
  const genPill = document.getElementById("genPill");
  const errPill = document.getElementById("errPill");
  const healthText = document.getElementById("healthText");
  const panelGroup = document.getElementById("panelGroup");
  const pdfPanel = document.getElementById("pdfPanel");
  const aiPanel = document.getElementById("aiPanel");
  const resizeHandle = document.getElementById("resizeHandle");

  let pdfs = [];
  let documents = [];
  let selectedIds = [];
  let selectionLoaded = false;
  let scopeEpoch = 0;
  let scopeSync = Promise.resolve();
  let libraryRevision = 0;
  let libraryRenderSignature = null;
  let currentRequest = null;
  const deletedIds = new Set();
  const documentList = document.getElementById("documentList");
  const uploadFiles = document.getElementById("uploadFiles");
  const queryMode = document.getElementById("queryMode");
  const answerLanguage = document.getElementById("answerLanguage");
  const scopeStatus = document.getElementById("scopeStatus");
  const modeStatus = document.getElementById("modeStatus");

  let activeSources = [];
  let lastStickyView = null;
  let sessionId = (function () {
    try {
      const k = "insur_session_id";
      let v = localStorage.getItem(k);
      if (!v) {
        v = "s_" + Math.random().toString(36).slice(2) + Date.now().toString(36);
        localStorage.setItem(k, v);
      }
      return v;
    } catch (e) {
      return "s_" + String(Date.now());
    }
  })();
 // P3: keep last citation across follow-ups
  let currentView = { name: null, url: null, page: null, index: null, excerpt: "" };
  let isLoading = false;
  let abortController = null;
  let citeSeq = 0;
  let debugMode = false;
  let lastEmbeddingProvider = null;
  let pdfOpenSequence = 0;
  let pdfRenderTask = null;
  let pdfDocCache = { url: null, doc: null, renderToken: 0 };

  const DISPLAY_FALLBACK = {
    "sample_term_life.pdf": "示例终身寿险条款",
    "sample_critical_illness.pdf": "示例重大疾病保险条款",
  };

  // Configure PDF.js worker if available
  try {
    if (window.pdfjsLib) {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc =
        "/static/vendor/pdfjs/pdf.worker.min.js";
    }
  } catch (_e) {}

  function escapeHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function basename(name) {
    const s = String(name || "");
    const parts = s.replace(/\\/g, "/").split("/");
    return parts[parts.length - 1] || s;
  }

  function displayNameFor(docName, extra) {
    extra = extra || {};
    if (extra.display_name) return extra.display_name;
    if (extra.metadata && extra.metadata.display_name) {
      return extra.metadata.display_name;
    }
    const bare = basename(docName);
    if (DISPLAY_FALLBACK[bare]) return DISPLAY_FALLBACK[bare];
    const hit = resolvePdfByName(docName);
    if (hit && hit.display_name) return hit.display_name;
    return bare || "未知文档";
  }

  function cleanExcerpt(text) {
    let t = String(text || "")
      .replace(/\s+/g, " ")
      .trim();
    t = t
      .replace(
        /^(文档名称|产品名称|文档类型|生效日期|状态)[：:][^。；;\n]{0,40}[。；;\s]*/g,
        ""
      )
      .trim();
    if (t.length > 180) t = t.slice(0, 180) + "…";
    return t;
  }

  function resolvePdfByName(docName) {
    if (!docName) return null;
    const bare = basename(docName).toLowerCase();
    return (
      pdfs.find(function (p) {
        return basename(p.name).toLowerCase() === bare;
      }) ||
      pdfs.find(function (p) {
        return (
          basename(p.name).toLowerCase().includes(bare) ||
          bare.includes(basename(p.name).toLowerCase())
        );
      }) ||
      null
    );
  }

  function setEvidence(source, index) {
    if (!evidenceStrip) return;
    if (!source) {
      evidenceStrip.hidden = true;
      if (evidenceCite) evidenceCite.textContent = "";
      if (evidenceExcerpt) evidenceExcerpt.textContent = "";
      return;
    }
    const name = displayNameFor(
      source.document_name || source.source || source.filename,
      source
    );
    const page = source.page_number ?? source.page;
    const excerpt = cleanExcerpt(
      source.content || source.excerpt || source.text || ""
    );
    evidenceStrip.hidden = false;
    evidenceCite.textContent =
      "[" +
      (index || "?") +
      "] " +
      name + " · #" + source.document_id +
      (page != null && page !== "" ? " · p." + page : "");
    evidenceExcerpt.textContent = excerpt || "（无摘录）";
  }

  function hidePdfViews() {
    if (pdfFrame) {
      pdfFrame.hidden = true;
      pdfFrame.removeAttribute("src");
    }
    if (pdfJsView) pdfJsView.hidden = true;
    if (pdfEmpty) pdfEmpty.hidden = false;
  }

  function pickHighlightNeedle(excerpt) {
    const t = String(excerpt || "")
      .replace(/\s+/g, " ")
      .trim();
    if (!t) return "";
    // Prefer a mid-length continuous clause fragment
    const parts = t.split(/[。；;！!？?\n]/).map(function (s) {
      return s.trim();
    });
    let best = "";
    parts.forEach(function (p) {
      if (p.length >= 8 && p.length <= 48 && p.length > best.length) best = p;
    });
    if (best) return best;
    return t.slice(0, Math.min(36, t.length));
  }

  async function renderPdfJs(url, pageNumber, excerpt) {
    if (!window.pdfjsLib || !pdfCanvas || !pdfJsView) {
      return false;
    }
    const page =
      pageNumber != null && pageNumber !== "" && !Number.isNaN(Number(pageNumber))
        ? Math.max(1, Number(pageNumber))
        : 1;
    const token = ++pdfDocCache.renderToken;
    try {
      if (pdfRenderTask) {
        pdfRenderTask.cancel();
        await pdfRenderTask.promise.catch(() => {});
      }
      if (pdfDocCache.url !== url || !pdfDocCache.doc) {
        const loadingTask = window.pdfjsLib.getDocument({ url: url });
        const doc = await loadingTask.promise;
        if (token !== pdfDocCache.renderToken) { await doc.destroy(); return false; }
        if (pdfDocCache.doc) await pdfDocCache.doc.destroy();
        if (token !== pdfDocCache.renderToken) { await doc.destroy(); return false; }
        pdfDocCache = { url: url, doc: doc, renderToken: token };
      }
      const doc = pdfDocCache.doc;
      const pdfPage = await doc.getPage(Math.min(page, doc.numPages || page));
      if (token !== pdfDocCache.renderToken) return false;

      const wrap = document.getElementById("pdfViewerWrap");
      const wrapW = (wrap && wrap.clientWidth) || 640;
      const unscaled = pdfPage.getViewport({ scale: 1 });
      const scale = Math.min(2.2, Math.max(1.0, (wrapW - 8) / unscaled.width));
      const viewport = pdfPage.getViewport({ scale: scale });

      const canvas = pdfCanvas;
      const ctx = canvas.getContext("2d");
      const outputScale = window.devicePixelRatio || 1;
      canvas.width = Math.floor(viewport.width * outputScale);
      canvas.height = Math.floor(viewport.height * outputScale);
      canvas.style.width = Math.floor(viewport.width) + "px";
      canvas.style.height = Math.floor(viewport.height) + "px";
      const transform =
        outputScale !== 1 ? [outputScale, 0, 0, outputScale, 0, 0] : null;

      if (pdfTextLayer) {
        pdfTextLayer.innerHTML = "";
        pdfTextLayer.style.width = canvas.style.width;
        pdfTextLayer.style.height = canvas.style.height;
      }
      if (pdfHighlightLayer) {
        pdfHighlightLayer.innerHTML = "";
        pdfHighlightLayer.style.width = canvas.style.width;
        pdfHighlightLayer.style.height = canvas.style.height;
      }

      const renderTask = pdfPage.render({canvasContext:ctx, viewport:viewport, transform:transform});
      pdfRenderTask = renderTask;
      await renderTask.promise;
      if (pdfRenderTask === renderTask) pdfRenderTask = null;
      if (token !== pdfDocCache.renderToken) return false;

      // Text layer for selection + highlight targeting
      const textContent = await pdfPage.getTextContent();
      if (token !== pdfDocCache.renderToken) return false;
      if (pdfTextLayer && window.pdfjsLib.TextLayer) {
        // Prefer modern TextLayer API if present
        try {
          const textLayer = new window.pdfjsLib.TextLayer({
            textContentSource: textContent,
            container: pdfTextLayer,
            viewport: viewport,
          });
          await textLayer.render();
        } catch (_tl) {
          // Fallback: manual span placement
          textContent.items.forEach(function (item) {
            if (!item.str) return;
            const tx = window.pdfjsLib.Util.transform(
              viewport.transform,
              item.transform
            );
            const span = document.createElement("span");
            span.textContent = item.str;
            span.style.left = tx[4] + "px";
            span.style.top = tx[5] - item.height * scale + "px";
            span.style.fontSize = Math.max(6, item.height * scale) + "px";
            span.style.position = "absolute";
            span.style.whiteSpace = "pre";
            span.style.color = "transparent";
            span.style.transformOrigin = "0% 0%";
            pdfTextLayer.appendChild(span);
          });
        }
      } else if (pdfTextLayer) {
        textContent.items.forEach(function (item) {
          if (!item.str) return;
          const tx = window.pdfjsLib.Util.transform(
            viewport.transform,
            item.transform
          );
          const span = document.createElement("span");
          span.textContent = item.str;
          span.style.left = tx[4] + "px";
          span.style.top = tx[5] - (item.height || 10) * scale + "px";
          span.style.fontSize =
            Math.max(6, (item.height || 10) * scale) + "px";
          span.style.position = "absolute";
          span.style.whiteSpace = "pre";
          span.style.color = "transparent";
          pdfTextLayer.appendChild(span);
        });
      }

      // Highlight excerpt spans if found
      const needle = pickHighlightNeedle(excerpt);
      if (needle && pdfTextLayer && pdfHighlightLayer) {
        const spans = Array.from(pdfTextLayer.querySelectorAll("span"));
        const joined = spans
          .map(function (s) {
            return s.textContent || "";
          })
          .join("");
        const idx = joined.indexOf(needle);
        // Also try shorter needle
        let useNeedle = needle;
        let useIdx = idx;
        if (useIdx < 0 && needle.length > 12) {
          useNeedle = needle.slice(0, 12);
          useIdx = joined.indexOf(useNeedle);
        }
        if (useIdx >= 0) {
          let cursor = 0;
          spans.forEach(function (span) {
            const t = span.textContent || "";
            const start = cursor;
            const end = cursor + t.length;
            cursor = end;
            if (end <= useIdx || start >= useIdx + useNeedle.length) return;
            const rect = span.getBoundingClientRect();
            const parent = pdfJsView.getBoundingClientRect();
            const mark = document.createElement("div");
            mark.className = "pdf-highlight-mark";
            mark.style.left = rect.left - parent.left + pdfJsView.scrollLeft + "px";
            mark.style.top = rect.top - parent.top + pdfJsView.scrollTop + "px";
            mark.style.width = Math.max(4, rect.width) + "px";
            mark.style.height = Math.max(10, rect.height) + "px";
            pdfHighlightLayer.appendChild(mark);
            span.classList.add("is-hit");
          });
        } else {
          // Portfolio-honest fallback: soft page banner when text not located
          const banner = document.createElement("div");
          banner.className = "pdf-highlight-fallback";
          banner.textContent = "摘录定位：本页 · 未精确匹配文本层（仍显示页码证据）";
          pdfHighlightLayer.appendChild(banner);
          if (typeof showToast === "function") {
            showToast("无法高亮摘录，已打开原页");
          }
        }
      }

      pdfEmpty.hidden = true;
      pdfFrame.hidden = true;
      pdfJsView.hidden = false;
      return true;
    } catch (err) {
      if (token !== pdfDocCache.renderToken) return false;
      console.warn("PDF.js render failed, falling back to iframe", err);
      if (typeof showToast === "function") {
        showToast("无法高亮摘录，已打开原页");
      }
      return false;
    }
  }

  async function openCitedPdf(docName, pageNumber, source, index, opts) {
    opts = opts || {};
    const viewSequence = ++pdfOpenSequence;
    const hit = pdfs.find(p => Number(p.document_id) === Number(source && source.document_id));
    if (!hit || !hit.url || deletedIds.has(Number(hit.document_id)) || (source && source.source_status === "deleted")) {
      if (opts.forceEmpty !== false && !lastStickyView) {
        hidePdfViews();
        pdfSelect.hidden = true;
        pageBadge.hidden = true;
        pdfTitle.textContent = "引用原文";
        pdfSubtitle.textContent = "暂未匹配到可打开的条款 PDF";
        setEvidence(null);
      }
      return false;
    }

    const page =
      pageNumber != null && pageNumber !== "" && !Number.isNaN(Number(pageNumber))
        ? Number(pageNumber)
        : null;
    const excerpt = cleanExcerpt(
      (source && (source.content || source.excerpt || source.text)) || ""
    );

    currentView = {
      document_id: hit.document_id,
      name: hit.name,
      url: hit.url,
      page: page,
      index: index || null,
      excerpt: excerpt,
    };
    lastStickyView = Object.assign({}, currentView, {
      source: source || null,
      display: hit.display_name || displayNameFor(hit.name, source || {}),
    });

    const nice = hit.display_name || displayNameFor(hit.name, source || {});
    pdfTitle.textContent = nice + " · #" + hit.document_id;
    pdfSubtitle.textContent = page
      ? "定位到引用页 p." + page
      : "答案引用的条款原文";
    if (page) {
      pageBadge.hidden = false;
      pageBadge.textContent = "p." + page;
    } else {
      pageBadge.hidden = true;
      pageBadge.textContent = "";
    }

    populatePdfSelect(hit.document_id);

    if (source) setEvidence(source, index);

    const ok = await renderPdfJs(hit.url, page || 1, excerpt);
    if (viewSequence !== pdfOpenSequence || deletedIds.has(Number(hit.document_id)) || Number(currentView.document_id) !== Number(hit.document_id)) return false;
    if (!ok) {
      // iframe fallback
      const hash = page && page > 0 ? "#page=" + page : "";
      const nextSrc = hit.url + hash;
      pdfEmpty.hidden = true;
      if (pdfJsView) pdfJsView.hidden = true;
      pdfFrame.hidden = false;
      if (pdfFrame.getAttribute("src") !== nextSrc) {
        pdfFrame.src = nextSrc;
      }
    }
    return true;
  }

  function populatePdfSelect(documentId) {
    pdfSelect.hidden = !pdfs.length || documentId == null;
    pdfSelect.innerHTML = pdfs.map(pdf => '<option value="' + pdf.document_id + '">' +
      escapeHtml((pdf.display_name || pdf.name) + " · #" + pdf.document_id) + '</option>').join("");
    if (documentId != null) pdfSelect.value = String(documentId);
  }

  function firstAnswerCiteIndex(answerText) {
    const m = String(answerText || "").match(/(?:\[|【)(\d+)(?:\]|】)/);
    if (!m) return 1;
    const n = parseInt(m[1], 10);
    return n > 0 ? n : 1;
  }

  function followAnswerCitation(sources, answerText, opts) {
    opts = opts || {};
    activeSources = Array.isArray(sources) ? sources : [];
    if (!activeSources.length) {
      // P3 stickiness: keep last citation unless forced clear
      if (lastStickyView && !opts.forceClear) {
        pdfTitle.textContent = lastStickyView.display || "引用原文";
        pdfSubtitle.textContent = "沿用上一轮引用（本轮无新引用）";
        return;
      }
      pdfTitle.textContent = "引用原文";
      pdfSubtitle.textContent = "本次回答没有可用引用";
      setEvidence(null);
      return;
    }
    let idx = firstAnswerCiteIndex(answerText);
    if (idx > activeSources.length) idx = 1;

    // P3: if same first cite as current, keep view (avoid flicker)
    const src = activeSources[idx - 1];
    const nextName = basename(src.document_name || src.source || src.filename || "");
    const nextPage = src.page_number ?? src.page;
    if (
      lastStickyView &&
      Number(lastStickyView.document_id) === Number(src.document_id) &&
      Number(lastStickyView.page) === Number(nextPage) &&
      !opts.force
    ) {
      setEvidence(src, idx);
      return;
    }
    openCitedPdf(
      src.document_name || src.source || src.filename,
      src.page_number ?? src.page,
      src,
      idx,
      {}
    );
  }

  function setLoading(v, stage) {
    isLoading = v;
    sendBtn.disabled = v || !selectedIds.length || !String(input.value || "").trim();
    if (genPill) {
      genPill.hidden = !v;
      if (v) setStatus(stage || "generate");
    }
    if (stopBtn) stopBtn.hidden = !v;
    const lastIcon = messagesEl.querySelector(
      ".message.is-assistant:last-child .message-icon"
    );
    if (lastIcon) lastIcon.classList.toggle("spin", v);
  }

  function showError(msg) {
    if (!errPill) return;
    if (!msg) {
      errPill.hidden = true;
      errPill.textContent = "";
      return;
    }
    errPill.hidden = false;
    errPill.textContent = "错误：" + msg;
  }

  function showToast(msg, ms) {
    ms = ms || 3200;
    let el = document.getElementById("uiToast");
    if (!el) {
      el = document.createElement("div");
      el.id = "uiToast";
      el.className = "ui-toast";
      el.hidden = true;
      document.body.appendChild(el);
    }
    if (!msg) {
      el.hidden = true;
      el.textContent = "";
      return;
    }
    el.textContent = msg;
    el.hidden = false;
    clearTimeout(showToast._t);
    showToast._t = setTimeout(function () {
      el.hidden = true;
    }, ms);
  }

  function setStatus(stage) {
    if (!genPill) return;
    const map = {
      retrieve: "检索中",
      generate: "生成中",
      done: "完成",
      idle: "生成中",
    };
    const texts = genPill.querySelectorAll("span");
    if (texts && texts.length) texts[0].textContent = map[stage] || "生成中";
  }


  function errorText(data) {
    const detail = data && (data.detail || data.message || data.error);
    return typeof detail === "string" ? detail : JSON.stringify(detail || data || "请求失败");
  }

  function persistSelection() {
    try { localStorage.setItem("insur_document_ids", JSON.stringify(selectedIds)); } catch (_) {}
  }

  function syncScope() {
    const ids = selectedIds.slice();
    const epoch = scopeEpoch;
    const sid = sessionId;
    scopeSync = scopeSync.catch(() => {}).then(async () => {
      const res = await fetch("/api/v1/queries/session", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({session_id: sid, document_ids: ids})
      });
      const data = await res.json();
      if (!res.ok) throw new Error(errorText(data));
      return data;
    }).catch(err => {
      if (epoch === scopeEpoch && sid === sessionId) {
        showError("范围准备失败，请刷新文档库重试：" + err.message);
      }
      return {error:err};
    });
  }

  function changeSelection(ids) {
    const next = [...new Set(ids.map(Number))].filter(id => documents.some(d => d.document_id === id && d.ready && !deletedIds.has(id))).sort((a,b) => a-b);
    if (JSON.stringify(next) !== JSON.stringify(selectedIds)) {
      selectedIds = next;
      libraryRenderSignature = null;
      scopeEpoch += 1;
      syncScope();
    }
    persistSelection();
    updateScopeLabel();
  }

  function updateScopeLabel() {
    scopeStatus.textContent = selectedIds.length ? "本次范围：" + selectedIds.map(id => "#" + id).join("、") + "（" + selectedIds.length + " 份）" : "请先勾选可提问文档；空选择不会查询全库";
    sendBtn.disabled = isLoading || !selectedIds.length || !String(input.value || "").trim();
  }

  function clearPdfSource() {
    pdfOpenSequence += 1;
    if (pdfRenderTask) { pdfRenderTask.cancel(); pdfRenderTask = null; }
    pdfDocCache.renderToken += 1;
    if (pdfDocCache.doc) pdfDocCache.doc.destroy().catch(() => {});
    pdfDocCache.doc = null;
    pdfDocCache.url = null;
    if (pdfCanvas) pdfCanvas.getContext("2d").clearRect(0, 0, pdfCanvas.width, pdfCanvas.height);
    pdfTextLayer.innerHTML = "";
    pdfHighlightLayer.innerHTML = "";
    currentView = {name:null, url:null, page:null, index:null, excerpt:""};
    lastStickyView = null;
    hidePdfViews();
    setEvidence(null);
    pdfSelect.hidden = true;
    pageBadge.hidden = true;
    pdfTitle.textContent = "引用原文";
    pdfSubtitle.textContent = "来源已删除或不可用";
  }

  function invalidateSources(ids) {
    ids.forEach(id => deletedIds.add(Number(id)));
    if (currentRequest && currentRequest.document_ids.some(id => deletedIds.has(id))) {
      currentRequest.cancelReason = "来源删除，回答已中止；部分输出不是完整答案。";
      currentRequest.controller.abort();
    }
    document.querySelectorAll(".message.is-assistant").forEach(row => {
      if (!(row._sources || []).some(src => deletedIds.has(Number(src.document_id)))) return;
      const sources = row._sources.map(src => deletedIds.has(Number(src.document_id)) ? {
        chunk_id:src.chunk_id, document_id:src.document_id, document_name:src.document_name,
        page_number:src.page_number, source_status:"deleted"
      } : src);
      setAssistantBody(row, row._answerText, sources, {...row._meta, source_status:"deleted"}, {silent:true});
    });
    activeSources = activeSources.map(src => deletedIds.has(Number(src.document_id)) ? {
      document_id:src.document_id, document_name:src.document_name, page_number:src.page_number, source_status:"deleted"
    } : src);
    if (deletedIds.has(Number(currentView.document_id))) clearPdfSource();
  }

  function renderLibrary() {
    const signature = JSON.stringify([documents, selectedIds, [...deletedIds]]);
    if (signature === libraryRenderSignature) return;
    libraryRenderSignature = signature;
    const focused = documentList.contains(document.activeElement) ? document.activeElement : null;
    const focusAttribute = focused && ["data-id", "data-preview", "data-delete"].find(name => focused.hasAttribute(name));
    const focusValue = focusAttribute && focused.getAttribute(focusAttribute);
    const phase = {pending:"待处理",queued:"排队中",parsing:"解析中",indexing:"索引中",ready:"可提问",failed:"处理失败"};
    const deletionPhase = {invalidate:"取消相关工作",indexes:"清理检索索引",original:"清理原文件",history:"清理历史摘录",chunks:"清理正文片段",complete:"完成检查"};
    const failure = {page_limit_exceeded:"超过 200 页上限，整份文件未入库",file_too_large:"超过文件大小限制",not_pdf:"文件不是有效 PDF",damaged_pdf:"PDF 损坏或无法读取",encrypted_pdf:"暂不支持加密 PDF",empty_pdf:"PDF 没有页面",no_readable_text:"未找到可读取文字",unreadable_pages:"含扫描或不可读内容，整份拒收",local_embedding_unavailable:"本地模型未就绪，请准备模型后重传",input_budget_exceeded:"正文片段超出安全处理预算",index_publication_failed:"索引发布失败，请检查服务后重传"};
    const cleanupFailure = {index_readers_or_cleanup_pending:"等待索引读取释放",source_readers_still_releasing:"等待原文读取释放",native_ingestion_still_releasing_source:"等待本地处理释放原文",OSError:"文件清理失败，请重试",PermissionError:"文件被占用或权限不足，请关闭外部阅读器后重试",RuntimeError:"请检查存储配置和文件路径后重试"};
    documentList.innerHTML = documents.length ? documents.map(doc => {
      const removed = doc.deletion_status !== "none" || deletedIds.has(doc.document_id);
      const state = removed ? (doc.deletion_status === "deleted" ? "来源已删除" : "删除中 · " + (deletionPhase[doc.deletion_stage] || "等待清理")) : (phase[doc.processing_status] || doc.processing_status);
      const errors = (failure[doc.error_code] || (doc.error_code || doc.error_message ? "处理失败，请检查文件或服务后重试" : "")) + ((doc.error_pages || []).length ? " · 失败页：" + doc.error_pages.join(", ") : "") + (doc.deletion_error ? " · 清理未完成：" + (cleanupFailure[doc.deletion_error] || "请重试或检查服务记录") : "");
      const retry = doc.deletion_task_status === "failed";
      return '<div class="document-row" data-document-id="' + doc.document_id + '"><label><input type="checkbox" class="document-check" data-id="' + doc.document_id + '"' + (selectedIds.includes(doc.document_id) ? ' checked' : '') + (!doc.ready || removed ? ' disabled' : '') + '> ' + escapeHtml(doc.name) + ' · #' + doc.document_id + '<span class="document-state">' + escapeHtml(state + (doc.page_count ? " · " + doc.page_count + " 页" : "") + (errors ? " · " + errors : "")) + '</span></label><button type="button" class="btn btn-outline btn-xs" data-preview="' + doc.document_id + '"' + (removed || !doc.file_url ? ' disabled' : '') + '>预览</button>' + (doc.deletion_status === "deleted" ? '' : '<button type="button" class="btn btn-outline btn-xs" data-delete="' + doc.document_id + '"' + (removed && !retry ? ' disabled' : '') + '>' + (retry ? "重试删除" : "删除") + '</button>') + '</div>';
    }).join("") : '<p class="library-note">文档库为空。上传文字 PDF 后，等待可提问并手动勾选。</p>';
    if (focusAttribute) {
      const target = documentList.querySelector("[" + focusAttribute + "='" + focusValue + "']");
      if (target && !target.disabled) target.focus();
    }
  }

  async function loadCorpus() {
    const revision = ++libraryRevision;
    try {
      const res = await fetch("/api/v1/documents", {cache:"no-store"});
      if (!res.ok) throw new Error("文档库接口 " + res.status);
      const data = await res.json();
      if (revision !== libraryRevision) return;
      documents = data.documents || [];
      const invalid = documents.filter(d => d.deletion_status !== "none").map(d => d.document_id);
      if (invalid.length) invalidateSources(invalid);
      pdfs = documents.filter(d => d.file_url && d.deletion_status === "none" && !deletedIds.has(d.document_id)).map(d => ({...d, url:d.file_url, display_name:d.name, pages:d.page_count}));
      if (currentView.document_id != null) populatePdfSelect(currentView.document_id);
      if (!selectionLoaded) {
        let saved = [];
        try { saved = JSON.parse(localStorage.getItem("insur_document_ids") || "[]"); } catch (_) {}
        if (!Array.isArray(saved)) saved = [];
        selectionLoaded = true;
        changeSelection(saved);
        if (!selectedIds.length) syncScope();
      } else {
        changeSelection(selectedIds);
      }
      renderLibrary();
      const count = documents.filter(d => d.deletion_status !== "deleted").length;
      corpusMeta.textContent = count + " 份文档";
      document.getElementById("libraryLimits").textContent = "每份最多 " + data.max_pages + " 页、" + Math.round(data.upload_max_bytes / 1024 / 1024) + " MiB；仅支持文字 PDF，扫描/混合扫描整份拒收。";
      document.getElementById("libraryNotice").textContent = data.capacity_warning || "新文档就绪后需手动勾选；引用预览不会改变问答范围。";
      uploadFiles.dataset.maxBytes = data.upload_max_bytes;
    } catch (err) {
      corpusMeta.textContent = "文档库加载失败";
      showError(err.message);
    }
  }

  documentList.addEventListener("change", event => {
    if (!event.target.matches(".document-check")) return;
    const id = Number(event.target.dataset.id);
    changeSelection(event.target.checked ? selectedIds.concat(id) : selectedIds.filter(i => i !== id));
  });
  document.getElementById("clearSelection").addEventListener("click", () => {
    const wasEmpty = selectedIds.length === 0;
    changeSelection([]);
    if (wasEmpty) { scopeEpoch += 1; syncScope(); }
    renderLibrary();
  });
  documentList.addEventListener("click", async event => {
    const preview = event.target.closest("[data-preview]");
    if (preview) {
      const doc = documents.find(d => d.document_id === Number(preview.dataset.preview));
      if (doc) openCitedPdf(doc.name, 1, {document_id:doc.document_id, document_name:doc.name}, 1, {});
    }
    const button = event.target.closest("[data-delete]");
    if (!button || button.disabled) return;
    const doc = documents.find(d => d.document_id === Number(button.dataset.delete));
    if (!doc || !window.confirm("删除 #" + doc.document_id + "「" + doc.name + "」？将清除原文件与引用摘录，保留已有答案正文。")) return;
    button.disabled = true;
    try {
      const res = await fetch("/api/v1/documents/" + doc.document_id, {method:"DELETE"});
      const data = await res.json();
      if (!res.ok) throw new Error(errorText(data));
      invalidateSources([doc.document_id]);
      changeSelection(selectedIds.filter(id => id !== doc.document_id));
      renderLibrary();
      await loadCorpus();
    } catch (err) { button.disabled = false; showError(err.message); }
  });
  uploadFiles.addEventListener("change", async () => {
    const files = Array.from(uploadFiles.files || []);
    const status = document.getElementById("uploadStatus");
    uploadFiles.disabled = true;
    const results = [];
    try {
      for (const file of files) {
        status.textContent = "上传中：" + file.name;
        if (file.size > Number(uploadFiles.dataset.maxBytes || Infinity)) {
          results.push(file.name + "：超过文件大小限制");
          continue;
        }
        const body = new FormData(); body.append("file", file);
        try {
          const res = await fetch("/api/v1/documents", {method:"POST", body});
          const data = await res.json();
          if (!res.ok) throw new Error(errorText(data));
          results.push(file.name + "：" + (res.status === 200 ? "已存在" : "已接收，后台处理中") + " · #" + data.document_id);
        } catch (err) { results.push(file.name + "：" + err.message); }
        await loadCorpus();
      }
    } finally { status.textContent = results.join("\n"); uploadFiles.value = ""; uploadFiles.disabled = false; }
  });
  async function pollLibrary() { await loadCorpus(); setTimeout(pollLibrary, 1800); }

  async function loadHealth() {
    try {
      const res = await fetch("/api/v1/health/");
      if (!res.ok) throw new Error("health " + res.status);
      const data = await res.json();
      const st = (data.status || "ok").toLowerCase();
      healthText.textContent =
        st === "healthy" || st === "ok" ? "服务正常" : st;
    } catch (_e) {
      healthText.textContent = "离线";
    }
  }

  function linkCitations(escapedText) {
    return escapedText.replace(/(?:\[|【)(\d+)(?:\]|】)/g, function (_m, n) {
      return (
        '<button type="button" class="cite-ref" data-cite-index="' +
        n +
        '" title="打开引用 [' +
        n +
        ']">[' +
        n +
        "]</button>"
      );
    });
  }

  function formatAnswerHtml(text) {
    const raw = String(text || "");
    const sectionRe = /(?:【\s*(结论|条款依据|不确定[/／]?边界|边界)\s*】|\[(Conclusion|Clause evidence|Uncertainty\s*\/\s*boundaries)\])\s*([\s\S]*?)(?=【\s*(?:结论|条款依据|不确定[/／]?边界|边界)\s*】|\[(?:Conclusion|Clause evidence|Uncertainty\s*\/\s*boundaries)\]|$)/gi;
    const parts = [];
    let m;
    while ((m = sectionRe.exec(raw)) !== null) parts.push({title:m[1] || m[2], body:(m[3] || "").trim()});
    if (parts.length >= 2) {
      const blocks = parts
        .map(function (p) {
          let kind = "is-evidence";
          let label = p.title;
          if (p.title.indexOf("结论") >= 0 || /conclusion/i.test(p.title)) {
            kind = "is-conclusion";
            label = /conclusion/i.test(p.title) ? "Conclusion" : "结论";
          } else if (p.title.indexOf("依据") >= 0 || /evidence/i.test(p.title)) {
            kind = "is-evidence";
            label = /evidence/i.test(p.title) ? "Clause evidence" : "条款依据";
          } else {
            kind = "is-boundary";
            label = /uncertainty/i.test(p.title) ? "Uncertainty / boundaries" : "不确定 / 边界";
          }
          return (
            '<section class="answer-section ' +
            kind +
            '"><div class="answer-section-label">' +
            escapeHtml(label) +
            '</div><div class="answer-section-body">' +
            linkCitations(escapeHtml(p.body)) +
            "</div></section>"
          );
        })
        .join("");
      return '<div class="answer-sections">' + blocks + "</div>";
    }

    return linkCitations(
      escapeHtml(raw).replace(
        /^(【[^】]+】|[一二三四五六七八九十]+[、.．]|结论|条款依据|不确定[/／]?边界)([^\n]*)/gm,
        function (_mm, head, rest) {
          return '<span class="sec-head">' + head + (rest || "") + "</span>";
        }
      )
    );
  }

  function debugPanelHtml(sources, meta) {
    if (!debugMode) return "";
    meta = meta || {};
    const rows = (sources || [])
      .map(function (s, i) {
        return (
          "<tr><td>" +
          (i + 1) +
          "</td><td>" +
          escapeHtml(String(s.chunk_id != null ? s.chunk_id : "—")) +
          "</td><td>" +
          escapeHtml(
            String(
              s.similarity_score != null
                ? Number(s.similarity_score).toFixed(4)
                : "—"
            )
          ) +
          "</td><td>" +
          escapeHtml(basename(s.document_name || "")) +
          "</td><td>p." +
          escapeHtml(String(s.page_number ?? s.page ?? "—")) +
          "</td></tr>"
        );
      })
      .join("");
    return (
      '<details class="debug-panel"><summary>调试 · chunk_id / 分数' +
      (meta.embedding_provider
        ? " · " + escapeHtml(meta.embedding_provider)
        : lastEmbeddingProvider
          ? " · " + escapeHtml(lastEmbeddingProvider)
          : "") +
      (meta.answer_kind ? " · kind=" + escapeHtml(meta.answer_kind) : "") +
      "</summary><table class=\"debug-table\"><thead><tr><th>#</th><th>chunk_id</th><th>score</th><th>doc</th><th>page</th></tr></thead><tbody>" +
      (rows || "<tr><td colspan=5>无片段</td></tr>") +
      "</tbody></table></details>"
    );
  }

  function citationsHtml(sources, groupId) {
    if (!sources || !sources.length) return "";
    const visible = sources;
    if (!visible.length) return "";
    const cards = visible
      .map(function (s, i) {
        const name = displayNameFor(
          s.document_name || s.source || s.filename || "doc",
          s
        );
        const page = s.page_number ?? s.page ?? "—";
        const removed = s.source_status === "deleted" || deletedIds.has(Number(s.document_id));
        const excerpt = removed ? "来源已删除 · 原文摘录不可用" : cleanExcerpt(s.content || s.excerpt || s.text || "");
        let score = "";
        if (s.similarity_score != null && s.similarity_score !== "") {
          const n = Number(s.similarity_score);
          if (Number.isFinite(n) && n > 0) score = n.toFixed(3);
        }
        return (
          '<button type="button" class="cite-card' + (removed ? ' is-deleted" disabled' : '"') + ' data-cite-index="' +
          (i + 1) +
          '" data-cite-group="' +
          groupId +
          '">' +
          '<div class="cite-card-top">' +
          '<span class="cite-idx">[' +
          (i + 1) +
          "]</span>" +
          '<span class="cite-doc">' +
          escapeHtml(name + " · #" + s.document_id) +
          "</span>" +
          '<span class="cite-page">p.' +
          escapeHtml(String(page)) +
          "</span>" +
          (debugMode && score
            ? '<span class="cite-score">' + escapeHtml(score) + "</span>"
            : "") +
          "</div>" +
          '<div class="cite-excerpt">' +
          escapeHtml(excerpt) +
          "</div></button>"
        );
      })
      .join("");
    return (
      '<div class="citations" data-cite-group="' +
      groupId +
      '"><div class="citations-title">来源 · ' +
      visible.length +
      " · 点击打开左侧原文</div>" +
      cards +
      "</div>"
    );
  }

  function ensureMessagesVisible() {
    emptyChat.hidden = true;
    emptyChat.setAttribute("aria-hidden", "true");
    messagesEl.hidden = false;
  }

  function kindClass(kind) {
    const k = String(kind || "answer").toLowerCase();
    if (k === "refusal" || k === "insufficient_evidence") return "is-refusal";
    if (k === "advice") return "is-advice";
    if (k === "llm_unavailable") return "is-llm-unavailable";
    if (k === "degraded") return "is-degraded";
    return "is-answer";
  }

  function kindBadge(kind) {
    const k = String(kind || "answer").toLowerCase();
    const map = {
      refusal: "拒答 / 无充分依据",
      insufficient_evidence: "拒答 / 无充分依据",
      advice: "边界 / 不构成建议",
      llm_unavailable: "LLM 不可用 · 仅检索",
      degraded: "未完成",
      cancelled: "已取消 · 部分输出不是完整答案",
      timeout: "超时 · 回答未完成",
      clarification: "请澄清或修改问答方式",
      local_unavailable: "本地模型 / 索引不可用",
      answer: "",
    };
    const label = map[k] || "";
    if (!label) return "";
    return (
      '<div class="kind-badge">' + escapeHtml(label) + "</div>"
    );
  }

  function appendMessage(role, html, extraClass) {
    ensureMessagesVisible();
    const div = document.createElement("div");
    div.className =
      "message is-" + role + (extraClass ? " " + extraClass : "");
    const icon =
      role === "user"
        ? '<svg class="message-icon" viewBox="0 0 24 24" width="18" height="18" fill="#1D9CFF"><path d="M12 12a5 5 0 1 0-5-5 5 5 0 0 0 5 5zm0 2c-4.4 0-8 2.2-8 5v1h16v-1c0-2.8-3.6-5-8-5z"/></svg>'
        : '<svg class="message-icon" viewBox="0 0 24 24" width="18" height="18" fill="#1D9CFF"><path d="M12 2l1.2 3.6L17 7l-3.8 1.2L12 12l-1.2-3.8L7 7l3.8-1.4L12 2zm6.5 9l.8 2.3 2.2.7-2.2.7-.8 2.3-.8-2.3-2.2-.7 2.2-.7.8-2.3z"/></svg>';
    // Safe: fixed SVG icon + pre-escaped/composed html only.
    div.innerHTML =
      '<div class="message-row">' +
      icon +
      '<div class="message-body">' +
      html +
      "</div></div>";
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  function setAssistantBody(row, answerText, sources, meta, opts) {
    opts = opts || {};
    meta = meta || {};
    const body = row.querySelector(".message-body");
    if (!body) return;
    citeSeq += 1;
    const groupId = "g" + citeSeq;
    row.dataset.citeGroup = groupId;
    row._sources = sources || [];
    row._answerText = answerText || "";
    row._meta = meta;

    const kind = meta.answer_kind || "answer";
    row.classList.remove(
      "is-pending",
      "is-refusal",
      "is-advice",
      "is-llm-unavailable",
      "is-degraded",
      "is-answer"
    );
    row.classList.add(kindClass(kind));

    // During stream we may pass preformatted; for final always rebuild
    const provenance = meta.document_ids ? '<div class="message-provenance">范围 ' + escapeHtml(meta.document_ids.join(", ")) + ' · ' + escapeHtml(meta.query_mode || "qa") + ' · ' + escapeHtml(meta.answer_language || "auto") + ' · epoch ' + escapeHtml(meta.scope_epoch ?? "—") + '</div>' : '';
    body.innerHTML = provenance +
      kindBadge(kind) +
      formatAnswerHtml(answerText || "") +
      citationsHtml(sources || [], groupId) +
      debugPanelHtml(sources || [], meta);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    // P1: only open PDF/evidence after final setAssistantBody (never mid-stream)
    if (opts.silent || (row._scopeEpoch != null && row._scopeEpoch !== scopeEpoch)) return;
    if (sources && sources.length) {
      followAnswerCitation(sources, answerText || {}, { force: false });
      const idx = firstAnswerCiteIndex(answerText);
      highlightActiveCite(groupId, Math.min(idx, sources.length || 1));
    } else {
      followAnswerCitation([], answerText || "", { forceClear: false });
    }

    if (meta.embedding_provider) {
      lastEmbeddingProvider = meta.embedding_provider;
      if (embedMeta) {
        embedMeta.textContent = "embed:" + meta.embedding_provider;
      }
    }
  }

  function highlightActiveCite(groupId, index) {
    document.querySelectorAll(".cite-card.is-active").forEach(function (el) {
      el.classList.remove("is-active");
    });
    if (!groupId || !index) return;
    const card = document.querySelector(
      '.cite-card[data-cite-group="' +
        groupId +
        '"][data-cite-index="' +
        index +
        '"]'
    );
    if (card) {
      card.classList.add("is-active");
      try {
        card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      } catch (e) {}
    }
  }

  function activateCitation(index, sources, groupId) {
    const list = sources || activeSources || [];
    const i = Number(index) - 1;
    if (i < 0 || i >= list.length) return;
    const s = list[i];
    if (s.source_status === "deleted" || deletedIds.has(Number(s.document_id))) { showToast("来源已删除"); return; }
    openCitedPdf(
      s.document_name || s.source || s.filename,
      s.page_number ?? s.page,
      s,
      Number(index),
      {}
    );
    highlightActiveCite(groupId, index);
  }

  function normalizeSources(data) {
    if (!data) return [];
    let list = [];
    if (Array.isArray(data.retrieved_chunks) && data.retrieved_chunks.length) {
      list = data.retrieved_chunks;
    } else if (Array.isArray(data.sources)) {
      list = data.sources;
    } else if (Array.isArray(data.citations)) {
      list = data.citations;
    }
    return list.map(src => deletedIds.has(Number(src.document_id)) ? {chunk_id:src.chunk_id, document_id:src.document_id, document_name:src.document_name, page_number:src.page_number, source_status:"deleted"} : src);
  }

  function responseMeta(data) {
    data = data || {};
    return {
      document_ids:data.document_ids, query_mode:data.query_mode, answer_language:data.answer_language,
      scope_epoch:data.scope_epoch, session_id:data.session_id, source_status:data.source_status,
      answer_kind: data.answer_kind || "answer",
      embedding_provider: data.embedding_provider || lastEmbeddingProvider,
      confidence_score: data.confidence_score,
    };
  }

  async function askOnce(snapshot) {
    const res = await fetch("/api/v1/queries/ask", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify(snapshot.body), signal:snapshot.controller.signal
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(errorText(data));
    return data;
  }

  async function askStream(snapshot, onDelta, onFinal) {
    const res = await fetch("/api/v1/queries/ask", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify(snapshot.body), signal:snapshot.controller.signal
    });
    if (!res.ok) throw new Error(errorText(await res.json().catch(() => ({}))));
    if (!res.body || !res.body.getReader) throw new Error("浏览器无法读取答案流，请关闭流式后重试。");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "", full = "", finalPayload = null;
    function consume(block) {
      const data = block.split(/\r?\n/).filter(line => line.startsWith("data:")).map(line => line.slice(5).trimStart()).join("\n");
      if (!data || data === "[DONE]") return;
      const obj = JSON.parse(data);
      if (obj.type === "start" && snapshot.epoch === scopeEpoch) modeStatus.textContent = "本次方式：" + modeLabel(obj.query_mode);
      if (obj.type === "token") { full += obj.content || ""; onDelta(full); }
      if (obj.type === "end") finalPayload = obj;
      if (obj.type === "error") throw new Error(obj.message || "答案流异常中止");
    }
    try {
      while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream:true});
        const blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop() || "";
        blocks.forEach(consume);
      }
      buffer += decoder.decode();
      if (buffer.trim()) consume(buffer);
      if (!finalPayload) throw new Error("答案流提前结束，部分输出不是完整答案。");
      onFinal(finalPayload);
    } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
  }

  function modeLabel(mode) { return ({auto:"自动识别",qa:"普通问答",compare:"逐文档比较",clarification:"请澄清或手动选择方式"})[mode] || mode; }
  queryMode.addEventListener("change", () => { modeStatus.textContent = "当前方式：" + modeLabel(queryMode.value); });

  async function handleAsk(question) {
    const q = String(question || "").trim();
    if (!q || isLoading) return;
    if (!selectedIds.length) { showError("请先勾选至少一份可提问文档。"); return; }
    if (queryMode.value === "compare" && selectedIds.length > 5) { showError("最多比较 5 份文档，请缩小范围。"); return; }
    const snapshot = {
      body:{question:q, document_ids:selectedIds.slice(), answer_language:answerLanguage.value,
        query_mode:queryMode.value, stream:streamToggle.checked, show_sources:true, session_id:sessionId},
      document_ids:selectedIds.slice(), epoch:scopeEpoch, preparation:scopeSync, controller:new AbortController(), cancelReason:null
    };
    currentRequest = snapshot;
    abortController = snapshot.controller;
    showError("");
    setLoading(true, "retrieve");
    let assistantRow;
    try {
      const preparation = await snapshot.preparation;
      if (!preparation || preparation.error) throw (preparation && preparation.error) || new Error("范围尚未准备完成");
      snapshot.body.scope_epoch = preparation.scope_epoch;
      if (snapshot.controller.signal.aborted) throw new DOMException("Stopped", "AbortError");
      ensureMessagesVisible();
      appendMessage("user", escapeHtml(q));
      assistantRow = appendMessage("assistant", "…", "is-pending");
      assistantRow._scopeEpoch = snapshot.epoch;
      setStatus("generate");
      function finish(data) {
        if (snapshot.cancelReason || snapshot.document_ids.some(id => deletedIds.has(id))) {
          setAssistantBody(assistantRow, snapshot.cancelReason || "来源已删除，回答已中止。", [], {answer_kind:"cancelled", document_ids:snapshot.document_ids});
          return;
        }
        if (snapshot.epoch === scopeEpoch) modeStatus.textContent = "本次方式：" + modeLabel(data.query_mode);
        setAssistantBody(assistantRow, data.answer || data.response || "", normalizeSources(data), responseMeta(data));
      }
      if (snapshot.body.stream) {
        await askStream(snapshot, partial => {
          const body = assistantRow.querySelector(".message-body");
          if (body && !snapshot.cancelReason) body.innerHTML = formatAnswerHtml(partial);
        }, finish);
      } else { finish(await askOnce(snapshot)); }
    } catch (err) {
      if (err.name !== "AbortError") showError(err.message || String(err));
      if (assistantRow) setAssistantBody(assistantRow, snapshot.cancelReason || (err.name === "AbortError" ? "已停止，回答未完成。" : "请求未完成：" + err.message), [], {answer_kind:"cancelled", document_ids:snapshot.document_ids});
    } finally {
      if (currentRequest === snapshot) {
        currentRequest = null;
        abortController = null;
        setLoading(false);
        input.value = "";
        input.focus();
        updateScopeLabel();
      }
    }
  }

  function resetChat() {
    if (currentRequest) { currentRequest.cancelReason = "已开始新对话，旧回答已停止。"; currentRequest.controller.abort(); }
    currentRequest = null;
    abortController = null;
    sessionId = "s_" + Math.random().toString(36).slice(2) + Date.now().toString(36);
    try { localStorage.setItem("insur_session_id", sessionId); } catch (_) {}
    scopeEpoch += 1;
    syncScope();
    clearPdfSource();
    setLoading(false);
    showError("");
    messagesEl.innerHTML = "";
    messagesEl.hidden = true;
    emptyChat.hidden = false;
    emptyChat.removeAttribute("aria-hidden");
    activeSources = [];
    lastStickyView = null;
    currentView = { name: null, url: null, page: null, index: null, excerpt: "" };
    hidePdfViews();
    pdfSelect.hidden = true;
    pageBadge.hidden = true;
    pdfTitle.textContent = "引用原文";
    pdfSubtitle.textContent = "提问后自动打开答案引用的条款 PDF";
    setEvidence(null);
    input.value = "";
    sendBtn.disabled = true;
    input.focus();
  }

  messagesEl.addEventListener("click", function (e) {
    const ref = e.target.closest(".cite-ref, .cite-card");
    if (!ref) return;
    e.preventDefault();
    const idx = ref.getAttribute("data-cite-index");
    const groupId =
      ref.getAttribute("data-cite-group") ||
      (ref.closest(".message") && ref.closest(".message").dataset.citeGroup);
    const row = ref.closest(".message");
    const sources = (row && row._sources) || activeSources;
    activateCitation(idx, sources, groupId);
  });

  pdfSelect.addEventListener("change", function () {
    const hit = pdfs.find(p => p.document_id === Number(pdfSelect.value));
    if (!hit) return;
    const match = activeSources.find(src => Number(src.document_id) === hit.document_id);
    openCitedPdf(hit.name, (match && match.page_number) || 1, match || {document_id:hit.document_id, document_name:hit.name}, 1, {});
  });

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    handleAsk(input.value);
  });

  input.addEventListener("input", function () {
    updateScopeLabel();
  });

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (String(input.value || "").trim()) handleAsk(input.value);
    }
  });

  resetBtn.addEventListener("click", function (e) {
    e.preventDefault();
    resetChat();
  });

  stopBtn.addEventListener("click", function (e) {
    e.preventDefault();
    if (abortController) abortController.abort();
  });

  if (debugToggle) {
    debugToggle.addEventListener("click", function (e) {
      e.preventDefault();
      debugMode = !debugMode;
      debugToggle.classList.toggle("is-active", debugMode);
      debugToggle.textContent = debugMode ? "调试开" : "调试";
      // Re-render last assistant bodies if any
      document.querySelectorAll(".message.is-assistant").forEach(function (row) {
        if (row._answerText != null) {
          setAssistantBody(row, row._answerText, row._sources || [], row._meta || {});
        }
      });
    });
  }

  document.querySelectorAll(".prompt-chip").forEach(function (btn) {
    btn.addEventListener("click", function () {
      handleAsk(btn.getAttribute("data-q") || btn.textContent);
    });
  });

  (function setupResize() {
    let dragging = false;
    resizeHandle.addEventListener("mousedown", function (e) {
      e.preventDefault();
      dragging = true;
      document.body.style.cursor =
        window.innerWidth <= 860 ? "row-resize" : "col-resize";
      document.body.style.userSelect = "none";
    });
    window.addEventListener("mouseup", function () {
      if (!dragging) return;
      dragging = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    });
    window.addEventListener("mousemove", function (e) {
      if (!dragging) return;
      const rect = panelGroup.getBoundingClientRect();
      const isVertical = window.innerWidth <= 860;
      if (isVertical) {
        const y = e.clientY - rect.top;
        const ratio = Math.min(0.78, Math.max(0.22, y / rect.height));
        pdfPanel.style.flex = "0 0 " + ratio * 100 + "%";
        aiPanel.style.flex = "0 0 " + (1 - ratio) * 100 + "%";
      } else {
        const x = e.clientX - rect.left;
        const ratio = Math.min(0.72, Math.max(0.28, x / rect.width));
        pdfPanel.style.flex = "0 0 " + ratio * 100 + "%";
        aiPanel.style.flex = "1 1 " + (1 - ratio) * 100 + "%";
      }
    });
  })();

  pollLibrary();
  loadHealth();
  sendBtn.disabled = true;
})();
