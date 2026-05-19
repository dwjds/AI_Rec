const API_BASE = window.AGENT_API_BASE || "";
const storedUser = JSON.parse(localStorage.getItem("mooc_agent_user") || "null");
let currentUser = storedUser || null;
let USER_ID = currentUser?.user_id || "";
const SESSION_ID = window.AGENT_SESSION_ID || localStorage.getItem("mooc_agent_session") || `web_${Date.now()}`;
localStorage.setItem("mooc_agent_session", SESSION_ID);

let resources = [];
let selectedDetail = null;
const savedResourceIds = new Set();
let savedResourceCache = [];
let agentSettings = { llm_generation: true, llm_rerank: true, hide_backend_details: true };
let noteDraft = { id: "", title: "", content: "", isNew: false };

const state = {
  mode: "resources",
  selectedId: "",
  view: "grid",
  currentQuery: "推荐人工智能入门课程",
  total: 0,
  page: 1,
  pageSize: 6,
  filters: {
    type: "all",
    difficulty: "all",
    knowledge: "all",
    sort: "score",
  },
};

const panels = {
  answer: document.getElementById("answerPanel"),
  resources: document.getElementById("resourcesPanel"),
  path: document.getElementById("pathPanel"),
  diagnosis: document.getElementById("diagnosisPanel"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function difficultyClass(value) {
  return value === "easy" ? "easy" : value === "hard" ? "hard" : "medium";
}

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function requireLogin() {
  if (USER_ID) return true;
  openAuthModal("login");
  return false;
}

function setChatVisible(visible) {
  document.getElementById("chatForm").classList.toggle("hidden", !visible);
}

function setDetailVisible(visible) {
  document.querySelector(".detail-panel").classList.toggle("hidden", !visible);
  document.getElementById("app").classList.toggle("no-detail", !visible);
}

async function apiFetch(path, options = {}) {
  const response = await fetch(apiUrl(path), {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

function normalizeResource(item) {
  const knowledge = item.knowledge_points || item.knowledge || [];
  return {
    id: item.id,
    title: item.title || "未命名资源",
    type: item.resource_type || item.type || "course",
    typeLabel: item.type_label || item.typeLabel || "课程",
    difficulty: item.difficulty || "medium",
    difficultyLabel: item.difficulty_label || item.difficultyLabel || "中等",
    knowledge,
    reason: item.reason || item.description || "该资源与当前学习问题相关。",
    score: Number(item.score || 0),
    duration: buildDuration(item),
    source: item.source || "MOOPer",
    updatedAt: "资源库",
    accent: accentForType(item.resource_type || item.type),
    evidence: normalizeEvidence(item.evidence || []),
    raw: item.raw || item,
  };
}

function normalizeEvidence(items) {
  return items
    .map((item) => String(item || "").trim())
    .filter(Boolean)
    .filter((item) => !/^E\d+$/.test(item) && !/^chunk/i.test(item))
    .slice(0, 8);
}

function buildDuration(item) {
  const chapters = Number(item.chapter_count || 0);
  const exercises = Number(item.exercise_count || 0);
  const challenges = Number(item.challenge_count || 0);
  const parts = [];
  if (chapters) parts.push(`${chapters} 章`);
  if (exercises) parts.push(`${exercises} 练习`);
  if (challenges) parts.push(`${challenges} 任务`);
  return parts.join(" · ") || `${Number(item.visits || 0)} 次访问`;
}

function accentForType(type) {
  if (type === "exercise" || type === "challenge") return "#18b56f";
  if (type === "chapter") return "#f5b21b";
  if (type === "knowledge_point" || type === "topic") return "#7957e6";
  return "#1d6dff";
}

function filteredResources() {
  const items = resources.filter((item) => {
    const matchType =
      state.filters.type === "all" ||
      item.type === state.filters.type ||
      (state.filters.type === "video" && item.type === "course") ||
      (state.filters.type === "article" && item.type === "chapter");
    const matchDifficulty = state.filters.difficulty === "all" || item.difficulty === state.filters.difficulty;
    const matchKnowledge = state.filters.knowledge === "all" || item.knowledge.includes(state.filters.knowledge);
    return matchType && matchDifficulty && matchKnowledge;
  });

  if (state.filters.sort === "difficulty") {
    const order = { easy: 1, medium: 2, hard: 3 };
    return [...items].sort((a, b) => order[a.difficulty] - order[b.difficulty]);
  }
  if (state.filters.sort === "duration") {
    return [...items].sort((a, b) => String(a.duration).localeCompare(String(b.duration), "zh-CN"));
  }
  return [...items].sort((a, b) => b.score - a.score);
}

function renderResources() {
  const grid = document.getElementById("resourceGrid");
  const items = filteredResources();
  document.getElementById("resultCount").textContent = `共找到 ${items.length} 个资源`;
  grid.classList.toggle("list-view", state.view === "list");

  if (!items.length) {
    grid.innerHTML = `<div class="empty-state">暂无资源结果。可以换一个问题，或确认后端 API 与数据库是否已启动。</div>`;
    renderDetail();
    return;
  }

  if (!state.selectedId || !items.some((item) => item.id === state.selectedId)) {
    state.selectedId = items[0].id;
  }

  grid.innerHTML = items
    .map((item) => {
      const selected = item.id === state.selectedId ? "selected" : "";
      const knowledge = item.knowledge.slice(0, 2).join("、") || "暂无知识点";
      const icon = item.type === "exercise" || item.type === "challenge" ? "T" : item.type === "chapter" ? "C" : "R";
      return `
        <article class="resource-card ${selected}" data-resource-id="${escapeHtml(item.id)}" tabindex="0">
          <div class="card-head">
            <div class="card-icon" style="background:${escapeHtml(item.accent)}">${icon}</div>
            <span class="card-type">${escapeHtml(item.typeLabel)}</span>
          </div>
          <h3>${escapeHtml(item.title)}</h3>
          <div class="meta-lines">
            <span>难度：<b class="difficulty ${difficultyClass(item.difficulty)}">${escapeHtml(item.difficultyLabel)}</b></span>
            <span>知识点：${escapeHtml(knowledge)}</span>
            <span>推荐理由：${escapeHtml(item.reason)}</span>
          </div>
          <div class="feedback-row">
            <button type="button" data-feedback="helpful">有用</button>
            <button type="button" data-feedback="too_hard">太难</button>
            <button type="button" data-feedback="too_easy">太简单</button>
            <button type="button" data-feedback="not_interested">不感兴趣</button>
          </div>
        </article>
      `;
    })
    .join("");

  grid.querySelectorAll(".resource-card").forEach((card) => {
    card.addEventListener("click", (event) => {
      if (event.target.matches("[data-feedback]")) return;
      selectResource(card.dataset.resourceId);
    });
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectResource(card.dataset.resourceId);
      }
    });
  });
  grid.querySelectorAll("[data-feedback]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.stopPropagation();
      const card = button.closest(".resource-card");
      await submitFeedback(card.dataset.resourceId, button.dataset.feedback, button);
    });
  });
  renderDetail();
  renderPagination();
}

async function selectResource(id) {
  state.selectedId = id;
  renderResources();
  document.querySelector(".detail-panel").classList.add("open");
  try {
    selectedDetail = await apiFetch(`/api/resources/${encodeURIComponent(id)}`);
  } catch (error) {
    selectedDetail = null;
  }
  renderDetail();
}

function renderDetail() {
  const item = resources.find((entry) => entry.id === state.selectedId);
  if (!item) {
    document.getElementById("detailTitle").textContent = "暂无资源详情";
    document.getElementById("detailReason").textContent = "请选择一个资源查看详情。";
    document.getElementById("evidenceList").innerHTML = "";
    return;
  }
  const detail = selectedDetail && selectedDetail.resource && selectedDetail.resource.id === item.id ? selectedDetail : null;
  const chapters = detail?.chapters || item.raw?.chapters || [];
  const exercises = detail?.exercises || item.raw?.exercises || [];
  const knowledge = detail?.knowledge_points || item.raw?.knowledge_points || item.knowledge || [];
  const evidence = item.evidence.length ? item.evidence : knowledge.slice(0, 5).map((entry) => entry.title || entry);

  document.getElementById("detailType").textContent = item.typeLabel;
  document.getElementById("detailPreviewTitle").textContent = item.title;
  document.getElementById("detailDuration").textContent = item.duration;
  document.getElementById("detailDurationTag").textContent = item.duration;
  document.getElementById("detailTitle").textContent = item.title;
  document.getElementById("detailDifficulty").textContent = item.difficultyLabel;
  document.getElementById("detailScore").textContent = `推荐度 ${Math.round(item.score)}%`;
  document.getElementById("detailSource").textContent = `来源：${item.source}　章节 ${chapters.length}　练习 ${exercises.length}`;
  document.getElementById("detailReason").textContent = item.reason;
  document.querySelector(".primary-button").textContent = "记录查看";
  document.querySelector(".primary-button").disabled = false;
  document.querySelector(".secondary-button").textContent = savedResourceIds.has(item.id) ? "已加入我的学习" : "加入我的学习";
  document.querySelector(".secondary-button").disabled = false;
  document.getElementById("evidenceList").innerHTML = evidence
    .slice(0, 5)
    .map((entry, index) => `<li><span>${index + 1}</span><strong>${escapeHtml(entry.title || entry)}</strong></li>`)
    .join("");
  document.querySelector(".link-button").textContent = `查看全部 ${Math.max(evidence.length, 1)} 条证据来源`;
}

function renderPagination() {
  const totalPages = Math.max(1, Math.ceil(state.total / state.pageSize));
  const pagination = document.querySelector(".pagination");
  pagination.innerHTML = `
    <span>共 ${state.total} 条</span>
    <button type="button" data-page="prev" ${state.page <= 1 ? "disabled" : ""}>‹</button>
    ${Array.from({ length: Math.min(totalPages, 5) }, (_, index) => index + 1)
      .map(
        (page) => `<button type="button" data-page="${page}" class="${page === state.page ? "active" : ""}">${page}</button>`
      )
      .join("")}
    <button type="button" data-page="next" ${state.page >= totalPages ? "disabled" : ""}>›</button>
    <select id="pageSizeSelect" aria-label="每页条数">
      <option value="6" ${state.pageSize === 6 ? "selected" : ""}>6 条/页</option>
      <option value="12" ${state.pageSize === 12 ? "selected" : ""}>12 条/页</option>
    </select>
  `;
  pagination.querySelectorAll("[data-page]").forEach((button) => {
    button.addEventListener("click", async () => {
      const target = button.dataset.page;
      if (target === "prev") state.page = Math.max(1, state.page - 1);
      else if (target === "next") state.page = Math.min(totalPages, state.page + 1);
      else state.page = Number(target);
      await loadResources(state.currentQuery, false);
    });
  });
  pagination.querySelector("#pageSizeSelect").addEventListener("change", async (event) => {
    state.pageSize = Number(event.target.value);
    state.page = 1;
    await loadResources(state.currentQuery, false);
  });
}

function switchMode(mode) {
  document.querySelector(".content-card").classList.remove("utility-active");
  setChatVisible(true);
  setDetailVisible(true);
  if (mode === "answer") ensureAnswerPanel();
  state.mode = mode;
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  Object.entries(panels).forEach(([name, panel]) => {
    panel.classList.toggle("hidden", name !== mode);
  });
}

function bindEvents() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => switchMode(button.dataset.mode));
  });

  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      if (button.dataset.section === "path") switchMode("path");
      if (button.dataset.section === "diagnosis") switchMode("diagnosis");
      if (button.dataset.section === "library") switchMode("resources");
      if (button.dataset.section === "qa") switchMode("answer");
      if (["profile", "notes"].includes(button.dataset.section)) {
        renderUtilityPage(button.dataset.section);
      }
    });
  });

  document.getElementById("typeFilter").addEventListener("change", async (event) => {
    state.filters.type = event.target.value;
    state.page = 1;
    await loadResources(state.currentQuery);
  });
  document.getElementById("difficultyFilter").addEventListener("change", (event) => {
    state.filters.difficulty = event.target.value;
    renderResources();
  });
  document.getElementById("knowledgeFilter").addEventListener("change", (event) => {
    state.filters.knowledge = event.target.value;
    renderResources();
  });
  document.getElementById("sortSelect").addEventListener("change", (event) => {
    state.filters.sort = event.target.value;
    renderResources();
  });

  document.getElementById("gridToggle").addEventListener("click", () => setView("grid"));
  document.getElementById("listToggle").addEventListener("click", () => setView("list"));
  document.getElementById("closeDetail").addEventListener("click", () => {
    document.querySelector(".detail-panel").classList.remove("open");
  });
  document.getElementById("expandSummary").addEventListener("click", () => {
    document.getElementById("taskSummary").textContent =
      "我会根据你的问题选择回答、推荐资源、学习路线或学习诊断，并尽量把推荐依据落到具体课程资料上。";
  });

  document.getElementById("chatForm").addEventListener("submit", handleChatSubmit);
  document.querySelector(".profile-chip").addEventListener("click", () => openAuthModal(USER_ID ? "profile" : "login"));
  document.querySelector(".primary-button").addEventListener("click", openSelectedResource);
  document.querySelector(".secondary-button").addEventListener("click", saveSelectedResource);
  document.querySelector(".link-button").addEventListener("click", expandEvidenceList);
}

function setView(view) {
  state.view = view;
  document.getElementById("gridToggle").classList.toggle("active", view === "grid");
  document.getElementById("listToggle").classList.toggle("active", view === "list");
  renderResources();
}

async function handleChatSubmit(event) {
  event.preventDefault();
  if (!requireLogin()) return;
  const input = document.getElementById("chatInput");
  const query = input.value.trim();
  if (!query) return;
  state.currentQuery = query;
  document.getElementById("taskTitle").textContent = query;
  document.getElementById("taskSummary").textContent = "Agent 正在分析任务、读取记忆、检索证据并生成回答。";
  input.value = "";
  setBusy(true);

  try {
    if (USER_ID) {
      await refreshAgentSettings();
    }
    const payload = {
      user_id: USER_ID,
      session_id: SESSION_ID,
      query,
      use_llm_generation: Boolean(agentSettings.llm_generation),
      use_llm_rerank: Boolean(agentSettings.llm_rerank),
      top_k: 8,
    };
    await streamAgentResult(payload);
  } catch (error) {
    try {
      const data = await apiFetch("/api/agent", {
        method: "POST",
        body: JSON.stringify({
          user_id: USER_ID,
          session_id: SESSION_ID,
          query,
          use_llm_generation: Boolean(agentSettings.llm_generation),
          use_llm_rerank: Boolean(agentSettings.llm_rerank),
          top_k: 8,
        }),
      });
      applyAgentResult(data);
    } catch (fallbackError) {
      document.getElementById("taskSummary").textContent = `后端请求失败：${fallbackError.message}`;
      document.getElementById("answerText").textContent = "Agent API 暂不可用，请确认 FastAPI 服务已启动。";
      switchMode("answer");
    }
  } finally {
    setBusy(false);
  }
}

async function refreshSavedResources() {
  if (!USER_ID) return;
  const data = await apiFetch(`/api/users/${encodeURIComponent(USER_ID)}/saved-resources`);
  savedResourceCache = data.resources || [];
  savedResourceIds.clear();
  savedResourceCache.forEach((item) => savedResourceIds.add(item.id));
}

async function refreshAgentSettings() {
  if (!USER_ID) return agentSettings;
  const data = await apiFetch(`/api/users/${encodeURIComponent(USER_ID)}/settings`);
  agentSettings = data.settings || agentSettings;
  return agentSettings;
}

async function streamAgentResult(payload) {
  if (!window.ReadableStream) {
    throw new Error("browser stream unsupported");
  }
  ensureAnswerPanel();
  switchMode("answer");
  document.getElementById("answerText").textContent = "";
  document.getElementById("apiStatusTag").textContent = "流式生成中";

  const response = await fetch(apiUrl("/api/agent/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok || !response.body) {
    throw new Error(await response.text() || `HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let finalData = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const rawEvent of events) {
      const parsed = parseSseEvent(rawEvent);
      if (!parsed) continue;
      if (parsed.event === "meta") {
        applyAgentResult(parsed.data, { preserveAnswer: true, keepAnswerVisible: true });
        document.getElementById("answerText").textContent = "";
        document.getElementById("apiStatusTag").textContent = "流式生成中";
        switchMode("answer");
      } else if (parsed.event === "answer_delta") {
        appendAnswerDelta(parsed.data.text || "");
      } else if (parsed.event === "done") {
        finalData = parsed.data;
      } else if (parsed.event === "error") {
        throw new Error(parsed.data.message || "stream error");
      }
    }
  }
  if (buffer.trim()) {
    const parsed = parseSseEvent(buffer);
    if (parsed?.event === "done") finalData = parsed.data;
  }
  if (finalData) {
    applyAgentResult(finalData, { keepAnswerVisible: true });
    document.getElementById("apiStatusTag").textContent = "生成完成";
  }
}

function parseSseEvent(rawEvent) {
  const lines = String(rawEvent || "").split("\n");
  let event = "message";
  const dataLines = [];
  lines.forEach((line) => {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  });
  if (!dataLines.length) return null;
  return { event, data: JSON.parse(dataLines.join("\n")) };
}

function appendAnswerDelta(text) {
  const target = document.getElementById("answerText");
  target.textContent += text;
}

function applyAgentResult(data, options = {}) {
  ensureAnswerPanel();
  document.getElementById("taskSummary").textContent = buildRequestSummary(data);
  if (!options.preserveAnswer) {
    document.getElementById("answerText").textContent = data.answer || "暂无回答。";
  }
  document.getElementById("apiStatusTag").textContent = "已连接资源库";

  if (Array.isArray(data.recommendations)) {
    resources = data.recommendations.map(normalizeResource);
    state.total = resources.length;
    state.page = 1;
    state.selectedId = resources[0]?.id || "";
    selectedDetail = null;
    renderResources();
  }

  renderPlannerPanel(data);
  renderDiagnosisPanel(data);

  if (options.keepAnswerVisible) switchMode("answer");
  else if (data.pipeline === "agent_loop_learning_path") switchMode("path");
  else if (data.pipeline === "agent_loop_diagnosis") switchMode("diagnosis");
  else if (data.pipeline === "rag_qa" || data.pipeline === "clarification") switchMode("answer");
  else switchMode(resources.length ? "resources" : "answer");
}

function buildRequestSummary(data) {
  const task = {
    recommend: "资源推荐",
    qa: "知识问答",
    learning_path: "学习路线",
    diagnosis: "学习诊断",
    feedback: "反馈调整",
  }[data.task_type || ""] || "学习咨询";
  const count = Array.isArray(data.recommendations) ? data.recommendations.length : 0;
  if (count) return `已完成${task}，为你筛选出 ${count} 个可参考资源。`;
  if (data.pipeline === "clarification") return "这个问题还需要补充一点学习背景，我会先问最关键的信息。";
  return `已完成${task}，回答已生成在下方。`;
}

function renderPlannerPanel(data) {
  const panel = document.getElementById("pathPanel");
  const answer = data.answer || "";
  panel.innerHTML = `
    <div class="timeline">
      ${answer
        .split("\n")
        .filter(Boolean)
        .slice(0, 6)
        .map(
          (line, index) => `
          <div class="timeline-step">
            <span>${index + 1}</span>
            <div>
              <h3>${index === 0 ? "路线建议" : "阶段要点"}</h3>
              <p>${escapeHtml(line.replace(/^\d+[.、]\s*/, ""))}</p>
            </div>
          </div>`
        )
        .join("")}
    </div>
  `;
}

function renderDiagnosisPanel(data) {
  const panel = document.getElementById("diagnosisPanel");
  const lines = (data.answer || "").split("\n").filter(Boolean);
  const chunks = lines.length ? lines.slice(0, 3) : ["暂无诊断结果", "可以补充学习卡点", "Agent 会继续检索相关补救资源"];
  panel.innerHTML = `
    <div class="diagnosis-grid">
      ${chunks
        .map(
          (line, index) => `
          <article>
            <h3>${["可能卡点", "原因判断", "下一步"][index] || "补充建议"}</h3>
            <p>${escapeHtml(line.replace(/^[-*]\s*/, ""))}</p>
          </article>`
        )
        .join("")}
    </div>
  `;
}

async function loadResources(query, resetPage = true) {
  if (resetPage) state.page = 1;
  setBusy(true);
  try {
    const type = state.filters.type === "video" ? "course" : state.filters.type;
    const data = await apiFetch(
      `/api/resources/search?query=${encodeURIComponent(query)}&resource_type=${encodeURIComponent(type)}&limit=${state.pageSize}&offset=${(state.page - 1) * state.pageSize}`
    );
    resources = (data.resources || []).map(normalizeResource);
    state.total = Number(data.total || resources.length);
    state.selectedId = resources[0]?.id || "";
    selectedDetail = null;
    document.getElementById("taskSummary").textContent = `已从资源库加载当前页 ${resources.length} 个候选资源。`;
    document.getElementById("apiStatusTag").textContent = "API 已连接";
    renderResources();
  } catch (error) {
    resources = [];
    document.getElementById("apiStatusTag").textContent = "API 未连接";
    document.getElementById("taskSummary").textContent = `资源 API 暂不可用：${error.message}`;
    renderResources();
  } finally {
    setBusy(false);
  }
}

async function submitFeedback(resourceId, feedbackType, button) {
  if (!requireLogin()) return;
  const original = button.textContent;
  button.textContent = "已记录";
  button.disabled = true;
  button.classList.add("active");
  try {
    await apiFetch("/api/feedback", {
      method: "POST",
      body: JSON.stringify({
        user_id: USER_ID,
        resource_id: resourceId,
        feedback_type: feedbackType,
        comment: `web:${feedbackType}`,
      }),
    });
    setTimeout(() => {
      button.textContent =
        button.classList.contains("secondary-button") && savedResourceIds.has(resourceId)
          ? "已加入我的学习"
          : original;
      button.disabled = false;
      button.classList.remove("active");
    }, 900);
  } catch (error) {
    button.textContent = "失败";
    setTimeout(() => {
      button.textContent = original;
      button.disabled = false;
      button.classList.remove("active");
    }, 1200);
  }
}

async function openSelectedResource() {
  const item = resources.find((entry) => entry.id === state.selectedId);
  if (!item) return;
  if (!requireLogin()) return;
  await apiFetch("/api/feedback", {
    method: "POST",
    body: JSON.stringify({
      user_id: USER_ID,
      resource_id: item.id,
      feedback_type: "opened",
      comment: "web:open_resource",
    }),
  }).catch(() => {});
  const button = document.querySelector(".primary-button");
  const original = button.textContent;
  button.textContent = "已记录";
  button.disabled = true;
  setTimeout(() => {
    button.textContent = original;
    button.disabled = false;
  }, 900);
  document.getElementById("taskSummary").textContent = `已记录你打开《${item.title}》的学习行为。`;
}

async function saveSelectedResource() {
  const item = resources.find((entry) => entry.id === state.selectedId);
  if (!item) return;
  if (!requireLogin()) return;
  const button = document.querySelector(".secondary-button");
  const original = button.textContent;
  button.textContent = "保存中";
  button.disabled = true;
  try {
    await apiFetch("/api/users/saved-resources", {
      method: "POST",
      body: JSON.stringify({
        user_id: USER_ID,
        resource_id: item.id,
      }),
    });
    await submitFeedback(item.id, "saved", button);
    await refreshSavedResources();
    document.getElementById("taskSummary").textContent = `已将《${item.title}》加入你的学习记录。`;
  } catch (error) {
    button.textContent = "保存失败";
    setTimeout(() => {
      button.textContent = original;
      button.disabled = false;
    }, 1000);
  }
  renderDetail();
}

function expandEvidenceList() {
  const item = resources.find((entry) => entry.id === state.selectedId);
  if (!item) return;
  const evidence = item.evidence.length ? item.evidence : ["课程资料：" + item.title, "MOOPer 资源库课程元数据"];
  document.getElementById("evidenceList").innerHTML = evidence
    .map((entry, index) => `<li><span>${index + 1}</span><strong>${escapeHtml(entry)}</strong></li>`)
    .join("");
}

async function renderUtilityPage(section) {
  document.querySelector(".content-card").classList.add("utility-active");
  setChatVisible(false);
  setDetailVisible(section !== "notes");
  Object.entries(panels).forEach(([name, panel]) => {
    panel.classList.toggle("hidden", name !== "answer");
  });
  document.querySelectorAll(".tab-button").forEach((button) => button.classList.remove("active"));

  const titles = {
    profile: "我的学习",
    notes: "笔记本",
  };
  document.getElementById("taskTitle").textContent = titles[section] || "学习工作台";
  document.getElementById("apiStatusTag").textContent = USER_ID ? "已登录" : "未登录";

  if (section === "profile") return renderProfilePage();
  if (section === "notes") return renderNotesPage();
}

function ensureAnswerPanel() {
  if (document.getElementById("answerText")) return;
  panels.answer.innerHTML = `
    <article class="agent-answer">
      <h2>回答</h2>
      <p id="answerText">输入学习问题后，Agent 会在这里给出基于证据的回答。</p>
    </article>
  `;
}

async function renderProfilePage() {
  if (!USER_ID) {
    document.getElementById("taskSummary").textContent = "登录后可以查看从用户数据库读取的学习画像。";
    panels.answer.innerHTML = `
      <article class="utility-page">
        <h2>我的学习</h2>
        <p>你还没有登录。登录后会读取 app.db 中的真实用户画像。</p>
        <button id="utilityLogin" type="button">登录 / 注册</button>
      </article>
    `;
    document.getElementById("utilityLogin").addEventListener("click", () => openAuthModal("login"));
    return;
  }
  let context = {};
  try {
    const data = await apiFetch(`/api/users/${encodeURIComponent(USER_ID)}/context`);
    context = data.context || {};
  } catch (error) {
    context = {};
  }
  const profile = context.profile || {};
  document.getElementById("taskSummary").textContent = "这里展示你的学习画像；尚未记录的字段会保持为空。";
  const preferredSubjects = (profile.preferred_subjects || []).join("、") || "未设置";
  const preferredTypes = (profile.preferred_resource_types || []).join("、") || "未设置";
  const constraintsText = Object.keys(profile.constraints || {}).length ? JSON.stringify(profile.constraints, null, 2) : "";
  panels.answer.innerHTML = `
    <section class="utility-page profile-page">
      <div class="utility-header">
        <h2>${escapeHtml(profile.display_name || currentUser?.username || "学习者")}</h2>
        <p>${escapeHtml(profile.goal || "还没有设置学习目标")}</p>
      </div>
      <form id="profileForm" class="profile-form">
        <label>
          <span>显示名称</span>
          <input id="profileDisplayName" type="text" value="${escapeHtml(profile.display_name || currentUser?.username || "")}" />
        </label>
        <label>
          <span>学习阶段</span>
          <select id="profileLearningStage">
            ${renderStageOptions(profile.learning_stage || "")}
          </select>
        </label>
        <label class="wide">
          <span>学习目标</span>
          <input id="profileGoal" type="text" placeholder="例如：三个月内入门机器学习算法" value="${escapeHtml(profile.goal || "")}" />
        </label>
        <label class="wide">
          <span>偏好方向</span>
          <input id="profileSubjects" type="text" placeholder="用逗号分隔，例如：人工智能, 机器学习, 算法" value="${escapeHtml(preferredSubjects === "未设置" ? "" : preferredSubjects)}" />
        </label>
        <label class="wide">
          <span>偏好资源类型</span>
          <input id="profileResourceTypes" type="text" placeholder="用逗号分隔，例如：课程, 练习, 章节" value="${escapeHtml(preferredTypes === "未设置" ? "" : preferredTypes)}" />
        </label>
        <label class="wide">
          <span>学习约束</span>
          <textarea id="profileConstraints" placeholder='可选，JSON 格式，例如 {"每周学习时间":"5小时"}'>${escapeHtml(constraintsText)}</textarea>
        </label>
        <label class="wide">
          <span>记忆摘要</span>
          <textarea id="profileMemorySummary" placeholder="可选，写给 Agent 的长期偏好摘要">${escapeHtml(profile.memory_summary || "")}</textarea>
        </label>
        <div class="profile-form-actions">
          <button type="submit">保存学习画像</button>
          <span id="profileSaveStatus"></span>
        </div>
      </form>
    </section>
  `;
  document.getElementById("profileForm").addEventListener("submit", saveProfileForm);
}

function renderStageOptions(current) {
  const options = [
    ["", "未设置"],
    ["beginner", "零基础 / 入门"],
    ["intermediate", "有一定基础"],
    ["advanced", "进阶提升"],
  ];
  return options
    .map(([value, label]) => `<option value="${value}" ${value === current ? "selected" : ""}>${label}</option>`)
    .join("");
}

async function saveProfileForm(event) {
  event.preventDefault();
  const status = document.getElementById("profileSaveStatus");
  let constraints = {};
  const constraintsText = document.getElementById("profileConstraints").value.trim();
  if (constraintsText) {
    try {
      constraints = JSON.parse(constraintsText);
    } catch (error) {
      status.textContent = "学习约束需要是合法 JSON";
      return;
    }
  }
  status.textContent = "保存中";
  const payload = {
    display_name: document.getElementById("profileDisplayName").value.trim() || null,
    learning_stage: document.getElementById("profileLearningStage").value || null,
    goal: document.getElementById("profileGoal").value.trim() || null,
    preferred_subjects: splitInputList(document.getElementById("profileSubjects").value),
    preferred_resource_types: splitInputList(document.getElementById("profileResourceTypes").value),
    constraints,
    memory_summary: document.getElementById("profileMemorySummary").value.trim() || null,
  };
  try {
    const data = await apiFetch(`/api/users/${encodeURIComponent(USER_ID)}/profile`, {
      method: "PUT",
      body: JSON.stringify(payload),
    });
    if (currentUser) {
      currentUser.display_name = data.profile.display_name || currentUser.display_name;
      currentUser.learning_stage = data.profile.learning_stage || currentUser.learning_stage;
      currentUser.goal = data.profile.goal || currentUser.goal;
      localStorage.setItem("mooc_agent_user", JSON.stringify(currentUser));
      updateUserView();
    }
    status.textContent = "已保存";
    document.getElementById("taskSummary").textContent = "学习画像已更新，后续推荐和规划会使用这些信息。";
  } catch (error) {
    status.textContent = `保存失败：${error.message}`;
  }
}

function splitInputList(value) {
  return String(value || "")
    .split(/[，,、|]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

async function renderNotesPage() {
  if (!requireLogin()) return;
  let notes = [];
  try {
    const data = await apiFetch(`/api/users/${encodeURIComponent(USER_ID)}/notes`);
    notes = data.notes || [];
  } catch (error) {
    notes = [];
  }
  document.getElementById("taskSummary").textContent = "笔记本是你的学习文档区，可以整理问题、结论和复盘。";
  if (!noteDraft.id && notes.length && !noteDraft.isNew) {
    noteDraft = {
      id: notes[0].id,
      title: cleanNoteText(notes[0].title || "学习笔记"),
      content: cleanNoteText(notes[0].content || ""),
      isNew: false,
    };
  }
  const editorHtml = noteContentToEditorHtml(noteDraft.content);
  panels.answer.innerHTML = `
    <section class="notes-workbench">
      <aside class="notes-list">
        <div class="notes-list-head">
          <h2>笔记</h2>
          <button id="newNoteButton" type="button">新建</button>
        </div>
        <div class="notes-items">
          ${notes.map((note) => renderNoteListItem(note)).join("") || '<p class="notes-empty">暂无笔记</p>'}
        </div>
      </aside>
      <article class="note-editor">
        <input id="noteTitleInput" class="note-title-input" type="text" placeholder="无标题笔记" value="${escapeHtml(noteDraft.title)}" />
        <div class="note-toolbar" aria-label="笔记格式工具栏">
          <button type="button" data-note-command="h2">二级标题</button>
          <button type="button" data-note-command="bold"><b>B</b></button>
          <button type="button" data-note-command="highlight">高亮</button>
          <button type="button" data-note-command="ul">列表</button>
          <button type="button" data-note-command="clear">清除格式</button>
        </div>
        <div id="noteContentInput" class="note-document" contenteditable="true" data-placeholder="像写文档一样记录今天的学习内容、问题、结论和复盘。">${editorHtml}</div>
        <div class="note-editor-actions">
          <button id="saveNoteButton" type="button">保存笔记</button>
          <button id="deleteCurrentNoteButton" type="button" ${noteDraft.id ? "" : "disabled"}>删除当前笔记</button>
          <span id="noteSaveStatus"></span>
        </div>
      </article>
    </section>
  `;
  document.getElementById("newNoteButton").addEventListener("click", () => {
    noteDraft = { id: "", title: "", content: "", isNew: true };
    renderNotesPage();
  });
  panels.answer.querySelectorAll("[data-open-note]").forEach((button) => {
    button.addEventListener("click", () => {
      const note = notes.find((item) => item.id === button.dataset.openNote);
      if (!note) return;
      noteDraft = {
        id: note.id,
        title: cleanNoteText(note.title || "学习笔记"),
        content: cleanNoteText(note.content || ""),
        isNew: false,
      };
      renderNotesPage();
    });
  });
  panels.answer.querySelectorAll("[data-note-command]").forEach((button) => {
    button.addEventListener("click", () => executeNoteCommand(button.dataset.noteCommand));
  });
  document.getElementById("saveNoteButton").addEventListener("click", async () => {
    const title = document.getElementById("noteTitleInput").value.trim();
    const content = sanitizeNoteHtml(document.getElementById("noteContentInput").innerHTML);
    const textContent = stripHtml(content).trim();
    const status = document.getElementById("noteSaveStatus");
    if (!textContent) {
      status.textContent = "正文为空，未保存";
      return;
    }
    status.textContent = "保存中";
    const payload = {
      content,
      title: title || textContent.slice(0, 24),
      linked_resource_id: state.selectedId || null,
    };
    const saved = noteDraft.id
      ? await apiFetch(`/api/users/${encodeURIComponent(USER_ID)}/notes/${encodeURIComponent(noteDraft.id)}`, {
          method: "PUT",
          body: JSON.stringify(payload),
        })
      : await apiFetch("/api/users/notes", {
          method: "POST",
          body: JSON.stringify({ user_id: USER_ID, ...payload }),
        });
    noteDraft = {
      id: saved.note.id,
      title: cleanNoteText(saved.note.title || ""),
      content: cleanNoteText(saved.note.content || ""),
      isNew: false,
    };
    status.textContent = "已保存";
    await renderNotesPage();
  });
  document.getElementById("deleteCurrentNoteButton").addEventListener("click", async () => {
    if (!noteDraft.id) return;
    await apiFetch(`/api/users/${encodeURIComponent(USER_ID)}/notes/${encodeURIComponent(noteDraft.id)}`, {
      method: "DELETE",
    });
    noteDraft = { id: "", title: "", content: "", isNew: false };
    await renderNotesPage();
  });
}

function renderNoteListItem(note) {
  const title = cleanNoteText(note.title || "学习笔记");
  const content = cleanNoteText(note.content || "");
  const active = note.id === noteDraft.id ? "active" : "";
  return `
    <button type="button" class="note-list-item ${active}" data-open-note="${escapeHtml(note.id)}">
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(stripHtml(content).slice(0, 42) || "空白笔记")}</span>
    </button>
  `;
}

function executeNoteCommand(command) {
  const editor = document.getElementById("noteContentInput");
  if (!editor) return;
  editor.focus();
  if (command === "h2") {
    document.execCommand("formatBlock", false, "h2");
  } else if (command === "bold") {
    document.execCommand("bold", false, null);
  } else if (command === "highlight") {
    document.execCommand("backColor", false, "#fff3a3");
  } else if (command === "ul") {
    document.execCommand("insertUnorderedList", false, null);
  } else if (command === "clear") {
    document.execCommand("removeFormat", false, null);
    document.execCommand("formatBlock", false, "p");
  }
}

function noteContentToEditorHtml(content) {
  const value = cleanNoteText(content || "");
  if (!value) return "";
  if (/<(h2|p|div|ul|ol|li|strong|b|mark|span|br)\b/i.test(value)) {
    return sanitizeNoteHtml(value);
  }
  return escapeHtml(value)
    .split(/\n{2,}/)
    .map((block) => `<p>${block.replace(/\n/g, "<br>")}</p>`)
    .join("");
}

function sanitizeNoteHtml(html) {
  const template = document.createElement("template");
  template.innerHTML = String(html || "");
  const allowedTags = new Set(["H2", "P", "DIV", "BR", "UL", "OL", "LI", "STRONG", "B", "EM", "I", "MARK", "SPAN"]);
  const walker = document.createTreeWalker(template.content, NodeFilter.SHOW_ELEMENT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach((node) => {
    if (!allowedTags.has(node.tagName)) {
      node.replaceWith(document.createTextNode(node.textContent || ""));
      return;
    }
    [...node.attributes].forEach((attr) => {
      const isHighlight =
        node.tagName === "SPAN" &&
        attr.name === "style" &&
        /background-color:\s*(rgb\(255,\s*243,\s*163\)|#fff3a3|yellow)/i.test(attr.value);
      if (!isHighlight) node.removeAttribute(attr.name);
    });
    if (node.tagName === "SPAN" && node.getAttribute("style")) {
      node.setAttribute("style", "background-color: #fff3a3;");
    }
  });
  return template.innerHTML.trim();
}

function stripHtml(html) {
  const template = document.createElement("template");
  template.innerHTML = String(html || "");
  return (template.content.textContent || "").replace(/\s+/g, " ").trim();
}

function cleanNoteText(text) {
  const value = String(text || "");
  if (/^\?+$/.test(value.replace(/\s/g, "")) || /\?{2,}/.test(value)) {
    return "内容显示异常，建议删除后重新保存";
  }
  return value;
}

function openAuthModal(mode) {
  let modal = document.getElementById("authModal");
  if (!modal) {
    modal = document.createElement("div");
    modal.id = "authModal";
    modal.className = "auth-modal";
    document.body.appendChild(modal);
  }
  if (mode === "profile" && currentUser) {
    modal.innerHTML = `
      <div class="auth-card">
        <button class="auth-close" type="button">×</button>
        <h2>当前用户</h2>
        <p>${escapeHtml(currentUser.display_name || currentUser.username)}，已登录。</p>
        <button id="logoutButton" type="button">退出登录</button>
      </div>
    `;
  } else {
    modal.innerHTML = `
      <div class="auth-card">
        <button class="auth-close" type="button">×</button>
        <h2>${mode === "register" ? "注册账号" : "登录账号"}</h2>
        <input id="authUsername" type="text" placeholder="用户名" />
        <input id="authPassword" type="password" placeholder="密码，至少 6 位" />
        ${mode === "register" ? '<input id="authDisplayName" type="text" placeholder="显示名称，可选" />' : ""}
        <p id="authMessage"></p>
        <button id="authSubmit" type="button">${mode === "register" ? "注册并登录" : "登录"}</button>
        <button id="authSwitch" type="button">${mode === "register" ? "已有账号，去登录" : "没有账号，去注册"}</button>
      </div>
    `;
  }
  modal.classList.add("open");
  modal.querySelector(".auth-close").addEventListener("click", () => modal.classList.remove("open"));
  modal.querySelector("#authSwitch")?.addEventListener("click", () => openAuthModal(mode === "register" ? "login" : "register"));
  modal.querySelector("#logoutButton")?.addEventListener("click", () => {
    currentUser = null;
    USER_ID = "";
    localStorage.removeItem("mooc_agent_user");
    updateUserView();
    modal.classList.remove("open");
  });
  modal.querySelector("#authSubmit")?.addEventListener("click", async () => {
    const username = document.getElementById("authUsername").value.trim();
    const password = document.getElementById("authPassword").value;
    const displayName = document.getElementById("authDisplayName")?.value.trim();
    const message = document.getElementById("authMessage");
    try {
      const endpoint = mode === "register" ? "/api/auth/register" : "/api/auth/login";
      const payload = mode === "register" ? { username, password, display_name: displayName } : { username, password };
      const user = await apiFetch(endpoint, { method: "POST", body: JSON.stringify(payload) });
      currentUser = user;
      USER_ID = user.user_id;
      localStorage.setItem("mooc_agent_user", JSON.stringify(user));
      updateUserView();
      await refreshSavedResources().catch(() => {});
      await refreshAgentSettings().catch(() => {});
      modal.classList.remove("open");
    } catch (error) {
      message.textContent = error.message.includes("already") ? "用户名已存在" : "登录或注册失败，请检查输入。";
    }
  });
}

function updateUserView() {
  const name = currentUser?.display_name || currentUser?.username || "未登录";
  document.querySelector(".profile-chip strong").textContent = name;
  document.querySelector(".profile-chip small").textContent = currentUser ? "已登录 · MOOC 学习" : "点击登录 / 注册";
  document.querySelector(".avatar").textContent = name.slice(0, 1);
}

function setBusy(isBusy) {
  const button = document.querySelector(".chat-composer button");
  button.disabled = isBusy;
  button.textContent = isBusy ? "处理中" : "发送";
}

bindEvents();
updateUserView();
if (USER_ID) {
  refreshSavedResources().catch(() => {});
  refreshAgentSettings().catch(() => {});
}
loadResources(state.currentQuery);
