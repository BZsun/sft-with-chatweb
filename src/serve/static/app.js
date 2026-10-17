// Medical Helper 前端逻辑
// - 对接 OpenAI 兼容 API（/v1/chat/completions，由 src/modelServer.py 同端口托管）
// - 多轮会话（localStorage）、SSE 流式渲染、停止、清空、示例问题

(function () {
    "use strict";

    const SYSTEM_PROMPT = "你是一名专业的中文医疗助手，请基于医学知识为患者提供准确、严谨的回答，并在必要时建议就医。";
    const HISTORY_KEY = "medical_helper_history_v2";
    const DEFAULT_MODEL = "Qwen3-8B"; // 兜底；正常从 /v1/models 拉取 .env 的 SERVED_MODEL_NAME

    // 内联 SVG 图标（与 style.css 配套）
    const ICON_CROSS = '<svg class="role-ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true"><path d="M8 3v10M3 8h10"/></svg>';
    const ICON_CLOCK = '<svg class="meta-ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" aria-hidden="true"><circle cx="8" cy="8" r="5.7"/><path d="M8 5.2V8l2 1.4"/></svg>';
    const ICON_PERSON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="8.2" r="3.4"/><path d="M5.5 19c1.2-3.2 3.6-4.8 6.5-4.8s5.3 1.6 6.5 4.8"/></svg>';
    const ICON_CHEVRON = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 6l4 4 4-4"/></svg>';

    // ---- DOM ----
    const $messages = document.getElementById("messages");
    const $input = document.getElementById("userInput");
    const $sendBtn = document.getElementById("sendBtn");
    const $stopBtn = document.getElementById("stopBtn");
    const $clearBtn = document.getElementById("clearBtn");
    const $welcome = document.getElementById("welcome");
    const $chatContainer = document.getElementById("chatContainer");

    // ---- 状态 ----
    let servedModel = DEFAULT_MODEL;
    let history = loadHistory();
    let abortController = null;
    let busy = false;

    // ---- 初始化 ----
    loadServedModel();
    renderHistory();

    $sendBtn.addEventListener("click", onSend);
    $stopBtn.addEventListener("click", onStop);
    $clearBtn.addEventListener("click", onClear);
    $input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onSend();
        }
    });
    $input.addEventListener("input", autoResize);

    document.querySelectorAll(".example-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            $input.value = btn.dataset.q;
            autoResize();
            $input.focus();
        });
    });

    // ---- 服务模型名 ----
    async function loadServedModel() {
        try {
            const res = await fetch("/v1/models");
            if (!res.ok) return;
            const data = await res.json();
            const id = data && Array.isArray(data.data) && data.data[0] && data.data[0].id;
            if (id) servedModel = id;
        } catch (e) {
            console.warn("拉取 /v1/models 失败，使用默认模型名:", e);
        }
    }

    // ---- 历史 ----
    function loadHistory() {
        try {
            const raw = localStorage.getItem(HISTORY_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch {
            return [];
        }
    }

    function saveHistory() {
        try {
            localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
        } catch (e) {
            console.warn("localStorage 写入失败:", e);
        }
    }

    // ---- 渲染 ----
    function renderHistory() {
        $messages.innerHTML = "";
        if (history.length === 0) {
            $welcome.style.display = "block";
        } else {
            $welcome.style.display = "none";
            history.forEach((m) => appendMessageDOM(m.role, m.content, m.meta || null, m.reasoning || null));
        }
    }

    // 思考过程区块；opts: { text, collapsed, streaming, onToggle }
    function buildThinkingEl(opts) {
        opts = opts || {};
        const thinking = document.createElement("div");
        thinking.className = "thinking" + (opts.collapsed ? " collapsed" : "");
        const header = document.createElement("button");
        header.type = "button";
        header.className = "thinking-header";
        header.innerHTML = "<span>思考过程</span>" + ICON_CHEVRON;
        const body = document.createElement("div");
        body.className = "thinking-body" + (opts.streaming ? " streaming" : "");
        body.textContent = opts.text || "";
        header.addEventListener("click", () => {
            thinking.classList.toggle("collapsed");
            if (opts.onToggle) {
                opts.onToggle(thinking.classList.contains("collapsed"));
            }
        });
        thinking.appendChild(header);
        thinking.appendChild(body);
        return { thinking, body };
    }

    function appendMessageDOM(role, content, meta, reasoning) {
        const li = document.createElement("li");
        li.className = `message ${role}`;

        // 圆形头像：用户（人形剪影）固定左侧，医疗助手（柴犬图）固定右侧（CSS row-reverse 镜像）
        const avatar = document.createElement("div");
        avatar.className = "avatar";
        if (role === "user") {
            avatar.classList.add("avatar-user");
            avatar.innerHTML = ICON_PERSON;
        } else {
            avatar.classList.add("avatar-ai");
            avatar.innerHTML = '<img src="/static/avatar.jpg" alt="医疗助手" onerror="this.style.display=\'none\'">';
        }

        const head = document.createElement("div");
        head.className = "msg-head";
        head.innerHTML = role === "user"
            ? "<span class=\"msg-role\">用户</span>"
            : `<span class="msg-role msg-role-ai">${ICON_CROSS}医疗助手</span>`;

        const bubbleWrap = document.createElement("div");
        bubbleWrap.className = "msg-body";

        const bubble = document.createElement("div");
        bubble.className = "bubble";
        if (role === "assistant" && window.marked) {
            bubble.innerHTML = window.marked.parse(content || "");
        } else {
            bubble.textContent = content;
        }

        bubbleWrap.appendChild(head);

        // 历史里的思考内容默认折叠展示
        if (role === "assistant" && reasoning) {
            const { thinking } = buildThinkingEl({ text: reasoning, collapsed: true });
            bubbleWrap.appendChild(thinking);
        }

        bubbleWrap.appendChild(bubble);

        if (meta) {
            bubbleWrap.appendChild(buildMetaEl(meta));
        }

        li.appendChild(avatar);
        li.appendChild(bubbleWrap);
        $messages.appendChild(li);
        scrollToBottom();
        return { li, bubble, bubbleWrap };
    }

    function buildMetaEl(meta) {
        const metaEl = document.createElement("div");
        metaEl.className = "meta";
        const parts = [];
        if (meta.elapsed_ms != null) {
            parts.push(`<span class="elapsed">${ICON_CLOCK}${(meta.elapsed_ms / 1000).toFixed(2)}s</span>`);
        }
        if (meta.new_tokens != null) {
            parts.push(`<span>${meta.new_tokens} tokens</span>`);
        }
        metaEl.innerHTML = parts.join('<span class="meta-dot">·</span>');
        return metaEl;
    }

    function scrollToBottom() {
        $chatContainer.scrollTop = $chatContainer.scrollHeight;
    }

    function autoResize() {
        $input.style.height = "auto";
        $input.style.height = Math.min($input.scrollHeight, 160) + "px";
    }

    // ---- 发送（OpenAI 兼容流式接口） ----
    async function onSend() {
        if (busy) return;

        const text = $input.value.trim();
        if (!text) return;

        $welcome.style.display = "none";

        history.push({ role: "user", content: text });
        appendMessageDOM("user", text, null);
        saveHistory();

        const { bubble, bubbleWrap } = appendMessageDOM("assistant", "", null);
        bubble.innerHTML = '<span class="typing-dots"></span>';
        bubble.classList.add("streaming");

        $input.value = "";
        autoResize();

        const messagesForApi = [{ role: "system", content: SYSTEM_PROMPT }, ...history];

        busy = true;
        $sendBtn.disabled = true;
        $stopBtn.disabled = false;

        abortController = new AbortController();
        const t0 = performance.now();
        let accumulated = "";
        let reasoningAccumulated = "";
        let thinkingEl = null;
        let thinkingBody = null;
        let thinkingAutoCollapsed = false;
        let thinkingManualToggled = false;
        let meta = null;
        let streamDone = false;

        // 首个思考增量到达时创建思考块（展开态 + 流式光标），并撤掉气泡的打字点
        function ensureThinkingBlock() {
            if (thinkingEl) return;
            const built = buildThinkingEl({
                streaming: true,
                onToggle: (collapsed) => {
                    thinkingManualToggled = true;
                    if (collapsed) {
                        thinkingBody.classList.remove("streaming");
                    } else if (!thinkingAutoCollapsed) {
                        thinkingBody.classList.add("streaming");
                    }
                },
            });
            thinkingEl = built.thinking;
            thinkingBody = built.body;
            bubbleWrap.insertBefore(thinkingEl, bubble);
            bubble.classList.remove("streaming");
            bubble.innerHTML = "";
        }

        try {
            const res = await fetch("/v1/chat/completions", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: servedModel,
                    messages: messagesForApi,
                    stream: true,
                    stream_options: { include_usage: true },
                }),
                signal: abortController.signal,
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error((err && err.error && err.error.message) || `HTTP ${res.status}`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (!streamDone) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                const events = buffer.split("\n\n");
                buffer = events.pop() || "";

                for (const ev of events) {
                    const line = ev.trim();
                    if (!line.startsWith("data:")) continue;
                    const payload = line.slice(5).trim();
                    if (!payload) continue;
                    if (payload === "[DONE]") {
                        streamDone = true;
                        break;
                    }
                    let obj;
                    try {
                        obj = JSON.parse(payload);
                    } catch (parseErr) {
                        console.warn("SSE 解析失败:", parseErr, payload);
                        continue;
                    }
                    if (obj.error) {
                        throw new Error(obj.error.message || "服务端推理失败");
                    }
                    if (obj.usage) {
                        meta = {
                            elapsed_ms: performance.now() - t0,
                            new_tokens: obj.usage.completion_tokens,
                        };
                    }
                    const choice = obj.choices && obj.choices[0];
                    const delta = choice && choice.delta;
                    // 思考内容：浅灰小字区块实时流式展示
                    if (delta && typeof delta.reasoning_content === "string" && delta.reasoning_content) {
                        ensureThinkingBlock();
                        reasoningAccumulated += delta.reasoning_content;
                        thinkingBody.textContent = reasoningAccumulated;
                        scrollToBottom();
                    }
                    // 正文：开始输出正文时自动折叠思考块
                    if (delta && typeof delta.content === "string" && delta.content) {
                        if (thinkingEl && !thinkingAutoCollapsed) {
                            thinkingAutoCollapsed = true;
                            thinkingBody.classList.remove("streaming");
                            if (!thinkingManualToggled) {
                                thinkingEl.classList.add("collapsed");
                            }
                        }
                        bubble.classList.add("streaming");
                        accumulated += delta.content;
                        bubble.innerHTML = window.marked
                            ? window.marked.parse(accumulated)
                            : accumulated;
                        scrollToBottom();
                    }
                }
            }
        } catch (err) {
            if (err.name === "AbortError") {
                accumulated += "\n\n[已停止生成]";
            } else {
                accumulated += `\n\n[错误: ${err.message}]`;
            }
            bubble.innerHTML = window.marked
                ? window.marked.parse(accumulated)
                : accumulated;
        } finally {
            if (thinkingEl) {
                thinkingBody.classList.remove("streaming");
                if (!thinkingManualToggled && !thinkingAutoCollapsed) {
                    // 没有正文输出（如中途停止）时也收起，保持界面紧凑
                    thinkingEl.classList.add("collapsed");
                }
            }
            if (!meta) {
                meta = { elapsed_ms: performance.now() - t0 };
            }
            const entry = { role: "assistant", content: accumulated, meta };
            if (reasoningAccumulated) {
                entry.reasoning = reasoningAccumulated;
            }
            history.push(entry);
            saveHistory();

            bubble.classList.remove("streaming");
            bubbleWrap.appendChild(buildMetaEl(meta));

            busy = false;
            abortController = null;
            $sendBtn.disabled = false;
            $stopBtn.disabled = true;
        }
    }

    function onStop() {
        if (abortController) {
            abortController.abort();
        }
    }

    function onClear() {
        if (!confirm("确定清空对话历史？")) return;
        history = [];
        saveHistory();
        renderHistory();
    }
})();
