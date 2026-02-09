(() => {
  const BASE_URL = "";
  const ENDPOINT_STREAM = "/api/chat/stream";
  const ENDPOINT_JSON = "/api/chat";

  // ====== ELEMENTS ======
  const main = document.getElementById("cw-main");
  const form = document.getElementById("cw-form");
  const input = document.getElementById("cw-input");
  const sendBtn = document.getElementById("cw-send");
  const loader = document.getElementById("cw-loader");
  const clearBtn = document.getElementById("cw-clear");
  const statusEl = document.getElementById("cw-status");
  const apiEl = document.getElementById("cw-api");

  let isSending = false;

  function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", !!isError);
  }

  function scrollToBottom() {
    main.scrollTop = main.scrollHeight;
  }

  function escapeHtml(str) {
    return str.replace(/[&<>"']/g, (m) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
    }[m]));
  }

  function createMessage(role, text) {
    const wrap = document.createElement("div");
    wrap.className = `cw-msg ${role}`;

    const bubble = document.createElement("div");
    bubble.className = "cw-bubble";

    const roleEl = document.createElement("div");
    roleEl.className = "cw-role";
    roleEl.textContent = role === "user" ? "You" : "Syezain AI";

    const content = document.createElement("div");
    content.className = "cw-content";
    content.innerHTML = escapeHtml(text || "");

    bubble.appendChild(roleEl);
    bubble.appendChild(content);
    wrap.appendChild(bubble);

    main.appendChild(wrap);
    scrollToBottom();

    return { wrap, bubble, content };
  }

  function setSending(on) {
    isSending = on;
    sendBtn.disabled = on;
    sendBtn.classList.toggle("loading", on);
    loader.style.display = on ? "inline-block" : "none";
    input.disabled = on;
  }

  // Auto-resize textarea
  function resizeTextarea() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  }
  input.addEventListener("input", resizeTextarea);

  // Enter to send, Shift+Enter new line
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  clearBtn.addEventListener("click", () => {
    // remove all messages except the hint (first block)
    const keep = main.querySelector(".cw-hint");
    main.innerHTML = "";
    if (keep) main.appendChild(keep);
    setStatus("Cleared");
  });

  // ====== SSE (POST) parser ======
  // Server gửi dạng:
  // event: token
  // data: {"delta":"..."}
  //
  // (blank line)
  function parseSseChunk(buffer) {
    const events = [];
    const parts = buffer.split("\n\n");
    const tail = parts.pop(); // phần chưa đủ

    for (const part of parts) {
      const lines = part.split("\n");
      let eventName = "message";
      let dataStr = "";

      for (const line of lines) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        if (line.startsWith("data:")) dataStr += line.slice(5).trim();
      }

      if (!dataStr) continue;

      try {
        events.push({ event: eventName, data: JSON.parse(dataStr) });
      } catch {
        // ignore parse errors
      }
    }

    return { events, tail };
  }

  async function chatStream(question, onToken) {
    apiEl.textContent = ENDPOINT_STREAM;
    setStatus("Streaming…");

    const res = await fetch(BASE_URL + ENDPOINT_STREAM, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });

    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => "");
      throw new Error(`Stream HTTP ${res.status}: ${text || res.statusText}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let full = "";
    let donePayload = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parsed = parseSseChunk(buffer);
      buffer = parsed.tail;

      for (const ev of parsed.events) {
        if (ev.event === "token") {
          const delta = (ev.data && ev.data.delta) ? ev.data.delta : "";
          if (delta) {
            full += delta;
            onToken(delta);
          }
        } else if (ev.event === "done") {
          donePayload = ev.data || null;
        } else if (ev.event === "error") {
          const msg = ev.data?.message || "Unknown error";
          throw new Error(msg);
        }
      }
    }

    return { answer: full, done: donePayload };
  }

  async function chatJson(question) {
    apiEl.textContent = ENDPOINT_JSON;
    setStatus("Waiting full JSON…");

    const res = await fetch(BASE_URL + ENDPOINT_JSON, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });

    const data = await res.json().catch(() => null);
    if (!res.ok) {
      throw new Error(data?.detail || `HTTP ${res.status}`);
    }
    return data;
  }

  // ====== Submit handler ======
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (isSending) return;

    const question = input.value.trim();
    if (!question) return;

    // UI: add user message
    createMessage("user", question);
    input.value = "";
    resizeTextarea();

    // UI: create AI message bubble (empty, then stream append)
    const aiMsg = createMessage("ai", "");

    setSending(true);

    try {
      // Try streaming first
      let lastFlush = 0;
      await chatStream(question, (delta) => {
        // append delta text
        aiMsg.content.textContent += delta;

        // throttle scroll a bit
        const now = Date.now();
        if (now - lastFlush > 50) {
          scrollToBottom();
          lastFlush = now;
        }
      });

      setStatus("Done");
    } catch (err) {
      // Fallback: JSON
      setStatus("Stream failed → fallback JSON…", true);

      try {
        const data = await chatJson(question);
        aiMsg.content.textContent = data?.answer || "(no answer)";
        setStatus("Done (JSON fallback)");
      } catch (err2) {
        aiMsg.content.textContent = `❌ Error: ${err2?.message || String(err2)}`;
        setStatus("Error", true);
      }
    } finally {
      setSending(false);
      scrollToBottom();
    }
  });

  // Initial status
  setStatus("Idle");
  resizeTextarea();
})();
