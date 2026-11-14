const $ = id => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const INTERVAL_MS = 250;           // minimum gap between requests
let running = false, busy = false, faces = [], frameSize = null;

function say(text, cls = "") { $("msg").textContent = text; $("msg").className = "msg " + cls; }

$("startBtn").onclick = async () => {
  try {
    await startCamera(video);
  } catch (e) { return say(e.message, "err"); }
  $("placeholder").classList.add("hidden");
  $("startBtn").classList.add("hidden");
  $("stopBtn").classList.remove("hidden");
  say("Recognition running…");
  running = true;
  loop();
  requestAnimationFrame(draw);
};

$("stopBtn").onclick = () => {
  running = false; faces = [];
  stopCamera(video);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("placeholder").classList.remove("hidden");
  $("stopBtn").classList.add("hidden");
  $("startBtn").classList.remove("hidden");
  renderList([]);
  say("Recognition stopped.");
};

async function loop() {
  while (running) {
    const t0 = performance.now();
    if (video.videoWidth && !busy) {
      busy = true;
      const r = await api("/api/recognize", { image: grabFrame(video, 0.7) });
      busy = false;
      if (!running) break;
      if (r.ok) {
        faces = r.data.faces; frameSize = r.data.frame;
        renderList(faces); say("Recognition running…");
      } else say(r.data.error || MSG.noServer, "err");
    }
    const wait = Math.max(30, INTERVAL_MS - (performance.now() - t0));
    await new Promise(res => setTimeout(res, wait));
  }
}

/* Map video-pixel coordinates to canvas coordinates, accounting for
   object-fit:contain letterboxing and the mirrored <video>. */
function draw() {
  if (!running) return;
  const cw = canvas.clientWidth, ch = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
  if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) { canvas.width = cw * dpr; canvas.height = ch * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  if (video.videoWidth && faces.length) {
    const vw = video.videoWidth, vh = video.videoHeight;
    const scale = Math.min(cw / vw, ch / vh);
    const offX = (cw - vw * scale) / 2, offY = (ch - vh * scale) / 2;
    ctx.lineWidth = 3; ctx.font = "600 15px system-ui, sans-serif"; ctx.textBaseline = "top";
    for (const f of faces) {
      const [x, y, w, h] = f.box;
      const bx = cw - (offX + (x + w) * scale);     // mirror horizontally
      const by = offY + y * scale, bw = w * scale, bh = h * scale;
      const color = f.recognized ? "#22c55e" : "#ef4444";
      ctx.strokeStyle = color; ctx.strokeRect(bx, by, bw, bh);
      const label = f.recognized ? `${f.name} · ${f.role}` : "Unknown";
      const tw = ctx.measureText(label).width + 12, ty = by - 26 < 0 ? by + bh : by - 26;
      ctx.fillStyle = color; ctx.fillRect(bx, ty, tw, 24);
      ctx.fillStyle = "#fff"; ctx.fillText(label, bx + 6, ty + 4);
    }
  }
  requestAnimationFrame(draw);
}

function renderList(list) {
  const ul = $("faceList");
  $("faceCount").textContent = list.length ? `(${list.length})` : "";
  if (!list.length) { ul.innerHTML = '<li class="msg">No faces detected.</li>'; return; }
  ul.innerHTML = "";
  list.forEach(f => {
    const li = document.createElement("li");
    li.className = "face-item " + (f.recognized ? "known" : "unknown");
    const dist = f.distance == null ? "" : ` · distance ${f.distance.toFixed(2)}`;
    const main = document.createElement("div");
    const b = document.createElement("b"); b.textContent = f.recognized ? f.name : "Unknown";
    const s = document.createElement("small");
    s.textContent = f.recognized ? `${f.role}${dist}` : `This person is not registered${dist}`;
    main.append(b, s); li.append(main); ul.append(li);
  });
}
