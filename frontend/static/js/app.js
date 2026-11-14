/* Shared helpers */
(function () {
  const saved = localStorage.getItem("theme");
  if (saved) document.documentElement.dataset.theme = saved;
  const btn = document.getElementById("themeBtn");
  if (btn) btn.onclick = () => {
    const dark = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() === "#0f1420";
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("theme", next);
  };
})();

const MSG = {
  camDenied: "Camera access is required. Please allow camera permission and try again.",
  noServer: "Unable to connect to recognition server.",
};

async function api(path, body) {
  try {
    const res = await fetch(path, {
      method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await res.json().catch(() => ({}));
    return { ok: res.ok, status: res.status, data };
  } catch (e) {
    return { ok: false, status: 0, data: { error: MSG.noServer } };
  }
}

async function startCamera(video) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
    throw new Error("This browser cannot access the camera (HTTPS or localhost is required).");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false });
    video.srcObject = stream;
    await video.play();
    return stream;
  } catch (e) {
    throw new Error(MSG.camDenied);
  }
}

function stopCamera(video) {
  const s = video.srcObject;
  if (s) s.getTracks().forEach(t => t.stop());
  video.srcObject = null;
}

/* Grab the current video frame as a JPEG data URL (un-mirrored, i.e. true camera pixels). */
const _grab = document.createElement("canvas");
function grabFrame(video, quality = 0.8) {
  _grab.width = video.videoWidth; _grab.height = video.videoHeight;
  _grab.getContext("2d").drawImage(video, 0, 0);
  return _grab.toDataURL("image/jpeg", quality);
}

async function initHome() {
  const set = (id, ok, text) => {
    document.getElementById("dot" + id).className = "dot " + (ok ? "ok" : "bad");
    document.getElementById("txt" + id).textContent = text;
  };
  const r = await api("/api/status");
  if (r.ok) {
    set("Engine", r.data.engine_ready, r.data.engine_ready ? "Ready" : "Unavailable");
    document.getElementById("txtUsers").textContent = r.data.registered_users;
  } else set("Engine", false, "Offline");
  const cam = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  set("Cam", cam, cam ? "Available" : "Not supported");
}
