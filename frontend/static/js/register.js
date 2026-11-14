const $ = id => document.getElementById(id);
const video = $("video"), canvas = $("overlay");
const sessionId = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random()));
let capturing = false, count = 0, busy = false, timer = null;

function say(text, cls = "") { $("msg").textContent = text; $("msg").className = "msg " + cls; }
function progress(n) {
  count = n;
  $("bar").style.width = (n / TARGET * 100) + "%";
  $("count").textContent = `${n}/${TARGET}`;
}

$("camBtn").onclick = async () => {
  try {
    await startCamera(video);
    $("placeholder").classList.add("hidden");
    $("guide").classList.remove("hidden");
    $("capBtn").disabled = false;
    $("camBtn").disabled = true;
    say("Camera ready. Enter name and role, then click Start Capture.");
  } catch (e) { say(e.message, "err"); }
};

$("capBtn").onclick = () => {
  if (capturing) return stopCapture("Capture paused.");
  const name = $("name").value.trim(), role = $("role").value.trim();
  if (!name) return say("Name cannot be empty.", "err");
  if (!role) return say("Role cannot be empty.", "err");
  capturing = true;
  $("capBtn").textContent = "Pause Capture";
  say("Look at the camera. Slowly turn your head a little between samples.");
  timer = setInterval(sendSample, 450);
};

function stopCapture(text) {
  capturing = false; clearInterval(timer);
  $("capBtn").textContent = "Start Capture";
  if (text) say(text);
}

async function sendSample() {
  if (busy || !capturing || !video.videoWidth) return;   // never queue requests
  busy = true;
  const r = await api("/api/register", {
    session_id: sessionId, name: $("name").value, role: $("role").value,
    image: grabFrame(video),
  });
  busy = false;
  if (!capturing) return;
  if (!r.ok) { stopCapture(); return say(r.data.error || MSG.noServer, "err"); }
  const d = r.data;
  if (typeof d.count === "number") progress(d.count);
  if (!d.accepted) return say(d.message, "err");
  say(d.message, "ok");
  if (d.count >= TARGET) finish();
}

async function finish() {
  stopCapture();
  say("Saving and training…");
  const r = await api("/api/register", {
    session_id: sessionId, name: $("name").value, role: $("role").value, finish: true });
  if (!r.ok) return say(r.data.error || "Registration could not be completed. Please try again.", "err");
  stopCamera(video);
  $("rName").textContent = r.data.name;
  $("rRole").textContent = r.data.role;
  $("rSamples").textContent = r.data.samples;
  $("formView").classList.add("hidden");
  $("doneView").classList.remove("hidden");
}
