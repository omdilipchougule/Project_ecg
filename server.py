"""server.py - Raspberry Pi 5 FastAPI + WebSocket + UDP Telemetry Server for ECG Arrhythmia Pipeline."""
import os, sys, time, json, socket, asyncio, threading, smtplib
import urllib.request
from email.message import EmailMessage
from collections import deque
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import ecg_pipeline as ep
from ecg_features import FS
from arrhythmia_rules import CFG

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'ecg_improved_v3.pkl')
META_DIR   = os.path.join(os.path.dirname(__file__), 'metadata')
UDP_PORT   = 5005
CHUNK_S    = 0.5
CHUNK_N    = int(CHUNK_S * FS)  # 125 samples per 0.5s hop

# --- TELEMEDICINE TRIAGE ALERT CONFIGURATION ---
# You can set these via environment variables or paste your credentials directly below
DISCORD_WEBHOOK_URL = os.getenv("ECG_DISCORD_WEBHOOK", "https://discord.com/api/webhooks/1565770713908908064/f2l4l9dU9JdOiQ2sSKPJOCmrNZa8FeRNLTTzsQc0IK5rYT7CRYppu8Vd2eunRWzGCqs4")
SMTP_SENDER_EMAIL   = os.getenv("ECG_SMTP_SENDER", "chouguleo918@gmail.com")
SMTP_APP_PASSWORD   = os.getenv("ECG_SMTP_PASSWORD", "mbygkigfqcgnshmd")
PHYSICIAN_EMAIL     = os.getenv("ECG_PHYSICIAN_EMAIL", "om2018chougule@gmail.com")
ALERT_COOLDOWN_SEC  = 60.0

app = FastAPI(title="Real-Time ECG Arrhythmia Pipeline")
bundle = ep.load_bundle(MODEL_PATH)

def build_synthetic_stream(mode='normal', seconds=120):
    """Fallback generator if .npy files are not present, or for artifact injection."""
    n = int(seconds * FS); t = np.arange(n) / FS
    rng = np.random.default_rng(42)
    sig = rng.normal(0, 0.025, n) + 0.15 * np.sin(2 * np.pi * 0.25 * t)
    g = lambda c, s, a: a * np.exp(-0.5 * ((t - c) / s) ** 2)
    pos, beat_idx = 0.8, 0
    while pos < seconds - 1.0:
        if mode == 'ectopy' and beat_idx % 6 == 4:
            # Premature wide-QRS PVC + compensatory pause
            pos -= 0.22
            sig += g(pos, 0.022, -1.35) + g(pos + 0.22, 0.05, -0.45)
            pos += 1.02
        else:
            sig += g(pos, 0.009, 1.1) + g(pos + 0.25, 0.04, 0.28) + g(pos - 0.16, 0.02, 0.12)
            pos += 0.80 + rng.normal(0, 0.015)
        beat_idx += 1
    if mode == 'artifact':
        sig += rng.choice([-2.8, 0.0, 2.8], size=n, p=[0.22, 0.56, 0.22]) + rng.normal(0, 0.7, n)
    return sig.astype(np.float32)

def load_demo_signals():
    norm_p = os.path.join(META_DIR, 'demo_normal.npy')
    ect_p  = os.path.join(META_DIR, 'demo_ectopy.npy')
    norm = np.load(norm_p) if os.path.exists(norm_p) else build_synthetic_stream('normal')
    ect  = np.load(ect_p)  if os.path.exists(ect_p)  else build_synthetic_stream('ectopy')
    art  = build_synthetic_stream('artifact', seconds=60)
    return {'DEMO_NORMAL': norm, 'DEMO_ECTOPY': ect, 'DEMO_ARTIFACT': art}

DEMOS = load_demo_signals()

# --- HRV & AUTONOMIC STRESS ENGINE ---
def derive_hrv_metrics(hr_bpm, nrmssd, nn_times_deque):
    """
    Computes RMSSD in milliseconds and autonomic nervous system (ANS) stress state.
    Uses exact conversion from detector's nRMSSD + HR, or falls back to NN timestamps.
    """
    rmssd_ms = 0.0
    if hr_bpm > 0.0 and nrmssd > 0.0:
        mean_rr_ms = 60000.0 / hr_bpm
        rmssd_ms = nrmssd * mean_rr_ms
    elif len(nn_times_deque) >= 4:
        rrs = np.diff(np.array(nn_times_deque, dtype=np.float64)) * 1000.0
        valid_rr = rrs[(rrs > 300.0) & (rrs < 2000.0)]
        if len(valid_rr) >= 2:
            rmssd_ms = float(np.sqrt(np.mean(np.diff(valid_rr) ** 2)))

    rmssd_ms = round(float(rmssd_ms), 1)
    if rmssd_ms <= 0.0:
        stress_state = "Calibrating NN..."
    elif rmssd_ms < 22.0:
        stress_state = "High Sympathetic Load"
    elif rmssd_ms < 42.0:
        stress_state = "Moderate Autonomic Tone"
    else:
        stress_state = "Optimal Vagal Tone"

    return rmssd_ms, stress_state

# --- ASYNCHRONOUS DUAL-CHANNEL TRIAGE DISPATCHER ---
def _send_alerts_worker(diagnosis, hr_bpm, burden_v, burden_s, rmssd_ms, stress_state, mode):
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    clean_diag = diagnosis.replace("_", " ").upper()

    # 1. Discord Rich Webhook
    if DISCORD_WEBHOOK_URL and "YOUR_DISCORD" not in DISCORD_WEBHOOK_URL:
        try:
            payload = {
                "username": "Clinical ECG Telemetry Node",
                "embeds": [{
                    "title": f"🚨 CRITICAL TRIAGE ALERT: {clean_diag}",
                    "color": 15548997,
                    "fields": [
                        {"name": "Heart Rate", "value": f"**{hr_bpm} BPM**", "inline": True},
                        {"name": "PVC / PAC Burden", "value": f"V: {burden_v}% | S: {burden_s}%", "inline": True},
                        {"name": "HRV (RMSSD)", "value": f"{rmssd_ms} ms ({stress_state})", "inline": True},
                        {"name": "Signal Source", "value": f"`{mode}`", "inline": True},
                        {"name": "Timestamp", "value": timestamp_str, "inline": True}
                    ],
                    "footer": {"text": "Raspberry Pi 5 • 4-Class LightGBM Arrhythmia Pipeline"}
                }]
            }
            req = urllib.request.Request(
                DISCORD_WEBHOOK_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "ECG-Monitor/1.0"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[Triage Warning] Discord webhook failed: {e}")

    # 2. Automated SMTP Email
    if SMTP_APP_PASSWORD and "YOUR_16_CHAR" not in SMTP_APP_PASSWORD:
        try:
            msg = EmailMessage()
            msg["Subject"] = f"[URGENT ECG ALERT] {clean_diag} Detected ({hr_bpm} BPM)"
            msg["From"] = SMTP_SENDER_EMAIL
            msg["To"] = PHYSICIAN_EMAIL
            msg.set_content(
                f"AUTOMATED TELEMEDICINE TRIAGE REPORT\n"
                f"====================================\n"
                f"Primary Diagnosis : {clean_diag}\n"
                f"Signal Source     : {mode}\n"
                f"Timestamp         : {timestamp_str}\n\n"
                f"WINDOW METRICS (30s Rolling Buffer):\n"
                f"- Heart Rate      : {hr_bpm} BPM\n"
                f"- Ventricular (V) : {burden_v}%\n"
                f"- Atrial (S)      : {burden_s}%\n"
                f"- HRV (RMSSD)     : {rmssd_ms} ms ({stress_state})\n"
                f"====================================\n"
                f"Action Required: Review live Raspberry Pi 5 telemetry dashboard immediately."
            )
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as smtp:
                smtp.login(SMTP_SENDER_EMAIL, SMTP_APP_PASSWORD)
                smtp.send_message(msg)
        except Exception as e:
            print(f"[Triage Warning] SMTP Email failed: {e}")


class PipelineState:
    def __init__(self):
        self.mode = 'DEMO_NORMAL'
        self.ptr = 0
        self.udp_queue = deque(maxlen=FS * 10)
        self.nn_times = deque(maxlen=40)
        self.det = ep.StreamingArrhythmiaDetector(bundle, fs=FS, buf_s=30, hop_s=0.5, margin_s=1.5)
        self.clients = set()
        self.last_alert_ts = 0.0
        self.alert_latch = False
        self.last_win = dict(
            primary_type='warming_up', types=[], hr_bpm=0.0,
            burden_V=0.0, burden_S=0.0, burden_Q=0.0, nrmssd=0.0,
            rmssd_ms=0.0, stress_state='Calibrating NN...',
            n_beats=0, alert_dispatched=False
        )

    def reset_detector(self, new_mode):
        self.mode = new_mode
        self.ptr = 0
        self.nn_times.clear()
        self.alert_latch = False
        self.det = ep.StreamingArrhythmiaDetector(bundle, fs=FS, buf_s=30, hop_s=0.5, margin_s=1.5)
        self.last_win = dict(
            primary_type='warming_up', types=[], hr_bpm=0.0,
            burden_V=0.0, burden_S=0.0, burden_Q=0.0, nrmssd=0.0,
            rmssd_ms=0.0, stress_state='Calibrating NN...',
            n_beats=0, alert_dispatched=False
        )

    def next_chunk(self):
        if self.mode == 'ESP32_LIVE':
            if len(self.udp_queue) >= CHUNK_N:
                return np.array([self.udp_queue.popleft() for _ in range(CHUNK_N)], dtype=np.float32)
            return np.zeros(CHUNK_N, dtype=np.float32)
        arr = DEMOS[self.mode]
        if self.ptr + CHUNK_N > len(arr):
            self.ptr = 0
        chunk = arr[self.ptr:self.ptr + CHUNK_N]
        self.ptr += CHUNK_N
        return chunk

state = PipelineState()

class ModeRequest(BaseModel):
    mode: str

@app.post("/api/mode")
async def set_mode(req: ModeRequest):
    if req.mode in ('DEMO_NORMAL', 'DEMO_ECTOPY', 'DEMO_ARTIFACT', 'ESP32_LIVE'):
        state.reset_detector(req.mode)
        return {"status": "ok", "mode": state.mode}
    return {"status": "error", "message": "Invalid mode"}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state.clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        state.clients.discard(ws)

async def udp_listener():
    """Non-blocking UDP socket listener for ESP32-S3 packets."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', UDP_PORT))
    sock.setblocking(False)
    loop = asyncio.get_event_loop()
    while True:
        try:
            data, _ = await loop.sock_recvfrom(sock, 4096)
            try:
                vals = [float(x) for x in data.decode('utf-8').strip().split(',') if x]
            except UnicodeDecodeError:
                vals = np.frombuffer(data, dtype=np.float32).tolist()
            state.udp_queue.extend(vals)
        except Exception:
            await asyncio.sleep(0.005)

async def inference_broadcaster():
    """Feeds 0.5s chunks into StreamingArrhythmiaDetector and broadcasts pipeline telemetry."""
    non_critical_states = {'warming_up', 'insufficient_data', 'no_arrhythmia_detected', 'poor_quality'}

    while True:
        t0 = time.perf_counter()
        chunk = state.next_chunk()
        res = state.det.push(chunk)
        comp_ms = (time.perf_counter() - t0) * 1000.0

        new_beats = []
        if res is not None:
            for b in res['new_beats']:
                pr = b['probs']
                t_val = round(float(b['t']), 2)
                lbl = b['label']
                if lbl == 'N':
                    state.nn_times.append(t_val)
                new_beats.append(dict(
                    t=t_val,
                    label=lbl,
                    pN=round(float(pr[0]), 3),
                    pS=round(float(pr[1]), 3),
                    pV=round(float(pr[2]), 3),
                    pQ=round(float(pr[3]), 3) if len(pr) > 3 else 0.0
                ))

            w = res['window']
            primary = w['primary_type']
            hr_bpm = 0.0 if np.isnan(w['hr_bpm']) else round(float(w['hr_bpm']), 1)
            burden_v = 0.0 if np.isnan(w['burden_V']) else round(float(w['burden_V']) * 100, 1)
            burden_s = 0.0 if np.isnan(w['burden_S']) else round(float(w['burden_S']) * 100, 1)
            burden_q = 0.0 if np.isnan(w.get('burden_Q', 0.0)) else round(float(w.get('burden_Q', 0.0)) * 100, 1)
            nrmssd = 0.0 if np.isnan(w.get('nrmssd', np.nan)) else round(float(w['nrmssd']), 3)

            rmssd_ms, stress_state = derive_hrv_metrics(hr_bpm, nrmssd, state.nn_times)

            # Evaluate Triage Alert Trigger (with 60s cooldown)
            now = time.time()
            if primary not in non_critical_states:
                if (now - state.last_alert_ts) >= ALERT_COOLDOWN_SEC:
                    state.last_alert_ts = now
                    state.alert_latch = True
                    threading.Thread(
                        target=_send_alerts_worker,
                        args=(primary, hr_bpm, burden_v, burden_s, rmssd_ms, stress_state, state.mode),
                        daemon=True
                    ).start()
            else:
                state.alert_latch = False

            state.last_win = dict(
                primary_type=primary,
                types=w['types'],
                hr_bpm=hr_bpm,
                burden_V=burden_v,
                burden_S=burden_s,
                burden_Q=burden_q,
                nrmssd=nrmssd,
                rmssd_ms=rmssd_ms,
                stress_state=stress_state,
                n_beats=int(w['n_beats']),
                alert_dispatched=state.alert_latch
            )

        payload = json.dumps(dict(
            mode=state.mode,
            compute_ms=round(comp_ms, 2),
            buf_fill_s=round(len(state.det.buf) / FS, 1),
            signal=[round(float(x), 4) for x in chunk[::2]], 
            new_beats=new_beats,
            window=state.last_win
        ))

        dead = set()
        for ws in state.clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        state.clients -= dead

        elapsed = time.perf_counter() - t0
        await asyncio.sleep(max(0.01, CHUNK_S - elapsed))

@app.on_event("startup")
async def startup_tasks():
    asyncio.create_task(udp_listener())
    asyncio.create_task(inference_broadcaster())

@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(os.path.dirname(__file__), "dashboard.html"), encoding="utf-8") as f:
        return f.read()