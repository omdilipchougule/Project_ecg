"""server.py - Raspberry Pi 5 FastAPI + WebSocket + UDP Telemetry Server for ECG Arrhythmia Pipeline."""
import os, sys, io, uuid, time, json, socket, asyncio, threading, smtplib
from typing import Optional, List, Dict
import urllib.request
from email.message import EmailMessage
from collections import deque
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import ecg_pipeline as ep
import patient_db as pdb
from ecg_features import FS
from arrhythmia_rules import CFG

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'ecg_improved_v3.pkl')
META_DIR   = os.path.join(os.path.dirname(__file__), 'metadata')
UDP_PORT   = 5005
CHUNK_S    = 0.5
CHUNK_N    = int(CHUNK_S * FS)  # 125 samples per 0.5s hop

# --- TELEMEDICINE TRIAGE ALERT CONFIGURATION ---
DISCORD_WEBHOOK_URL = os.getenv("ECG_DISCORD_WEBHOOK", "https://discord.com/api/webhooks/1565770713908908064/f2l4l9dU9JdOiQ2sSKPJOCmrNZa8FeRNLTTzsQc0IK5rYT7CRYppu8Vd2eunRWzGCqs4")
SMTP_SENDER_EMAIL   = os.getenv("ECG_SMTP_SENDER", "chouguleo918@gmail.com")
SMTP_APP_PASSWORD   = os.getenv("ECG_SMTP_PASSWORD", "mbygkigfqcgnshmd")
PHYSICIAN_EMAIL     = os.getenv("ECG_PHYSICIAN_EMAIL", "om2018chougule@gmail.com")
ALERT_COOLDOWN_SEC  = 60.0

app = FastAPI(title="Clinical ECG Telemetry")
pdb.init_db()
bundle = ep.load_bundle(MODEL_PATH)

# --- REAL PATIENT CLINICAL ARRHYTHMIA CATALOG ---
PATIENTS_DIR = os.path.join(META_DIR, 'patients')
CATALOG_PATH = os.path.join(META_DIR, 'patients_catalog.json')

def load_patient_library():
    catalog = []
    arrays = {}
    if os.path.exists(CATALOG_PATH):
        try:
            with open(CATALOG_PATH, 'r') as f:
                catalog = json.load(f)
            for item in catalog:
                fpath = os.path.join(PATIENTS_DIR, item['file'])
                if os.path.exists(fpath):
                    arrays[item['id']] = np.load(fpath)
        except Exception as e:
            print(f"[Warning] Failed to load patient library: {e}")
    return catalog, arrays

PATIENTS_CATALOG, PATIENT_ARRAYS = load_patient_library()

def build_synthetic_stream(mode='normal', seconds=120):
    """Fallback generator if .npy files are not present, or for artifact injection."""
    n = int(seconds * FS); t = np.arange(n) / FS
    rng = np.random.default_rng(42)
    sig = rng.normal(0, 0.025, n) + 0.15 * np.sin(2 * np.pi * 0.25 * t)
    g = lambda c, s, a: a * np.exp(-0.5 * ((t - c) / s) ** 2)
    pos, beat_idx = 0.8, 0
    while pos < seconds - 1.0:
        if mode == 'ectopy' and beat_idx % 6 == 4:
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
    if 'p_frequent_pvc' in PATIENT_ARRAYS:
        ect = PATIENT_ARRAYS['p_frequent_pvc']
    elif os.path.exists(ect_p):
        ect = np.load(ect_p)
    else:
        ect = build_synthetic_stream('ectopy')
    art  = build_synthetic_stream('artifact', seconds=60)
    return {'DEMO_NORMAL': norm, 'DEMO_ECTOPY': ect, 'DEMO_ARTIFACT': art}

DEMOS = load_demo_signals()

# --- CLINICAL RATIONALE GENERATOR ---
CLINICAL_DESCRIPTIONS = {
    "ventricular_tachycardia": (
        "Run of >=3 consecutive premature ventricular contractions (Class V) detected. "
        "Indicates rapid ventricular depolarization originating below the AV node; immediate clinical review advised."
    ),
    "frequent_PVC": (
        "Ventricular ectopy burden (Class V) exceeded the 10% clinical threshold across the 30s window. "
        "Characterized by wide, bizarre QRS complexes lacking preceding P-waves with compensatory pauses."
    ),
    "isolated_PVC": (
        "Sporadic premature ventricular contractions (Class V) identified within otherwise regular sinus rhythm."
    ),
    "supraventricular_tachycardia": (
        "Elevated supraventricular ectopic burden (Class S) combined with rapid ventricular response. "
        "Suggests abnormal atrial or AV junctional automaticity/re-entry."
    ),
    "frequent_PAC": (
        "Premature atrial contraction burden (Class S) exceeded threshold across the 30s window."
    ),
    "tachycardia": (
        "Sustained ventricular rate > 100 BPM with predominantly normal QRS morphology (Sinus Tachycardia)."
    ),
    "bradycardia": (
        "Sustained ventricular rate < 60 BPM across the 30s evaluation window (Sinus Bradycardia)."
    ),
    "irregular_rhythm_suspected": (
        "Elevated normalized root-mean-square of successive RR differences (nRMSSD) in the absence of simple ectopy. "
        "Suggests irregular R-R cadence requiring evaluation for atrial fibrillation or sinus arrhythmia."
    )
}

# Human-readable display names for arrhythmia types (used in alerts/reports)
DIAGNOSIS_LABELS = {
    "ventricular_tachycardia":      "Ventricular Tachycardia",
    "frequent_PVC":                 "Frequent PVCs (Ventricular Ectopy)",
    "isolated_PVC":                 "Isolated PVC",
    "supraventricular_tachycardia": "Supraventricular Tachycardia (SVT)",
    "frequent_PAC":                 "Frequent PACs (Atrial Ectopy)",
    "tachycardia":                  "Sinus Tachycardia",
    "bradycardia":                  "Sinus Bradycardia",
    "irregular_rhythm_suspected":   "Irregular Rhythm — Possible A-Fib",
    "no_arrhythmia_detected":       "Normal Sinus Rhythm",
    "poor_quality":                 "Poor Signal Quality",
    "insufficient_data":            "Insufficient Data",
    "warming_up":                   "Calibrating",
}

def derive_hrv_metrics(hr_bpm, nrmssd, nn_times_deque):
    rmssd_ms = 0.0
    if hr_bpm > 0.0 and nrmssd > 0.0:
        rmssd_ms = nrmssd * (60000.0 / hr_bpm)
    elif len(nn_times_deque) >= 4:
        rrs = np.diff(np.array(nn_times_deque, dtype=np.float64)) * 1000.0
        valid_rr = rrs[(rrs > 300.0) & (rrs < 2000.0)]
        if len(valid_rr) >= 2:
            rmssd_ms = float(np.sqrt(np.mean(np.diff(valid_rr) ** 2)))

    rmssd_ms = round(float(rmssd_ms), 1)
    if rmssd_ms <= 0.0:
        stress_state = "Calibrating"
    elif rmssd_ms < 22.0:
        stress_state = "High Sympathetic"
    elif rmssd_ms < 42.0:
        stress_state = "Moderate Tone"
    else:
        stress_state = "Optimal Vagal"
    return rmssd_ms, stress_state

def render_episode_png(sig_slice, diagnosis, hr_bpm):
    """Renders a 6-second clinical ECG strip to an in-memory PNG buffer."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 2.4), dpi=130)
        fig.patch.set_facecolor('#06080d')
        ax.set_facecolor('#06080d')

        t = np.arange(len(sig_slice)) / float(FS)
        ax.grid(True, which='both', color='#10b981', alpha=0.14, linestyle='-', linewidth=0.6)
        ax.plot(t, sig_slice, color='#06b6d4', linewidth=1.4)
        ax.set_title(
            f"6.0s Trigger Episode  |  {diagnosis.replace('_', ' ').upper()}  |  {hr_bpm} BPM",
            color='#f8fafc', fontsize=10, fontweight='bold', pad=8
        )
        ax.set_xlabel("Time (seconds)", color='#94a3b8', fontsize=8)
        ax.tick_params(colors='#64748b', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"[Strip Renderer Info] Matplotlib unavailable ({e}), sending text-only episode.")
        return None

def _post_discord_multipart(webhook_url, payload_dict, png_bytes=None):
    """Sends JSON + optional PNG attachment to Discord without external libraries."""
    if not png_bytes:
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload_dict).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "ECG-Monitor/1.0"}
        )
        urllib.request.urlopen(req, timeout=8)
        return

    boundary = f"----ECGBoundary{uuid.uuid4().hex}"
    body = io.BytesIO()

    # Part 1: payload_json
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="payload_json"\r\n')
    body.write(b'Content-Type: application/json\r\n\r\n')
    body.write(json.dumps(payload_dict).encode("utf-8"))
    body.write(b"\r\n")

    # Part 2: PNG file
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="files[0]"; filename="episode_strip.png"\r\n')
    body.write(b'Content-Type: image/png\r\n\r\n')
    body.write(png_bytes)
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        webhook_url,
        data=body.getvalue(),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "ECG-Monitor/1.0"
        }
    )
    urllib.request.urlopen(req, timeout=10)

def _send_alerts_worker(ep_snapshot):
    patient_id   = ep_snapshot.get("patient_id", "PT-LIVE")
    diagnosis    = ep_snapshot["primary"]
    all_rules    = ep_snapshot["types"]
    hr_bpm       = ep_snapshot["hr_bpm"]
    burden_v     = ep_snapshot["burden_v"]
    burden_s     = ep_snapshot["burden_s"]
    burden_q     = ep_snapshot["burden_q"]
    nrmssd       = ep_snapshot["nrmssd"]
    rmssd_ms     = ep_snapshot["rmssd_ms"]
    stress_state = ep_snapshot["stress_state"]
    mode         = ep_snapshot["mode"]
    recent_beats = ep_snapshot["recent_beats"]
    sig_slice    = ep_snapshot["sig_slice"]

    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    clean_diag = DIAGNOSIS_LABELS.get(diagnosis, diagnosis.replace("_", " ").title())
    rules_readable = [DIAGNOSIS_LABELS.get(r, r.replace("_", " ").title()) for r in all_rules]
    rationale = CLINICAL_DESCRIPTIONS.get(
        diagnosis,
        "Abnormal rhythm pattern crossed clinical rule thresholds across the 30-second evaluation window."
    )

    # Format the last 8 beats of the episode
    if recent_beats:
        beat_lines = []
        for b in recent_beats[-8:]:
            flag = "⚠️" if b["label"] in ("V", "S") else "•"
            beat_lines.append(
                f"{flag} t={b['t']}s | Class {b['label']} "
                f"(N:{b['pN']:.2f}, S:{b['pS']:.2f}, V:{b['pV']:.2f}, Q:{b['pQ']:.2f})"
            )
        beat_log_str = "\n".join(beat_lines)
    else:
        beat_log_str = "No individual R-peak probabilities logged in immediate sub-window."

    png_bytes = render_episode_png(sig_slice, diagnosis, hr_bpm) if len(sig_slice) > 50 else None

    # 1. Discord Rich Embed + Episode Strip Image
    if DISCORD_WEBHOOK_URL and "YOUR_DISCORD" not in DISCORD_WEBHOOK_URL:
        try:
            embed = {
                "title": f"🚨 CLINICAL TRIAGE ALERT — {clean_diag}",
                "description": f"**Clinical Rationale:** {rationale}",
                "color": 15548997,
                "fields": [
                    {"name": "Ventricular Rate", "value": f"**{hr_bpm} BPM**", "inline": True},
                    {"name": "Ectopic Burden (30s)", "value": f"PVC (V): **{burden_v}%** | PAC (S): **{burden_s}%**", "inline": True},
                    {"name": "Autonomic HRV", "value": f"**{rmssd_ms} ms** ({stress_state})\nnRMSSD: `{nrmssd}`", "inline": True},
                    {"name": "Triggered Rules", "value": ", ".join(f"`{r}`" for r in rules_readable) if rules_readable else f"`{clean_diag}`", "inline": True},
                    {"name": "Signal Quality (Q-Gate)", "value": f"Noise: {burden_q}% (`{mode}`)", "inline": True},
                    {"name": "Timestamp", "value": timestamp_str, "inline": True},
                    {"name": "Flagged Episode Beat Sequence (Last 8 Beats)", "value": f"```\n{beat_log_str}\n```", "inline": False}
                ],
                "footer": {"text": "Raspberry Pi 5 • 4-Class LightGBM Telemetry"}
            }
            if png_bytes:
                embed["image"] = {"url": "attachment://episode_strip.png"}

            payload = {"username": "ECG Clinical Triage", "embeds": [embed]}
            _post_discord_multipart(DISCORD_WEBHOOK_URL, payload, png_bytes)
        except Exception as e:
            print(f"[Triage Warning] Discord webhook failed: {e}")

    # 2. Automated Clinical SMTP Email + Attached Rhythm Strip
    if SMTP_APP_PASSWORD and "YOUR_16_CHAR" not in SMTP_APP_PASSWORD:
        try:
            msg = EmailMessage()
            msg["Subject"] = f"[ECG TRIAGE ALERT] {clean_diag} — {hr_bpm} BPM"
            msg["From"] = SMTP_SENDER_EMAIL
            msg["To"] = PHYSICIAN_EMAIL
            msg.set_content(
                f"CLINICAL TELEMEDICINE TRIAGE REPORT\n"
                f"============================================================\n"
                f"Primary Diagnosis  : {clean_diag}\n"
                f"Triggered Rules    : {', '.join(rules_readable) if rules_readable else clean_diag}\n"
                f"Timestamp          : {timestamp_str}\n"
                f"Telemetry Source   : {mode}\n\n"
                f"CLINICAL RATIONALE:\n"
                f"{rationale}\n\n"
                f"30-SECOND WINDOW PHYSIOLOGICAL METRICS:\n"
                f"  • Ventricular Rate      : {hr_bpm} BPM\n"
                f"  • PVC Burden (Class V)  : {burden_v}%\n"
                f"  • PAC Burden (Class S)  : {burden_s}%\n"
                f"  • Artifact Burden (Q)   : {burden_q}%\n"
                f"  • HRV (RMSSD / nRMSSD)  : {rmssd_ms} ms / {nrmssd} ({stress_state})\n\n"
                f"FLAGGED EPISODE BEAT LOG (Most Recent Beats Leading to Trigger):\n"
                f"------------------------------------------------------------\n"
                f"{beat_log_str}\n"
                f"------------------------------------------------------------\n"
                f"Note: A 6.0-second 250 Hz ECG rhythm strip of the episode is attached.\n"
            )
            if png_bytes:
                msg.add_attachment(png_bytes, maintype="image", subtype="png", filename="episode_strip.png")

            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=12) as smtp:
                smtp.login(SMTP_SENDER_EMAIL, SMTP_APP_PASSWORD)
                smtp.send_message(msg)
        except Exception as e:
            print(f"[Triage Warning] SMTP Email failed: {e}")

    # Record dispatched alert in persistent SQLite history
    try:
        channels = []
        if DISCORD_WEBHOOK_URL and "YOUR_DISCORD" not in DISCORD_WEBHOOK_URL:
            channels.append("Discord Webhook")
        if SMTP_APP_PASSWORD and "YOUR_16_CHAR" not in SMTP_APP_PASSWORD:
            channels.append(f"SMTP ({PHYSICIAN_EMAIL})")
        pdb.log_alert(
            patient_id=patient_id,
            diagnosis=clean_diag,
            channels=", ".join(channels) if channels else "Console Log",
            recipient=PHYSICIAN_EMAIL,
            status="DISPATCHED"
        )
    except Exception as e:
        print(f"[Record Warning] Failed to log alert in DB: {e}")


class PipelineState:
    def __init__(self):
        self.mode = 'DEMO_NORMAL'
        self.ptr = 0
        self.udp_queue = deque(maxlen=FS * 10)
        self.nn_times = deque(maxlen=40)
        self.recent_beats = deque(maxlen=16)
        self.det = ep.StreamingArrhythmiaDetector(bundle, fs=FS, buf_s=30, hop_s=0.5, margin_s=1.5)
        self.clients = set()
        self.last_alert_ts = 0.0
        self.alert_latch = False
        self.last_esp32_packet_time = 0.0
        self.esp32_packet_count = 0
        self.esp32_samples_received = 0
        self.packet_rate_hz = 0.0
        self._last_calc_ts = time.time()
        self._last_calc_samples = 0
        self.active_patient_id = 'p_frequent_pvc'
        self.active_patient_record_id = 'PT-LIVE'
        self.last_win = dict(
            primary_type='warming_up', types=[], hr_bpm=0.0,
            burden_V=0.0, burden_S=0.0, burden_Q=0.0, nrmssd=0.0,
            rmssd_ms=0.0, stress_state='Calibrating',
            n_beats=0, alert_dispatched=False
        )

    def set_patient(self, pid):
        if pid in PATIENT_ARRAYS:
            self.active_patient_id = pid
            DEMOS['DEMO_ECTOPY'] = PATIENT_ARRAYS[pid]
            self.reset_detector('DEMO_ECTOPY')

    def reset_detector(self, new_mode):
        self.mode = new_mode
        self.ptr = 0
        self.nn_times.clear()
        self.recent_beats.clear()
        self.alert_latch = False
        self.det = ep.StreamingArrhythmiaDetector(bundle, fs=FS, buf_s=30, hop_s=0.5, margin_s=1.5)
        self.last_win = dict(
            primary_type='warming_up', types=[], hr_bpm=0.0,
            burden_V=0.0, burden_S=0.0, burden_Q=0.0, nrmssd=0.0,
            rmssd_ms=0.0, stress_state='Calibrating',
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

@app.get("/api/patients")
async def get_patients():
    return {"status": "ok", "patients": PATIENTS_CATALOG, "active_id": state.active_patient_id}

class PatientSelectRequest(BaseModel):
    patient_id: str

@app.post("/api/patient")
async def select_patient(req: PatientSelectRequest):
    if req.patient_id in PATIENT_ARRAYS:
        state.set_patient(req.patient_id)
        p_info = next((p for p in PATIENTS_CATALOG if p['id'] == req.patient_id), None)
        return {"status": "ok", "patient_id": req.patient_id, "patient": p_info}
    return {"status": "error", "message": "Unknown patient ID"}

# --- PERSISTENT PATIENT RECORDS & HISTORY API ---
class AddPatientRecordRequest(BaseModel):
    patient_id: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = "Other"
    medical_notes: Optional[str] = ""

class SetActivePatientRecordRequest(BaseModel):
    patient_id: str

@app.get("/api/records/patients")
async def get_patient_records():
    pts = pdb.get_patients()
    return {"status": "ok", "patients": pts, "active_id": state.active_patient_record_id}

@app.post("/api/records/patients")
async def add_patient_record(req: AddPatientRecordRequest):
    p = pdb.add_patient(
        req.patient_id.strip(),
        req.name.strip(),
        req.age,
        req.gender,
        req.medical_notes.strip() if req.medical_notes else ""
    )
    return {"status": "ok", "patient": p}

@app.get("/api/records/active")
async def get_active_patient_record():
    p = pdb.get_patient(state.active_patient_record_id)
    return {"status": "ok", "patient": p, "active_id": state.active_patient_record_id}

@app.post("/api/records/active")
async def set_active_patient_record(req: SetActivePatientRecordRequest):
    p = pdb.get_patient(req.patient_id)
    if p:
        state.active_patient_record_id = req.patient_id
        return {"status": "ok", "active_id": state.active_patient_record_id, "patient": p}
    return {"status": "error", "message": "Patient not found"}

@app.get("/api/records/episodes")
async def get_patient_episodes(patient_id: Optional[str] = None, limit: int = 50):
    pid = patient_id if patient_id else state.active_patient_record_id
    episodes = pdb.get_episodes(pid, limit=limit)
    return {"status": "ok", "patient_id": pid, "episodes": episodes}

@app.get("/api/records/alerts")
async def get_patient_alerts(patient_id: Optional[str] = None, limit: int = 50):
    pid = patient_id if patient_id else state.active_patient_record_id
    alerts = pdb.get_alerts(pid, limit=limit)
    return {"status": "ok", "patient_id": pid, "alerts": alerts}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state.clients.add(ws)
    try:
        while True:
            msg = await ws.receive_text()
            # Support incoming ESP32 WebSocket batches directly:
            try:
                data = json.loads(msg)
                if isinstance(data, dict) and "signal" in data:
                    sig = np.array(data["signal"], dtype=np.float32)
                    if len(sig) > 0:
                        if np.max(sig) > 10.0:
                            sig = (sig - 2048.0) / 450.0
                        state.udp_queue.extend(sig.tolist())
                        state.last_esp32_packet_time = time.time()
                        state.esp32_packet_count += 1
                        state.esp32_samples_received += len(sig)
            except Exception:
                pass
    except WebSocketDisconnect:
        state.clients.discard(ws)

async def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', UDP_PORT))
    sock.setblocking(False)
    loop = asyncio.get_event_loop()
    print(f"[UDP] Listening for ESP32 telemetry packets on port {UDP_PORT}...")
    connected_esp = False
    while True:
        try:
            data, addr = await loop.sock_recvfrom(sock, 4096)
            if data.startswith(b"ECG_DISCOVER"):
                sock.sendto(b"ECG_OFFER", addr)
                print(f"[UDP Auto-Discovery] Handshake received from ESP32 at {addr}! Sent Pi offer.")
                continue
            try:
                vals = [float(x) for x in data.decode('utf-8').strip().split(',') if x]
            except (UnicodeDecodeError, ValueError):
                vals = np.frombuffer(data, dtype=np.float32).tolist()
            if not connected_esp:
                print(f"[UDP] Connection established! Receiving live ECG stream from ESP32 at {addr}")
                connected_esp = True
            state.udp_queue.extend(vals)
            state.last_esp32_packet_time = time.time()
            state.esp32_packet_count += 1
            state.esp32_samples_received += len(vals)
        except Exception:
            await asyncio.sleep(0.005)


async def inference_broadcaster():
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
                beat_obj = dict(
                    t=t_val,
                    label=lbl,
                    pN=round(float(pr[0]), 3),
                    pS=round(float(pr[1]), 3),
                    pV=round(float(pr[2]), 3),
                    pQ=round(float(pr[3]), 3) if len(pr) > 3 else 0.0
                )
                new_beats.append(beat_obj)
                state.recent_beats.append(beat_obj)

            w = res['window']
            primary = w['primary_type']
            hr_bpm = 0.0 if np.isnan(w['hr_bpm']) else round(float(w['hr_bpm']), 1)
            burden_v = 0.0 if np.isnan(w['burden_V']) else round(float(w['burden_V']) * 100, 1)
            burden_s = 0.0 if np.isnan(w['burden_S']) else round(float(w['burden_S']) * 100, 1)
            burden_q = 0.0 if np.isnan(w.get('burden_Q', 0.0)) else round(float(w.get('burden_Q', 0.0)) * 100, 1)
            nrmssd = 0.0 if np.isnan(w.get('nrmssd', np.nan)) else round(float(w['nrmssd']), 3)

            rmssd_ms, stress_state = derive_hrv_metrics(hr_bpm, nrmssd, state.nn_times)

            now = time.time()
            if primary not in non_critical_states:
                if (now - state.last_alert_ts) >= ALERT_COOLDOWN_SEC:
                    state.last_alert_ts = now
                    state.alert_latch = True
                    
                    pid = state.active_patient_record_id
                    clean_diag_str = DIAGNOSIS_LABELS.get(primary, primary.replace("_", " ").title())

                    # Persist irregular arrhythmia episode to SQLite database
                    try:
                        pdb.log_episode(
                            patient_id=pid,
                            primary_diagnosis=clean_diag_str,
                            triggered_rules=list(w['types']),
                            hr_bpm=hr_bpm,
                            burden_v=burden_v,
                            burden_s=burden_s,
                            burden_q=burden_q,
                            rmssd_ms=rmssd_ms,
                            stress_state=stress_state,
                            telemetry_source=state.mode,
                            duration_s=30.0
                        )
                    except Exception as e:
                        print(f"[Record Warning] Failed to log episode: {e}")

                    # Slice the last 6 seconds (1500 samples) from the detector buffer for the episode strip
                    buf_arr = np.array(state.det.buf, dtype=np.float32)
                    sig_slice = buf_arr[-int(6 * FS):].copy() if len(buf_arr) > 0 else np.array([])
                    snapshot = dict(
                        patient_id=pid,
                        primary=primary,
                        types=list(w['types']),
                        hr_bpm=hr_bpm,
                        burden_v=burden_v,
                        burden_s=burden_s,
                        burden_q=burden_q,
                        nrmssd=nrmssd,
                        rmssd_ms=rmssd_ms,
                        stress_state=stress_state,
                        mode=state.mode,
                        recent_beats=list(state.recent_beats),
                        sig_slice=sig_slice
                    )
                    threading.Thread(target=_send_alerts_worker, args=(snapshot,), daemon=True).start()
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

        now_ts = time.time()
        is_live_mode = (state.mode == 'ESP32_LIVE')
        time_since_esp = (now_ts - state.last_esp32_packet_time) if state.last_esp32_packet_time > 0 else 999.0

        has_real_normal = os.path.exists(os.path.join(META_DIR, 'demo_normal.npy'))
        has_real_ectopy = os.path.exists(os.path.join(META_DIR, 'demo_ectopy.npy'))

        active_p = next((p for p in PATIENTS_CATALOG if p['id'] == state.active_patient_id), None)
        if is_live_mode:
            is_fallback = (time_since_esp > 2.0)
            if is_fallback:
                stream_status = "FALLBACK_STARVED"
                source_detail = f"ESP32-S3 Stream Starved (no packets for {time_since_esp:.1f}s)"
            else:
                stream_status = "LIVE_HARDWARE"
                source_detail = "AD8232 Hardware (250 Hz Active Stream)"
        elif state.mode == 'DEMO_NORMAL':
            is_fallback = not has_real_normal
            stream_status = "DATABASE_REAL" if has_real_normal else "SYNTHETIC_FALLBACK"
            source_detail = "Icentia11k Clinical DB (Real Normal Sinus)" if has_real_normal else "Synthetic Fallback Wave"
        elif state.mode == 'DEMO_ECTOPY':
            is_fallback = len(PATIENT_ARRAYS) == 0 and not has_real_ectopy
            stream_status = "DATABASE_REAL" if not is_fallback else "SYNTHETIC_FALLBACK"
            source_detail = f"Clinical DB: {active_p['name']}" if active_p else "Icentia11k Clinical DB (Real Arrhythmia)"
        else:
            is_fallback = False
            stream_status = "ARTIFACT_SIMULATION"
            source_detail = "Motion / Tremor Noise Simulation"

        dt_calc = now_ts - state._last_calc_ts
        if dt_calc >= 1.0:
            if is_live_mode:
                state.packet_rate_hz = round((state.esp32_samples_received - state._last_calc_samples) / dt_calc, 1)
            else:
                state.packet_rate_hz = 250.0
            state._last_calc_ts = now_ts
            state._last_calc_samples = state.esp32_samples_received

        telemetry_info = {
            "mode": state.mode,
            "status": stream_status,
            "detail": source_detail,
            "active_patient_id": state.active_patient_id,
            "active_patient_name": active_p['name'] if active_p else "Patient 101",
            "is_fallback": is_fallback,
            "rate_hz": state.packet_rate_hz,
            "last_packet_s": round(time_since_esp, 1) if (is_live_mode and state.last_esp32_packet_time > 0) else 0.0,
            "packets_total": state.esp32_packet_count,
            "queue_len": len(state.udp_queue) if is_live_mode else len(state.det.buf)
        }

        payload = json.dumps(dict(
            mode=state.mode,
            telemetry=telemetry_info,
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

@app.get("/dashboard_premium.html", response_class=HTMLResponse)
@app.get("/dashboard_realtime.html", response_class=HTMLResponse)
@app.get("/dashboard.html", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def index():
    for name in ["dashboard_premium.html", "dashboard_realtime.html", "dashboard_anti_slop.html", "dashboard.html"]:
        p = os.path.join(os.path.dirname(__file__), name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return f.read()
    return "<h1>Dashboard not found</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
