"""server.py - Raspberry Pi 5 FastAPI + WebSocket + UDP Telemetry Server for ECG Arrhythmia Pipeline."""
import os, sys, time, json, socket, asyncio
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

class PipelineState:
    def __init__(self):
        self.mode = 'DEMO_NORMAL'
        self.ptr = 0
        self.udp_queue = deque(maxlen=FS * 10)
        self.det = ep.StreamingArrhythmiaDetector(bundle, fs=FS, buf_s=30, hop_s=0.5, margin_s=1.5)
        self.clients = set()
        self.last_win = dict(primary_type='warming_up', types=[], hr_bpm=0.0,
                             burden_V=0.0, burden_S=0.0, burden_Q=0.0, nrmssd=0.0, n_beats=0)

    def reset_detector(self, new_mode):
        self.mode = new_mode
        self.ptr = 0
        self.det = ep.StreamingArrhythmiaDetector(bundle, fs=FS, buf_s=30, hop_s=0.5, margin_s=1.5)

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
    while True:
        t0 = time.perf_counter()
        chunk = state.next_chunk()
        res = state.det.push(chunk)
        comp_ms = (time.perf_counter() - t0) * 1000.0

        new_beats = []
        if res is not None:
            w = res['window']
            state.last_win = dict(
                primary_type=w['primary_type'],
                types=w['types'],
                hr_bpm=0.0 if np.isnan(w['hr_bpm']) else round(float(w['hr_bpm']), 1),
                burden_V=0.0 if np.isnan(w['burden_V']) else round(float(w['burden_V']) * 100, 1),
                burden_S=0.0 if np.isnan(w['burden_S']) else round(float(w['burden_S']) * 100, 1),
                burden_Q=0.0 if np.isnan(w.get('burden_Q', 0.0)) else round(float(w.get('burden_Q', 0.0)) * 100, 1),
                nrmssd=0.0 if np.isnan(w.get('nrmssd', np.nan)) else round(float(w['nrmssd']), 3),
                n_beats=int(w['n_beats'])
            )
            for b in res['new_beats']:
                pr = b['probs']
                new_beats.append(dict(
                    t=round(float(b['t']), 2),
                    label=b['label'],
                    pN=round(float(pr[0]), 3),
                    pS=round(float(pr[1]), 3),
                    pV=round(float(pr[2]), 3),
                    pQ=round(float(pr[3]), 3) if len(pr) > 3 else 0.0
                ))

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