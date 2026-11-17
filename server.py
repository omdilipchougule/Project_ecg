# ECG WebSocket Server
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import numpy as np
from collections import deque

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

class ECGProcessor:
    def __init__(self):
        self.buffer = deque(maxlen=1000)  # 4 seconds at 250Hz
        self.r_peaks = []
        self.hr_bpm = 0
        self.window_size = 30  # 30 seconds window
        self.window_beats = 0
        self.burden_V = 0
        self.burden_S = 0
        self.burden_Q = 0
        self.rmssd_ms = 0
        self.stress_state = "Normal"
        self.primary_type = "warming_up"
        self.alert_dispatched = False
        self.types = []

    def process(self, data):
        self.buffer.extend(data)

        if len(self.buffer) >= 250:  # 1 second of data
            self.detect_r_peaks()
            self.calculate_hr()
            self.calculate_metrics()

        return {
            "signal": data,
            "hr_bpm": self.hr_bpm,
            "r_peaks": self.r_peaks[-10:] if self.r_peaks else [],
            "window": {
                "hr_bpm": self.hr_bpm,
                "burden_V": self.burden_V,
                "burden_S": self.burden_S,
                "burden_Q": self.burden_Q,
                "rmssd_ms": self.rmssd_ms,
                "stress_state": self.stress_state,
                "primary_type": self.primary_type,
                "alert_dispatched": self.alert_dispatched,
                "types": self.types,
                "n_beats": self.window_beats,
                "nrmssd": len(self.r_peaks)
            }
        }

    def detect_r_peaks(self):
        # Simple R-peak detection algorithm
        if len(self.buffer) < 250:  # Need at least 1 second of data
            return

        # Convert deque to numpy array
        data = np.array(self.buffer)

        # Simple peak detection
        threshold = np.mean(data) + 2 * np.std(data)
        peaks = np.where(data > threshold)[0]

        # Filter peaks (simple version)
        if len(peaks) > 0:
            # Add new peaks to our list
            self.r_peaks.extend(peaks)
            # Keep only the last 30 seconds of peaks
            if len(self.r_peaks) > self.window_size * 250:
                self.r_peaks = self.r_peaks[-self.window_size * 250:]

    def calculate_hr(self):
        if len(self.r_peaks) > 1:
            rr_intervals = np.diff(self.r_peaks)
            avg_rr = np.mean(rr_intervals)
            self.hr_bpm = 60 / (avg_rr / 250)  # 250Hz sample rate
        else:
            self.hr_bpm = 0

    def calculate_metrics(self):
        # Calculate window metrics
        if len(self.r_peaks) > 1:
            rr_intervals = np.diff(self.r_peaks)
            self.window_beats = len(rr_intervals)

            # Calculate RMSSD
            if len(rr_intervals) > 1:
                diff_rr = np.diff(rr_intervals)
                self.rmssd_ms = np.sqrt(np.mean(diff_rr**2)) / 250 * 1000  # Convert to ms
            else:
                self.rmssd_ms = 0

            # Determine stress state
            if self.rmssd_ms < 22:
                self.stress_state = "Stressed"
            elif self.rmssd_ms > 60:
                self.stress_state = "Relaxed"
            else:
                self.stress_state = "Normal"

            # Calculate burdens (simplified)
            self.burden_V = min(100, len([x for x in rr_intervals if x > 1.2 * np.mean(rr_intervals)]) / len(rr_intervals) * 100)
            self.burden_S = min(100, len([x for x in rr_intervals if x < 0.8 * np.mean(rr_intervals)]) / len(rr_intervals) * 100)
            self.burden_Q = min(100, len([x for x in rr_intervals if x < 0.5 * np.mean(rr_intervals)]) / len(rr_intervals) * 100)

            # Determine primary arrhythmia type
            if self.burden_V > 30:
                self.primary_type = "frequent_pvc_burden"
            elif self.burden_S > 30:
                self.primary_type = "frequent_pac_burden"
            elif self.hr_bpm > 100:
                self.primary_type = "tachycardia"
            elif self.hr_bpm < 60:
                self.primary_type = "bradycardia"
            else:
                self.primary_type = "no_arrhythmia_detected"

            # Check for alert conditions
            if self.burden_V > 50 or self.burden_S > 50 or self.hr_bpm > 120 or self.hr_bpm < 40:
                self.alert_dispatched = True
            else:
                self.alert_dispatched = False

            # Update active types
            self.types = []
            if self.burden_V > 20:
                self.types.append("PVC")
            if self.burden_S > 20:
                self.types.append("PAC")
            if self.hr_bpm > 100:
                self.types.append("Tachycardia")
            if self.hr_bpm < 60:
                self.types.append("Bradycardia")

manager = ConnectionManager()
processor = ECGProcessor()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            ecg_data = json.loads(data)
            processed_data = processor.process(ecg_data["signal"])
            await manager.broadcast(json.dumps(processed_data))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Serve dashboard directly
import os
from fastapi.responses import FileResponse

@app.get("/")
async def get_dashboard():
    # Prefer dashboard_realtime.html or dashboard.html in current directory
    for fname in ["dashboard_realtime.html", "dashboard.html", "index.html"]:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
        if os.path.exists(p):
            return FileResponse(p)
    return {"message": "ECG Telemetry Server Running. Put dashboard_realtime.html in the folder."}

if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")):
    app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8000
    print(f"Starting server on http://0.0.0.0:{port} ...")
    uvicorn.run(app, host="0.0.0.0", port=port)
