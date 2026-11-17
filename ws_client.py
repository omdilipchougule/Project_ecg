"""
ws_client.py - WebSocket Client for ECG Telemetry

Features:
1. Transmitter Mode (default): Simulates an ESP32 sending 250Hz ECG data batches to server.py.
   Allows testing the Raspberry Pi server and dashboard without hardware attached.
2. Receiver / Listener Mode: Connects to the server to monitor and print live telemetry metrics
   (Heart Rate, HRV/RMSSD, Arrhythmia Burden, Diagnosis).

Usage:
  # Simulate ESP32 sending ECG batches to the Raspberry Pi:
  python ws_client.py --host 192.168.39.48 --port 8080 --mode send

  # Listen to live telemetry from the server:
  python ws_client.py --host 192.168.39.48 --port 8080 --mode listen
"""

import asyncio
import json
import argparse
import time
import math
import sys

try:
    import websockets
except ImportError:
    print("Error: 'websockets' library is required. Install it using: pip install websockets")
    sys.exit(1)


def generate_synthetic_ecg_batch(batch_size=250, fs=250, bpm=75, noise_level=15.0, phase_offset=0.0):
    """
    Generates a realistic synthetic ECG batch of ADC values (0-4095 range, centered at ~2048)
    simulating an AD8232 sensor reading.
    """
    signal = []
    beat_period = fs / (bpm / 60.0)  # samples per beat (~200 samples at 75 bpm, 250 Hz)
    
    for i in range(batch_size):
        t_sample = (phase_offset + i) % beat_period
        t_phase = (t_sample / beat_period) * 2 * math.pi
        
        # Synthetic P-Q-R-S-T waves using sum of Gaussians
        # P wave
        p_wave = 120.0 * math.exp(-((t_phase - 1.2) ** 2) / 0.04)
        # Q wave
        q_wave = -150.0 * math.exp(-((t_phase - 1.9) ** 2) / 0.01)
        # R peak (high spike)
        r_wave = 1200.0 * math.exp(-((t_phase - 2.1) ** 2) / 0.008)
        # S wave
        s_wave = -250.0 * math.exp(-((t_phase - 2.3) ** 2) / 0.012)
        # T wave
        t_wave = 220.0 * math.exp(-((t_phase - 3.2) ** 2) / 0.08)
        
        ecg_val = 2048.0 + (p_wave + q_wave + r_wave + s_wave + t_wave)
        
        # Baseline wander (~0.25 Hz)
        wander = 50.0 * math.sin(2 * math.pi * 0.25 * ((phase_offset + i) / fs))
        ecg_val += wander
        
        # Clamp to 12-bit ADC range (0-4095)
        raw_val = int(max(0, min(4095, round(ecg_val))))
        signal.append(raw_val)
        
    return signal, (phase_offset + batch_size) % (fs * 60)


async def send_ecg_stream(uri, batch_size=250, sample_rate=250, bpm=72):
    """Connects as an ESP32 emulator and streams ECG batches continuously."""
    print(f"\n[SENDER] Connecting to WebSocket server at {uri} ...")
    interval = batch_size / sample_rate  # 1.0 second per batch

    while True:
        try:
            async with websockets.connect(uri) as ws:
                print(f"[SENDER] Connected to {uri}!")
                print(f"[SENDER] Streaming synthetic ECG at {sample_rate} Hz ({batch_size} samples/sec, ~{bpm} BPM)...")
                print("Press Ctrl+C to stop.\n")
                
                phase = 0.0
                batch_count = 0
                while True:
                    start_time = time.time()
                    signal, phase = generate_synthetic_ecg_batch(batch_size=batch_size, fs=sample_rate, bpm=bpm, phase_offset=phase)
                    
                    payload = json.dumps({"signal": signal})
                    await ws.send(payload)
                    batch_count += 1
                    
                    print(f"\r[SENDER] Sent batch #{batch_count} ({len(signal)} samples) | Last ADC: {signal[-1]}", end="", flush=True)
                    
                    # Pace the transmission to match real time
                    elapsed = time.time() - start_time
                    sleep_time = max(0.0, interval - elapsed)
                    await asyncio.sleep(sleep_time)

        except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
            print(f"\n[SENDER] Connection lost or refused: {e}. Reconnecting in 3 seconds...")
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            break


async def listen_ecg_stream(uri):
    """Connects as a client to display processed telemetry from server.py."""
    print(f"\n[RECEIVER] Connecting to WebSocket server at {uri} ...")
    while True:
        try:
            async with websockets.connect(uri) as ws:
                print(f"[RECEIVER] Connected to {uri}!")
                print("[RECEIVER] Waiting for telemetry datastream...\n")
                
                async for message in ws:
                    try:
                        data = json.loads(message)
                        w = data.get("window", {})
                        hr = w.get("hr_bpm", 0)
                        bv = w.get("burden_V", 0)
                        bs = w.get("burden_S", 0)
                        rmssd = w.get("rmssd_ms", 0)
                        stress = w.get("stress_state", "N/A")
                        primary = w.get("primary_type", "N/A")
                        alert = w.get("alert_dispatched", False)
                        
                        status = f"[TELEMETRY] HR: {hr:3.0f} BPM | RMSSD: {rmssd:5.1f}ms | PVC: {bv:4.1f}% | PAC: {bs:4.1f}% | Stress: {stress:<8} | Diagnosis: {primary}"
                        if alert:
                            status += " [ALERT!]"
                        print(status)
                    except json.JSONDecodeError:
                        print(f"[RECEIVER] Raw frame: {message[:80]}...")
        except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
            print(f"\n[RECEIVER] Connection lost or refused: {e}. Reconnecting in 3 seconds...")
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            break


def main():
    parser = argparse.ArgumentParser(description="ECG Telemetry WebSocket Client")
    parser.add_argument("--host", default="127.0.0.1", help="Server IP address or hostname (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--path", default="/ws", help="WebSocket path (default: /ws)")
    parser.add_argument("--mode", choices=["send", "listen"], default="send", 
                        help="Mode: 'send' (simulate ESP32 stream) or 'listen' (monitor processed telemetry)")
    parser.add_argument("--bpm", type=int, default=75, help="Simulated heart rate in BPM (send mode only)")

    args = parser.parse_args()
    uri = f"ws://{args.host}:{args.port}{args.path}"

    try:
        if args.mode == "send":
            asyncio.run(send_ecg_stream(uri, bpm=args.bpm))
        else:
            asyncio.run(listen_ecg_stream(uri))
    except KeyboardInterrupt:
        print("\n[CLIENT] Stopped by user.")


if __name__ == "__main__":
    main()
