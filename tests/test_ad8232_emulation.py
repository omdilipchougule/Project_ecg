"""test_ad8232_emulation.py - Offline emulation of AD8232 + ESP32 hardware impairments on Icentia11k."""
import os, numpy as np, pandas as pd
import ecg_pipeline as ep
from ecg_features import FS, LABEL_MAP
from pan_tompkins import match_peaks

DATA_DIR      = r"D:\megaproject\icentia11k"
MODEL         = 'ecg_improved_v3.pkl'
TEST_PATIENTS = 'test_patients.npy'


def emulate_ad8232_esp32(sig, fs=FS, seed=42):
    """Apply realistic AD8232 + ESP32 12-bit ADC impairments to a clean ECG signal:
       1. Respiratory baseline wander (0.25 Hz)
       2. 50 Hz powerline hum + broadband EMG muscle tremor
       3. Electrode motion / lead-off rail saturation burst (t = 60s..72s)
       4. ESP32 12-bit ADC quantization (0..4095 levels over +-3 mV dynamic range)"""
    rng = np.random.default_rng(seed)
    t = np.arange(len(sig)) / fs
    s = sig.astype(np.float64).copy()

    # 1. Baseline wander + 50 Hz mains interference + subtle EMG noise
    s += 0.25 * np.sin(2 * np.pi * 0.25 * t)
    s += 0.15 * np.sin(2 * np.pi * 50.0 * t)
    s += rng.normal(0, 0.03, len(s))

    # 2. Simulate 12-second electrode motion / intermittent lead-off saturation (t = 60..72s)
    a, b = int(60 * fs), int(72 * fs)
    if b < len(s):
        s[a:b] += rng.choice([-2.8, 0.0, 2.8], size=b - a, p=[0.25, 0.50, 0.25])
        s[a:b] += rng.normal(0, 0.8, b - a)

    # 3. 12-bit ADC quantization (4096 levels across [-3.0, +3.0] mV)
    vmin, vmax = -3.0, 3.0
    s_clipped = np.clip(s, vmin, vmax)
    adc_codes = np.round((s_clipped - vmin) / (vmax - vmin) * 4095.0)
    s_quant   = (adc_codes / 4095.0) * (vmax - vmin) + vmin
    return s_quant.astype(np.float32)


if __name__ == '__main__':
    bundle = ep.load_bundle(MODEL)
    test_pids = set(np.load(TEST_PATIENTS, allow_pickle=True).tolist())
    recs = [h for h in ep.select_records(DATA_DIR) if os.path.basename(h).split('_')[0] in test_pids][:20]

    # Pick a record with valid beats
    sig, smp, truth = None, None, None
    for h in recs:
        sig, smp, truth = ep.read_record(h)
        if sig is not None and len(smp) > 500:
            rec_name = os.path.basename(h)[:-4]
            break

    MAX_S = 180
    m = smp < MAX_S * FS
    ref_smp, ref_lab = smp[m], truth[m]

    sig_hw = emulate_ad8232_esp32(sig[:int(MAX_S * FS)])
    beats_clean, _, _ = ep.simulate_stream(sig, bundle, chunk_s=0.5, max_s=MAX_S)
    beats_hw, _, stats = ep.simulate_stream(sig_hw, bundle, chunk_s=0.5, max_s=MAX_S)

    mc = match_peaks(ref_smp, beats_clean['sample'].values, int(0.15 * FS))
    mh = match_peaks(ref_smp, beats_hw['sample'].values, int(0.15 * FS))
    print(f"=== AD8232 + ESP32 Emulation on {rec_name} (first {MAX_S}s) ===")
    print(f"Original stream  -> Peak Se={mc['se']:.4f} PPV={mc['ppv']:.4f}")
    print(f"Emulated AD8232  -> Peak Se={mh['se']:.4f} PPV={mh['ppv']:.4f} | Compute={stats['mean_s']*1000:.1f} ms/hop")

    # Inspect 30s window classifications across the 60..72s motion burst (Window 2: 60..90s)
    from arrhythmia_rules import analyse_stream
    wins = analyse_stream(beats_hw['label'].values, beats_hw['t'].values, MAX_S, 30.0, 30.0)
    w_df = pd.DataFrame([dict(t0=w['t0'], t1=w['t1'], n_beats=w['n_beats'],
                              burden_Q=round(w.get('burden_Q', 0.0), 3),
                              primary=w['primary_type'], types=', '.join(w['types'])) for w in wins])
    print("\nWindow-level output under emulated hardware stream (note t0=60.0 motion burst window):")
    print(w_df.to_string(index=False))