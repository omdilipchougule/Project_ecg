"""ecg_features.py - v3 config + 4-class (N/S/V/Q) cached feature extraction with long-range rhythm context."""

import os, random, time, warnings
import numpy as np, pandas as pd, joblib
from collections import Counter
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold, ParameterSampler
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score,
                             average_precision_score, precision_recall_curve)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED)

DATA_DIR     = r"D:\megaproject\icentia11k"
CACHE_DIR    = r"D:\megaproject\feature_cache"
FEAT_VERSION = "v3"                                # v3: 4 classes (N/S/V/Q) + 27 features
MAX_RECORDS  = 1005
FS           = 250
NOTCH_HZ     = None
BEAT_SYMS    = {'N', 'S', 'V', 'Q'}
LABEL_MAP    = {'N': 0, 'S': 1, 'V': 2, 'Q': 3}
INV_LABEL    = {v: k for k, v in LABEL_MAP.items()}
N_JOBS       = -1
N_ITER       = 20

try:
    import psutil
    def rss_mb(): return psutil.Process().memory_info().rss / 2**20
except ImportError:
    def rss_mb(): return float('nan')

class Timer:
    def __init__(self, name): self.name = name
    def __enter__(self): self.t = time.perf_counter(); return self
    def __exit__(self, *a): print(f"[{self.name}] {time.perf_counter()-self.t:.1f}s | RSS {rss_mb():.0f} MB")


def clean_signal(sig, fs=FS):
    sos = butter(4, [0.5, 40.0], 'band', fs=fs, output='sos')
    sig = sosfiltfilt(sos, sig)
    if NOTCH_HZ:
        b, a = iirnotch(NOTCH_HZ, 30.0, fs); sig = filtfilt(b, a, sig)
    return ((sig - sig.mean()) / (sig.std() + 1e-6)).astype(np.float32)

def _windows(sigp, idx, half):
    return sliding_window_view(sigp, 2*half + 1)[idx]

RR_FEATS = [
    'rr_prev', 'rr_next', 'rr_prev2', 'rr_next2', 'rr_med', 'rr_sd',
    'prev_over_med', 'prev_over_next', 'next_over_med', 'pause_ratio', 'prev_over_prev2',
    'rr_med20', 'rr_cv20', 'prev_over_med20', 'comp_deficit', 'local_premature_rate', 'rr_diff_norm'
]
MORPH    = ['r_amp', 'qrs_width_ms', 'qrs_area', 'max_slope', 'tmpl_corr']
WAVE     = ['p_energy', 't_energy']
QUALITY  = ['noise', 'baseline_off', 'amp_ratio']
FEATURES = RR_FEATS + MORPH + WAVE + QUALITY
NON_CAUSAL = ['rr_next', 'rr_next2', 'next_over_med', 'prev_over_next', 'pause_ratio', 'comp_deficit']


def get_features(sig, peaks, fs=FS):
    """Vectorised per-beat features (27 total). NaN = not computable (handled natively by LightGBM)."""
    N = len(sig); peaks = np.clip(np.asarray(peaks, dtype=np.int64), 0, N-1)
    half = int(0.35*fs); c = half
    ms = lambda t: int(round(t*fs/1000))
    sigp = np.pad(sig, half)

    # Refine peak to local |extremum| (+-40 ms)
    win = _windows(sigp, peaks, half)
    r = ms(40)
    sub = np.abs(win[:, c-r:c+r+1] - np.median(win, axis=1, keepdims=True))
    pk = np.clip(peaks + sub.argmax(1) - r, 0, N-1)
    win = _windows(sigp, pk, half)
    base = np.median(win, axis=1, keepdims=True)
    dev = win - base
    r_amp = dev[:, c]

    # QRS width (FWHM in ms)
    ab = np.abs(dev); above = ab >= 0.5*ab[:, [c]]
    idx = np.arange(win.shape[1])
    left  = np.where(~above[:, :c], idx[:c], -1).max(1)
    right = np.where(~above[:, c+1:], idx[c+1:], win.shape[1]).min(1)
    qrs_width = (right - left - 1) * 1000.0 / fs

    q = dev[:, c-ms(100):c+ms(100)+1]
    qrs_area  = np.abs(q).sum(1) / fs
    max_slope = np.abs(np.diff(q, axis=1)).max(1) * fs
    tmpl = np.median(q, axis=0)
    a = q - q.mean(1, keepdims=True); b = tmpl - tmpl.mean()
    tmpl_corr = (a @ b) / (np.linalg.norm(a, axis=1)*np.linalg.norm(b) + 1e-8)

    p_energy = (dev[:, c-ms(250):c-ms(90)]**2).mean(1)
    t_energy = (dev[:, c+ms(120):c+ms(300)]**2).mean(1)

    # Local HF noise via cumulative sums
    d = np.diff(sig.astype(np.float64), prepend=sig[0])
    cs1 = np.concatenate([[0], np.cumsum(d)]); cs2 = np.concatenate([[0], np.cumsum(d*d)])
    lo = np.clip(pk - fs//2, 0, N); hi = np.clip(pk + fs//2 + 1, 0, N); cnt = (hi - lo)
    var = (cs2[hi]-cs2[lo])/cnt - ((cs1[hi]-cs1[lo])/cnt)**2
    noise = np.sqrt(np.maximum(var, 0))

    # Short- and long-range RR context
    rr = np.diff(peaks) / fs
    rr_prev = np.r_[np.nan, rr]; rr_next = np.r_[rr, np.nan]
    rr_prev2 = np.r_[np.nan, rr_prev[:-1]]; rr_next2 = np.r_[rr_next[1:], np.nan]
    s_rr_prev = pd.Series(rr_prev)
    hist8  = s_rr_prev.shift(1).rolling(8, min_periods=3)
    rr_med = hist8.median().values; rr_sd = hist8.std().values

    hist20   = s_rr_prev.shift(1).rolling(20, min_periods=5)
    rr_med20 = hist20.median().values; rr_sd20 = hist20.std().values

    amp_med = pd.Series(np.abs(r_amp)).shift(1).rolling(8, min_periods=3).median().values
    e = 1e-6
    rr_cv20          = rr_sd20 / (rr_med20 + e)
    prev_over_med20  = rr_prev / (rr_med20 + e)
    comp_deficit     = (2.0 * rr_med - (rr_prev + rr_next)) / (rr_med + e)
    is_prem          = (rr_prev / (rr_med20 + e) < 0.85).astype(np.float32)
    local_prem_rate  = pd.Series(is_prem).shift(1).rolling(10, min_periods=3).mean().values
    rr_diff_norm     = (rr_prev - rr_prev2) / (rr_med + e)

    cols = dict(
        rr_prev=rr_prev, rr_next=rr_next, rr_prev2=rr_prev2, rr_next2=rr_next2,
        rr_med=rr_med, rr_sd=rr_sd,
        prev_over_med=rr_prev/(rr_med+e), prev_over_next=rr_prev/(rr_next+e),
        next_over_med=rr_next/(rr_med+e), pause_ratio=(rr_prev+rr_next)/(2*rr_med+e),
        prev_over_prev2=rr_prev/(rr_prev2+e),
        rr_med20=rr_med20, rr_cv20=rr_cv20, prev_over_med20=prev_over_med20,
        comp_deficit=comp_deficit, local_premature_rate=local_prem_rate, rr_diff_norm=rr_diff_norm,
        r_amp=r_amp, qrs_width_ms=qrs_width, qrs_area=qrs_area, max_slope=max_slope, tmpl_corr=tmpl_corr,
        p_energy=p_energy, t_energy=t_energy,
        noise=noise, baseline_off=np.abs(base[:, 0]), amp_ratio=np.abs(r_amp)/(amp_med+e)
    )
    return np.column_stack([cols[k] for k in FEATURES]).astype(np.float32)


def process_record(hp):
    """One record -> (status, X, y, patient_id). Cached on disk."""
    p = hp[:-4]; name = os.path.basename(p)
    cache = os.path.join(CACHE_DIR, f"{name}_{FEAT_VERSION}.npz")
    pid = name.split('_')[0]
    if os.path.exists(cache):
        z = np.load(cache); return 'ok', z['X'], z['y'], pid
    try:
        import wfdb
        rec = wfdb.rdrecord(p); ann = wfdb.rdann(p, 'atr')
        sig = rec.p_signal[:, 0]
        sym = np.array(ann.symbol); smp = np.asarray(ann.sample)
        keep = np.isin(sym, list(BEAT_SYMS)); sym, smp = sym[keep], smp[keep]
        if np.any(np.isnan(sig)) or len(smp) < 12: return 'skip', None, None, pid
        feat = get_features(clean_signal(sig), smp)
        m = np.isin(sym, list(LABEL_MAP))
        if not m.any(): return 'skip', None, None, pid
        X = feat[m]; y = np.array([LABEL_MAP[s] for s in sym[m]], dtype=np.int8)
        np.savez_compressed(cache, X=X, y=y)
        return 'ok', X, y, pid
    except Exception as ex:
        return f'err:{type(ex).__name__}', None, None, pid


def load_data(data_dir, max_records=MAX_RECORDS):
    os.makedirs(CACHE_DIR, exist_ok=True)
    hea = sorted(os.path.join(r, f) for r, _, fs in os.walk(data_dir) for f in fs if f.endswith('.hea'))
    random.Random(SEED).shuffle(hea); hea = hea[:max_records]
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(process_record)(h) for h in hea)
    Xs, ys, gs, bad = [], [], [], Counter()
    for st, X, y, pid in res:
        if st == 'ok': Xs.append(X); ys.append(y); gs.extend([pid]*len(y))
        else: bad[st] += 1
    print("Skipped/err:", dict(bad))
    return np.vstack(Xs), np.concatenate(ys), np.array(gs)