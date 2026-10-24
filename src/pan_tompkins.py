"""pan_tompkins.py - R-peak detector (Pan-Tompkins style, adaptive thresholds + search-back) and peak matching."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks


def _classic(f, d, integ, cand, fs):
    """Original recursive-threshold Pan-Tompkins (thresholds initialised from the first 2 s)."""
    seg = integ[:2 * fs]
    spki, npki = 0.25 * seg.max(), 0.5 * seg.mean()
    acc, rej, rr_hist = [], [], []
    hw = int(0.075 * fs)
    slope = lambda c: np.max(np.abs(d[max(0, c - hw):c + hw]))
    for c in cand:
        pk = integ[c]; thr1 = npki + 0.25 * (spki - npki)
        # search-back: long gap with no beat -> accept the strongest rejected candidate above half threshold
        if len(rr_hist) >= 2 and c - acc[-1] > 1.66 * np.mean(rr_hist[-8:]) and rej:
            sb = [r for r in rej if r > acc[-1] and integ[r] > 0.5 * thr1]
            if sb:
                r = max(sb, key=lambda x: integ[x]); rr_hist.append(r - acc[-1]); acc.append(r)
                spki = 0.25 * integ[r] + 0.75 * spki; rej = [x for x in rej if x > r]
        if pk > thr1:
            if acc and c - acc[-1] < int(0.36 * fs) and slope(c) < 0.5 * slope(acc[-1]):   # T-wave
                npki = 0.125 * pk + 0.875 * npki; rej.append(c); continue
            if acc: rr_hist.append(c - acc[-1])
            acc.append(c); spki = 0.125 * pk + 0.875 * spki; rej = []
        else:
            npki = 0.125 * pk + 0.875 * npki; rej.append(c)
    return acc


def _refine(acc, f, fs):
    h = int(0.1 * fs); out = []
    for c in acc:
        lo, hi = max(0, c - h), min(len(f), c + h + 1)
        out.append(lo + int(np.argmax(np.abs(f[lo:hi]))))
    return np.unique(out).astype(np.int64)


def _local(f, d, integ, cand, fs, k=0.35, win_s=5.0, sb_k=0.2):
    """Local-threshold variant: threshold = k * (90th percentile of candidate heights within +-win_s).
    No state carried from one beat to the next, so a single artifact or a bad start cannot lock the detector out."""
    env = np.sqrt(integ); h = env[cand]; tc = cand / fs
    lo = np.searchsorted(tc, tc - win_s); hi = np.searchsorted(tc, tc + win_s)
    ref = np.array([np.percentile(h[a:b], 90) for a, b in zip(lo, hi)])
    hw = int(0.075 * fs)
    slope = lambda c: np.max(np.abs(d[max(0, c - hw):c + hw]))
    acc = []
    for i in np.flatnonzero(h >= k * ref):
        c = cand[i]
        if acc and c - acc[-1] < int(0.36 * fs) and slope(c) < 0.5 * slope(acc[-1]): continue   # T-wave
        acc.append(c)
    # search-back: gap much longer than the recent RR -> take strongest weaker candidate inside the gap
    cl, i0, out = list(cand), 0, [acc[0]] if acc else []
    for c in acc[1:]:
        prev = out[-1]; rr = np.diff(out[-8:]); gap = c - prev
        if len(rr) >= 2 and gap > 1.66 * np.median(rr):
            m = (cand > prev + int(0.2 * fs)) & (cand < c - int(0.2 * fs))
            idx = np.flatnonzero(m)
            if len(idx):
                j = idx[np.argmax(h[idx])]
                if h[j] >= sb_k * ref[j]: out.append(cand[j])
        out.append(c)
    return out


def pan_tompkins(sig, fs=250, method='local'):
    """Return sorted R-peak sample indices. method: 'local' (default, robust) or 'classic'."""
    sig = np.asarray(sig, dtype=np.float64)
    f = sosfiltfilt(butter(2, [5, 15], 'band', fs=fs, output='sos'), sig)
    d = np.gradient(f); w = int(0.15 * fs)
    integ = np.convolve(d * d, np.ones(w) / w, mode='same')
    cand, _ = find_peaks(integ, distance=int(0.2 * fs))
    if len(cand) == 0:
        return np.array([], dtype=np.int64)
    acc = _classic(f, d, integ, cand, fs) if method == 'classic' else _local(f, d, integ, cand, fs)
    return _refine(acc, f, fs)


def match_peaks(ref, det, tol):
    """Greedy one-to-one matching of sorted peak arrays within tol samples.
    Returns dict(tp, fn, fp, se, ppv, pairs=[(ref_i, det_j), ...])."""
    ref, det = np.asarray(ref), np.asarray(det); i = j = 0; pairs = []
    while i < len(ref) and j < len(det):
        if abs(ref[i] - det[j]) <= tol:
            pairs.append((i, j)); i += 1; j += 1
        elif det[j] < ref[i]: j += 1
        else: i += 1
    tp = len(pairs); fn = len(ref) - tp; fp = len(det) - tp
    return dict(tp=tp, fn=fn, fp=fp, se=tp / max(len(ref), 1), ppv=tp / max(len(det), 1), pairs=pairs)
