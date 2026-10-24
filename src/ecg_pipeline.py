"""ecg_pipeline.py - end-to-end: signal -> R-peaks -> 4-class beat labels (N/S/V/Q) -> arrhythmia types."""
import os, random, time
from collections import deque
import numpy as np, pandas as pd, joblib

from ecg_features import clean_signal, get_features, FEATURES, BEAT_SYMS, FS, SEED, LABEL_MAP
from pan_tompkins import pan_tompkins, match_peaks
from arrhythmia_rules import analyse_window, analyse_stream, CFG, INSUFFICIENT, POOR_QUALITY

INV = np.array(['N', 'S', 'V', 'Q'])


# ----------------------------------------------------------------------------- model / beats
def load_bundle(path='ecg_improved_v3.pkl'):
    b = joblib.load(path)
    b['cols'] = [FEATURES.index(n) for n in b['features']]
    b['q_threshold'] = 0.75
    return b

def decide(P, thr_s, thr_q=0.75):
    """Call S if P(S)>=thr_s; call Q only if Q is argmax AND P(Q)>=thr_q; otherwise argmax over {N, V}."""
    Pn = P.copy()
    if Pn.shape[1] == 4:
        # Base choice between N (0) and V (2)
        pred = np.where(P[:, 2] > P[:, 0], 2, 0)
        # Call S if above tuned S threshold (and higher than V)
        is_s = (P[:, 1] >= thr_s) & (P[:, 1] >= P[:, 2])
        pred[is_s] = 1
        # Call Q ONLY if Q exceeds thr_q AND beats N, S, V
        is_q = (P[:, 3] >= thr_q) & (P[:, 3] == P.max(axis=1))
        pred[is_q] = 3
        return pred
    Pn[:, 1] = -1
    pred = Pn.argmax(1); pred[P[:, 1] >= thr_s] = 1
    return pred

def labels_from_probs(P, tV, tS, tQ=0.75):
    """V if P(V)>=tV, S if P(S)>=tS (higher probability wins if both), Q only if P(Q)>=tQ and argmax, else N."""
    lab = np.full(len(P), 'N', dtype='<U1')
    if P.shape[1] == 4:
        is_q = (P[:, 3] >= tQ) & (P[:, 3] == P.max(axis=1))
        lab[is_q] = 'Q'
    s, v = P[:, 1] >= tS, P[:, 2] >= tV
    lab[s] = 'S'; lab[v] = 'V'; both = s & v
    if both.any(): lab[both] = np.where(P[both, 1] > P[both, 2], 'S', 'V')
    return lab

def classify_beats(sig_clean, peaks, bundle):
    """Features at given peak positions -> (labels array of 'N'/'S'/'V'/'Q', probability matrix)."""
    n_cls = len(bundle.get('label_map', LABEL_MAP))
    if len(peaks) < 12:
        return np.array([], dtype='<U1'), np.zeros((0, n_cls))
    X = get_features(sig_clean, np.asarray(peaks))[:, bundle['cols']]
    P = bundle['model'].predict_proba(X)
    return INV[decide(P, bundle['s_threshold'], bundle.get('q_threshold', 0.50))], P


# ----------------------------------------------------------------------------- records
def select_records(data_dir, max_records=1005):
    hea = sorted(os.path.join(r, f) for r, _, fs in os.walk(data_dir) for f in fs if f.endswith('.hea'))
    random.Random(SEED).shuffle(hea); return hea[:max_records]

def read_record(hp):
    import wfdb
    p = hp[:-4]; rec = wfdb.rdrecord(p); ann = wfdb.rdann(p, 'atr')
    sig = rec.p_signal[:, 0]
    if np.any(np.isnan(sig)): return None, None, None
    sym, smp = np.array(ann.symbol), np.asarray(ann.sample)
    k = np.isin(sym, list(BEAT_SYMS)); return sig, smp[k], sym[k]


# ----------------------------------------------------------------------------- offline evaluation
def eval_record(hp, bundle, win_s=30.0, step_s=30.0, cfg=CFG):
    sig, smp, truth = read_record(hp)
    if sig is None or len(smp) < 50: return None
    sc = clean_signal(sig); t_end = len(sig) / FS
    truth_w = analyse_stream(truth, smp / FS, t_end, win_s, step_s, cfg=cfg)

    det = pan_tompkins(sc, FS); m = match_peaks(smp, det, int(0.15 * FS))
    lab_det, _ = classify_beats(sc, det, bundle)
    lab_ann, _ = classify_beats(sc, smp, bundle)
    if len(lab_det) == 0 or len(lab_ann) == 0: return None

    n_cls = len(LABEL_MAP)
    conf_det = np.zeros((n_cls, n_cls), int); conf_ann = np.zeros((n_cls, n_cls), int)
    for i, j in m['pairs']:
        if truth[i] in LABEL_MAP and lab_det[j] in LABEL_MAP:
            conf_det[LABEL_MAP[truth[i]], LABEL_MAP[lab_det[j]]] += 1
    for t, l in zip(truth, lab_ann):
        if t in LABEL_MAP and l in LABEL_MAP:
            conf_ann[LABEL_MAP[t], LABEL_MAP[l]] += 1

    rows, name = [], os.path.basename(hp)[:-4]
    for variant, lab, t in (('annotated_peaks', lab_ann, smp / FS), ('detected_peaks', lab_det, det / FS)):
        for k, (a, b) in enumerate(zip(truth_w, analyse_stream(lab, t, t_end, win_s, step_s, cfg=cfg))):
            rows.append(dict(variant=variant, record=name, w=k, truth_types=frozenset(a['types']),
                             pred_types=frozenset(b['types']), truth_primary=a['primary_type'],
                             pred_primary=b['primary_type'], hr_true=a['hr_bpm'], hr_pred=b['hr_bpm']))
    return dict(rows=rows, det={k: m[k] for k in ('tp', 'fn', 'fp')}, conf_det=conf_det, conf_ann=conf_ann)

def window_metrics(df):
    """Per-variant, per-arrhythmia-type window-level precision/recall/F1 + primary-type accuracy.
    Excludes windows where ground truth is insufficient_data or poor_quality."""
    df = df[~df.truth_primary.isin({INSUFFICIENT, POOR_QUALITY}) & (df.pred_primary != INSUFFICIENT)]
    out, prim = [], {}
    for v, d in df.groupby('variant'):
        allt = sorted(set().union(*d.truth_types, *d.pred_types) - {POOR_QUALITY})
        for t in allt:
            tr = d.truth_types.map(lambda s: t in s); pr = d.pred_types.map(lambda s: t in s)
            tp, fp, fn = int((tr & pr).sum()), int((~tr & pr).sum()), int((tr & ~pr).sum())
            p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            out.append(dict(variant=v, type=t, support=int(tr.sum()), tp=tp, fp=fp, fn=fn, precision=p,
                            recall=r, f1=(2 * p * r / (p + r) if p + r else 0.0)))
        prim[v] = float((d.truth_primary == d.pred_primary).mean())
    return pd.DataFrame(out).sort_values(['variant', 'support'], ascending=[True, False]), prim


# ----------------------------------------------------------------------------- streaming
class StreamingArrhythmiaDetector:
    def __init__(self, bundle, fs=FS, buf_s=30, hop_s=1.0, warm_s=5.0, margin_s=1.5, win_s=30, cfg=CFG):
        self.b, self.fs, self.cfg, self.win_s = bundle, fs, cfg, win_s
        self.buf_n, self.hop_n = int(buf_s * fs), int(hop_s * fs)
        self.warm, self.margin = int(warm_s * fs), int(margin_s * fs)
        self.buf = np.zeros(0, np.float32); self.start = 0; self.since = 0; self.last = -10 ** 9
        self.hist = deque(); self.compute_s = []

    def push(self, chunk):
        chunk = np.asarray(chunk, np.float32)
        self.buf = np.concatenate([self.buf, chunk]); self.since += len(chunk)
        if len(self.buf) > self.buf_n:
            cut = len(self.buf) - self.buf_n; self.buf = self.buf[cut:]; self.start += cut
        if self.since < self.hop_n or len(self.buf) < int(12 * self.fs): return None
        self.since = 0; t0 = time.perf_counter()
        out = self._process(); self.compute_s.append(time.perf_counter() - t0); return out

    def _process(self):
        sc = clean_signal(self.buf, self.fs); pk = pan_tompkins(sc, self.fs)
        labs, P = classify_beats(sc, pk, self.b)
        if len(labs) == 0: return None
        lo = self.warm if self.start > 0 else 0; hi = len(self.buf) - self.margin
        gate = self.last + int(0.2 * self.fs); new = []
        for p, l, pr in zip(pk, labs, P):
            a = int(p) + self.start
            if lo <= p < hi and a > gate:
                new.append(dict(sample=a, t=a / self.fs, label=str(l), probs=pr))
        if new:
            self.last = new[-1]['sample']
            for b in new: self.hist.append(b)
        while self.hist and self.hist[0]['t'] < self.hist[-1]['t'] - self.win_s: self.hist.popleft()
        win = analyse_window([h['label'] for h in self.hist], [h['t'] for h in self.hist],
                             np.array([h['probs'] for h in self.hist]) if self.hist else None, self.cfg)
        return dict(new_beats=new, window=win)

def simulate_stream(sig_raw, bundle, chunk_s=0.5, max_s=600, **kw):
    det = StreamingArrhythmiaDetector(bundle, **kw); n = int(chunk_s * FS)
    sig_raw = np.asarray(sig_raw)[:int(max_s * FS)]; beats, win = [], None
    for i in range(0, len(sig_raw), n):
        r = det.push(sig_raw[i:i + n])
        if r: beats += r['new_beats']; win = r['window']
    cs = np.array(det.compute_s) if det.compute_s else np.array([np.nan])
    return (pd.DataFrame(beats), win,
            dict(hops=len(cs), mean_s=float(cs.mean()), p95_s=float(np.nanpercentile(cs, 95)), max_s=float(cs.max()),
                 latency_s=(det.margin + det.hop_n) / det.fs + float(cs.mean())))


# ----------------------------------------------------------------------------- cached probabilities + threshold tuning
V_TYPES = ['isolated_PVC', 'PVC_couplet', 'PVC_triplet', 'NSVT', 'AIVR', 'sustained_VT', 'ventricular_bigeminy',
           'ventricular_trigeminy', 'ventricular_quadrigeminy', 'frequent_PVC']
S_TYPES = ['isolated_PAC', 'PAC_couplet', 'PAC_triplet', 'SVT_run', 'sustained_SVT', 'atrial_run_slow',
           'atrial_bigeminy', 'atrial_trigeminy', 'atrial_quadrigeminy', 'frequent_PAC']

def precompute_record(hp, bundle, win_s=30.0, cfg=CFG):
    sig, smp, truth = read_record(hp)
    if sig is None or len(smp) < 50: return None
    sc = clean_signal(sig); det = pan_tompkins(sc, FS)
    if len(det) < 12: return None
    _, P_det = classify_beats(sc, det, bundle); _, P_ann = classify_beats(sc, smp, bundle)
    t_end = len(sig) / FS; name = os.path.basename(hp)[:-4]
    return dict(name=name, pid=name.split('_')[0], t_end=t_end, t_ann=smp / FS, truth=truth, t_det=det / FS,
                P_det=P_det, P_ann=P_ann, det=match_peaks(smp, det, int(0.15 * FS)),
                truth_w=analyse_stream(truth, smp / FS, t_end, win_s, win_s, cfg=cfg))

def labels_from_probs(P, tV, tS, tQ=0.50):
    """V if P(V)>=tV, S if P(S)>=tS (higher probability wins if both), Q if P(Q)>=tQ, otherwise N."""
    lab = np.full(len(P), 'N', dtype='<U1')
    s, v = P[:, 1] >= tS, P[:, 2] >= tV
    lab[s] = 'S'; lab[v] = 'V'; both = s & v
    if both.any(): lab[both] = np.where(P[both, 1] > P[both, 2], 'S', 'V')
    if P.shape[1] == 4:
        lab[P[:, 3] >= tQ] = 'Q'
    return lab

def default_labels(P, bundle):
    return INV[decide(P, bundle['s_threshold'], bundle.get('q_threshold', 0.50))]

def window_rows(c, labels, variant, win_s=30.0, cfg=CFG):
    pw = analyse_stream(labels, c['t_det'], c['t_end'], win_s, win_s, cfg=cfg)
    return [dict(variant=variant, record=c['name'], w=k, truth_types=frozenset(a['types']),
                 pred_types=frozenset(b['types']), truth_primary=a['primary_type'], pred_primary=b['primary_type'])
            for k, (a, b) in enumerate(zip(c['truth_w'], pw))]

def micro_prf(df, types):
    df = df[~df.truth_primary.isin({INSUFFICIENT, POOR_QUALITY})]
    tp = fp = fn = 0
    for t in types:
        tr = df.truth_types.map(lambda s: t in s); pr = df.pred_types.map(lambda s: t in s)
        tp += int((tr & pr).sum()); fp += int((~tr & pr).sum()); fn += int((tr & ~pr).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp + fn

def sweep_threshold(cache, which, grid, other_thr=0.5, tQ=0.50, win_s=30.0, cfg=CFG):
    out = []
    for t in grid:
        rows = []
        for c in cache:
            tV, tS = (t, other_thr) if which == 'V' else (other_thr, t)
            rows += window_rows(c, labels_from_probs(c['P_det'], tV, tS, tQ), 'x', win_s, cfg=cfg)
        p, r, f, sup = micro_prf(pd.DataFrame(rows), V_TYPES if which == 'V' else S_TYPES)
        out.append(dict(thr=t, precision=p, recall=r, f1=f, support=sup))
    return pd.DataFrame(out)

def best_threshold(sw):
    if sw.f1.max() <= 0: return 0.5
    return float(sw.loc[sw.f1 == sw.f1.max(), 'thr'].max())