"""arrhythmia_rules.py - rule layer: per-beat labels (N/S/V/Q) + beat times -> arrhythmia types.

Handles 'Q' (noise/undefined) beats safely without creating false pauses, and flags Q-heavy
windows as 'poor_quality'.
"""
import numpy as np

CFG = dict(
    fast_bpm=100.0,        # run of >=3 ectopic beats at/above this rate = tachycardia run (NSVT / SVT)
    sustained_s=30.0,      # a fast run lasting >= this = sustained VT / SVT
    min_cycles=3,          # ectopic beats needed to call bi-/tri-/quadrigeminy
    frequent_burden=0.10,  # ectopic fraction of valid beats in window for "frequent PVC/PAC"
    min_beats=20,          # min valid beats in window for burden-based calls
    brady_bpm=60.0, tachy_bpm=100.0, pause_s=2.0,
    irr_nrmssd=0.12,       # normalised RMSSD of N-N intervals for "irregular rhythm suspected"
    irr_min_nn=20,         # min N-N intervals needed to assess irregularity
    min_event_conf=0.0,    # if probs given: drop runs of >=3 ectopic beats with mean P(class) below this
    max_q_frac=0.35,       # max fraction of 'Q' beats allowed before window is flagged as poor_quality
)

COL = {'N': 0, 'S': 1, 'V': 2, 'Q': 3}
NAMES = {
    'V': dict(single='isolated_PVC', couplet='PVC_couplet', triplet='PVC_triplet', run_fast='NSVT',
              run_sustained='sustained_VT', run_slow='AIVR', bigeminy='ventricular_bigeminy',
              trigeminy='ventricular_trigeminy', quadrigeminy='ventricular_quadrigeminy'),
    'S': dict(single='isolated_PAC', couplet='PAC_couplet', triplet='PAC_triplet', run_fast='SVT_run',
              run_sustained='sustained_SVT', run_slow='atrial_run_slow', bigeminy='atrial_bigeminy',
              trigeminy='atrial_trigeminy', quadrigeminy='atrial_quadrigeminy'),
}
PRIORITY = ['sustained_VT', 'NSVT', 'AIVR', 'sustained_SVT', 'SVT_run', 'atrial_run_slow',
            'ventricular_bigeminy', 'ventricular_trigeminy', 'ventricular_quadrigeminy',
            'PVC_triplet', 'PVC_couplet', 'atrial_bigeminy', 'atrial_trigeminy', 'atrial_quadrigeminy',
            'PAC_triplet', 'PAC_couplet', 'frequent_PVC', 'frequent_PAC', 'irregular_rhythm_suspected',
            'pause', 'isolated_PVC', 'isolated_PAC', 'tachycardia', 'bradycardia']
NONE_TYPE    = 'no_arrhythmia_detected'
INSUFFICIENT = 'insufficient_data'
POOR_QUALITY = 'poor_quality'


def _runs(mask):
    """(start, end_inclusive) for each run of consecutive True."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    br = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]))


def _mk(typ, ect, idx, times, probs):
    idx = np.asarray(idx)
    t0, t1 = float(times[idx[0]]), float(times[idx[-1]])
    rate = 60.0 * (len(idx) - 1) / (t1 - t0) if len(idx) > 1 and t1 > t0 else float('nan')
    conf = float(np.mean(probs[idx, COL[ect]])) if probs is not None else None
    return dict(type=typ, ectopic=ect, i0=int(idx[0]), i1=int(idx[-1]), n_ectopic=int(len(idx)),
                t0=t0, t1=t1, rate_bpm=rate, confidence=conf)


def detect_ectopy(labels, times, ect='V', probs=None, cfg=CFG):
    """Patterns formed by ectopic beats of one class ('V' or 'S')."""
    lab = np.asarray(labels); times = np.asarray(times, float); nm = NAMES[ect]
    ev, singles = [], []
    for s, e in _runs(lab == ect):
        n = e - s + 1
        if n == 1:
            singles.append(s); continue
        dur = times[e] - times[s]
        rate = 60.0 * (n - 1) / dur if dur > 0 else np.inf
        if n == 2:
            typ = nm['couplet']
        elif rate >= cfg['fast_bpm']:
            typ = nm['run_sustained'] if dur >= cfg['sustained_s'] else nm['run_fast']
        elif n == 3:
            typ = nm['triplet']
        else:
            typ = nm['run_slow']
        e_ = _mk(typ, ect, np.arange(s, e + 1), times, probs)
        if probs is not None and n >= 3 and e_['confidence'] < cfg['min_event_conf']:
            continue
        ev.append(e_)
    sg = np.array(singles, dtype=int); i = 0
    kind = {2: 'bigeminy', 3: 'trigeminy', 4: 'quadrigeminy'}
    while i < len(sg):
        chain, g = [sg[i]], (sg[i + 1] - sg[i]) if i + 1 < len(sg) else None
        if g in kind:
            k = i
            while k + 1 < len(sg) and sg[k + 1] - sg[k] == g and np.all(np.isin(lab[sg[k] + 1:sg[k + 1]], ['N', 'Q'])):
                chain.append(sg[k + 1]); k += 1
        if len(chain) >= cfg['min_cycles']:
            ev.append(_mk(nm[kind[g]], ect, chain, times, probs)); i += len(chain)
        else:
            ev.append(_mk(nm['single'], ect, [sg[i]], times, probs)); i += 1
    return sorted(ev, key=lambda x: x['i0'])


def rhythm_flags(labels, times, cfg=CFG):
    """Heart-rate and regularity flags from valid (non-Q) adjacent RR intervals."""
    lab = np.asarray(labels); t = np.asarray(times, float); flags, info = [], {}
    if len(t) < 3:
        return flags, info
    rr = np.diff(t)
    valid = (lab[:-1] != 'Q') & (lab[1:] != 'Q')
    if valid.sum() < 2:
        return flags, info
    rr_clean = rr[valid]
    hr = 60.0 / np.median(rr_clean); info['hr_bpm'] = float(hr)
    if hr < cfg['brady_bpm']: flags.append('bradycardia')
    if hr > cfg['tachy_bpm']: flags.append('tachycardia')
    if rr_clean.max() >= cfg['pause_s']: flags.append('pause')
    rn = rr[(lab[:-1] == 'N') & (lab[1:] == 'N')]
    if len(rn) >= cfg['irr_min_nn']:
        nr = float(np.sqrt(np.mean(np.diff(rn) ** 2)) / np.mean(rn)); info['nrmssd'] = nr
        if nr >= cfg['irr_nrmssd']: flags.append('irregular_rhythm_suspected')
    return flags, info


def analyse_window(labels, times, probs=None, cfg=CFG):
    """One window of beats -> dict(primary_type, types, events, burden, hr...)."""
    lab = np.asarray(labels); times = np.asarray(times, float); n = len(lab)
    if n < 5:
        return dict(n_beats=n, primary_type=INSUFFICIENT, types=[], events=[], hr_bpm=float('nan'),
                    burden_V=float('nan'), burden_S=float('nan'), burden_Q=float('nan'))
    bQ = float(np.mean(lab == 'Q'))
    n_valid = int(np.sum(lab != 'Q'))
    if bQ > cfg.get('max_q_frac', 0.20) or n_valid < 5:
        return dict(n_beats=n, primary_type=POOR_QUALITY, types=[POOR_QUALITY], events=[],
                    hr_bpm=float('nan'), nrmssd=float('nan'),
                    burden_V=float('nan'), burden_S=float('nan'), burden_Q=bQ)
    events = detect_ectopy(lab, times, 'V', probs, cfg) + detect_ectopy(lab, times, 'S', probs, cfg)
    flags, info = rhythm_flags(lab, times, cfg)
    bV = float(np.sum(lab == 'V') / max(n_valid, 1))
    bS = float(np.sum(lab == 'S') / max(n_valid, 1))
    types = {e['type'] for e in events} | set(flags)
    if n_valid >= cfg['min_beats']:
        if bV >= cfg['frequent_burden']: types.add('frequent_PVC')
        if bS >= cfg['frequent_burden']: types.add('frequent_PAC')
    primary = next((p for p in PRIORITY if p in types), NONE_TYPE)
    return dict(n_beats=n, primary_type=primary, types=sorted(types), events=events,
                hr_bpm=info.get('hr_bpm', float('nan')), nrmssd=info.get('nrmssd', float('nan')),
                burden_V=bV, burden_S=bS, burden_Q=bQ)


def analyse_stream(labels, times, t_end, win_s=30.0, step_s=30.0, probs=None, cfg=CFG):
    """Fixed windows over a whole recording (same windows for truth and predictions)."""
    lab = np.asarray(labels); times = np.asarray(times, float); out = []
    for t0 in np.arange(0.0, t_end - win_s + 1e-9, step_s):
        i0, i1 = np.searchsorted(times, t0, 'left'), np.searchsorted(times, t0 + win_s, 'left')
        r = analyse_window(lab[i0:i1], times[i0:i1], None if probs is None else probs[i0:i1], cfg)
        r.update(t0=float(t0), t1=float(t0 + win_s)); out.append(r)
    return out