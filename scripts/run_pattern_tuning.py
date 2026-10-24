"""run_pattern_tuning.py - AF irr_nrmssd check + extended threshold grid (0.3..0.995) cross-fitted tuning."""
import os, time, numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import precision_recall_curve
import ecg_pipeline as ep
from ecg_features import FS, SEED
from arrhythmia_rules import CFG, rhythm_flags

DATA_DIR      = r"D:\megaproject\icentia11k"
MODEL         = 'ecg_improved_v3.pkl'
TEST_PATIENTS = 'test_patients.npy'
N_REC         = 60
GRID          = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)

if __name__ == '__main__':
    bundle = ep.load_bundle(MODEL)
    test_pids = set(np.load(TEST_PATIENTS, allow_pickle=True).tolist())
    all_recs = ep.select_records(DATA_DIR)
    dev_recs = [h for h in all_recs if os.path.basename(h).split('_')[0] not in test_pids][:150]
    recs     = [h for h in all_recs if os.path.basename(h).split('_')[0] in test_pids][:N_REC]

    # Step 4: Check & Tune irr_nrmssd on Dev Records with AF annotations (if present)
    import wfdb
    nrmssd_vals, is_af_win = [], []
    for h in dev_recs:
        try:
            ann = wfdb.rdann(h[:-4], 'atr')
        except Exception:
            continue
        aux = [s.strip('\x00 ') for s in ann.aux_note]
        if not any('AF' in s for s in aux):
            continue
        rhythm = 'N'; af_beat = np.zeros(len(ann.sample), dtype=bool)
        for i, note in enumerate(aux):
            if note.startswith('('): rhythm = note[1:]
            af_beat[i] = ('AF' in rhythm)
        sym = np.array(ann.symbol); smp = np.asarray(ann.sample) / FS
        for t0 in np.arange(0, smp[-1] - 30.0, 30.0):
            m = (smp >= t0) & (smp < t0 + 30.0)
            if m.sum() < 20: continue
            _, info = rhythm_flags(sym[m], smp[m], CFG)
            if 'nrmssd' in info:
                nrmssd_vals.append(info['nrmssd'])
                is_af_win.append(af_beat[m].mean() >= 0.5)
    if sum(is_af_win) > 0:
        pr, rc, th = precision_recall_curve(is_af_win, nrmssd_vals)
        f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)
        ix_best = f1.argmax()
        CFG['irr_nrmssd'] = float(th[ix_best])
        print(f"[AF Tuning] Found {sum(is_af_win)} AF windows | Tuned irr_nrmssd={CFG['irr_nrmssd']:.4f} (F1={f1[ix_best]:.3f})")
    else:
        print(f"[AF Tuning] No aux_note AF labels found in dev sample; keeping default irr_nrmssd={CFG['irr_nrmssd']}")

    # Precompute test records
    cache = []
    for k, h in enumerate(recs, 1):
        t0 = time.perf_counter(); c = ep.precompute_record(h, bundle, cfg=CFG)
        if c is not None: cache.append(c)
        print(f"[{k}/{len(recs)}] {time.perf_counter()-t0:.1f}s")
    print(f"{len(cache)} records, {len({c['pid'] for c in cache})} patients")

    # Check Q-beat predictions under 4-class model
    cq = Counter(); nq = 0
    for c in cache:
        q = c['truth'] == 'Q'; nq += int(q.sum())
        cq.update(ep.default_labels(c['P_ann'][q], bundle).tolist())
    print("True Q beats on test subset:", nq, "-> predicted as:", dict(cq))

    # Cross-fitted tuning on extended GRID
    pids = sorted({c['pid'] for c in cache}); rng = np.random.default_rng(SEED); rng.shuffle(pids)
    half = {'A': set(pids[:len(pids) // 2]), 'B': set(pids[len(pids) // 2:])}
    report_rows, chosen = [], []
    tQ = bundle.get('q_threshold', 0.50)

    for tune_on, report_on in (('A', 'B'), ('B', 'A')):
        tune = [c for c in cache if c['pid'] in half[tune_on]]
        rep  = [c for c in cache if c['pid'] in half[report_on]]
        sv = ep.sweep_threshold(tune, 'V', GRID, other_thr=bundle['s_threshold'], tQ=tQ, cfg=CFG)
        tV = ep.best_threshold(sv)
        ss = ep.sweep_threshold(tune, 'S', GRID, other_thr=tV, tQ=tQ, cfg=CFG)
        tS = ep.best_threshold(ss)
        print(f"\n--- tuned on half {tune_on} ({len(tune)} records) ---")
        print(f"V sweep:\n{sv.round(3).to_string(index=False)}\nS sweep:\n{ss.round(3).to_string(index=False)}")
        print(f"chosen: tV={tV}, tS={tS}"); chosen.append((tune_on, tV, tS))
        for c in rep:
            report_rows += ep.window_rows(c, ep.default_labels(c['P_det'], bundle), 'default', cfg=CFG)
            report_rows += ep.window_rows(c, ep.labels_from_probs(c['P_det'], tV, tS, tQ), 'tuned', cfg=CFG)

    df = pd.DataFrame(report_rows)
    tab, prim = ep.window_metrics(df)
    print("\nPrimary-type accuracy (cross-fitted, detected peaks):", {k: round(v, 3) for k, v in prim.items()})
    for fam, types in (('PVC patterns', ep.V_TYPES), ('PAC patterns', ep.S_TYPES)):
        print(f"\n{fam} - micro precision / recall / F1 / support")
        for v in ('default', 'tuned'):
            p, r, f, s = ep.micro_prf(df[df.variant == v], types)
            print(f"  {v:8s} P={p:.3f} R={r:.3f} F1={f:.3f} support={s}")
    keep = tab[tab.support >= 10]
    print("\nPer-type (support >= 10 windows):\n", keep.round(3).to_string(index=False))
    tab.to_csv('pattern_tuning_metrics.csv', index=False); df.to_pickle('pattern_tuning_rows.pkl')
    print("\nthresholds chosen per fold:", chosen)