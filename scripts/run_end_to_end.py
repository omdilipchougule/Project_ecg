"""run_end_to_end.py - 4-class end-to-end evaluation + ectopy-rich streaming comparison (buffered vs causal)."""
import os, time, numpy as np, pandas as pd
import ecg_pipeline as ep
from ecg_features import FS, LABEL_MAP
from pan_tompkins import match_peaks

DATA_DIR      = r"D:\megaproject\icentia11k"
MODEL         = 'ecg_improved_v3.pkl'
CAUSAL_MODEL  = 'ecg_improved_v3_causal.pkl'
TEST_PATIENTS = 'test_patients.npy'
N_REC         = 40
WIN_S         = 30.0


def beat_report(c, title):
    labs = list(LABEL_MAP.keys())
    p = np.diag(c) / np.maximum(c.sum(0), 1); r = np.diag(c) / np.maximum(c.sum(1), 1)
    print(f"\n{title}\n", pd.DataFrame(c, index=labs, columns=labs))
    print(pd.DataFrame({'precision': p, 'recall': r, 'F1': 2*p*r/np.maximum(p+r, 1e-12)}, index=labs).round(3))


if __name__ == '__main__':
    bundle = ep.load_bundle(MODEL)
    test_pids = set(np.load(TEST_PATIENTS, allow_pickle=True).tolist())
    recs = [h for h in ep.select_records(DATA_DIR) if os.path.basename(h).split('_')[0] in test_pids][:N_REC]
    print(f"{len(recs)} test records | S thr={bundle['s_threshold']:.3f} | Q thr={bundle.get('q_threshold', 0.5):.3f}")

    n_cls = len(LABEL_MAP)
    rows, det_tot = [], dict(tp=0, fn=0, fp=0)
    conf_det, conf_ann = np.zeros((n_cls, n_cls), int), np.zeros((n_cls, n_cls), int)

    for k, h in enumerate(recs, 1):
        t0 = time.perf_counter(); r = ep.eval_record(h, bundle, WIN_S, WIN_S)
        if r is None: continue
        rows += r['rows']; conf_det += r['conf_det']; conf_ann += r['conf_ann']
        for key in det_tot: det_tot[key] += r['det'][key]
        print(f"  [{k}/{len(recs)}] {time.perf_counter()-t0:.1f}s")

    df = pd.DataFrame(rows)
    se = det_tot['tp'] / (det_tot['tp'] + det_tot['fn']); ppv = det_tot['tp'] / (det_tot['tp'] + det_tot['fp'])
    print(f"\nR-peak detection (150 ms tol): Se={se:.4f} PPV={ppv:.4f} {det_tot}")

    beat_report(conf_ann, "BEATS - annotated peaks + classifier (classifier error only)")
    beat_report(conf_det, "BEATS - Pan-Tompkins peaks + classifier (full pipeline, matched beats)")

    tab, prim = ep.window_metrics(df)
    print("\nWINDOW-LEVEL arrhythmia TYPE metrics (excluding poor_quality truth windows):")
    print(tab.round(3).to_string(index=False))
    print("\nPrimary-type accuracy per variant:", {k: round(v, 3) for k, v in prim.items()})
    tab.to_csv('window_metrics.csv', index=False); df.to_pickle('window_rows.pkl')

    # Step 5: Select test record with the most V+S ectopy in the first 600s for streaming test
    MAX_S = 600
    best_h, best_ect = recs[0], -1
    for h in recs:
        _, smp_, truth_ = ep.read_record(h)
        if smp_ is None: continue
        ect_cnt = int(np.isin(truth_[smp_ < MAX_S * FS], ['S', 'V']).sum())
        if ect_cnt > best_ect:
            best_ect, best_h = ect_cnt, h

    sig, smp, truth = ep.read_record(best_h)
    m_ = smp < MAX_S * FS; ref, ref_lab = smp[m_], truth[m_]
    print(f"\n--- STREAMING CHECK on {os.path.basename(best_h)[:-4]} ({best_ect} ectopic beats in first {MAX_S}s) ---")

    for tag, b_path, marg in [('Buffered (1.5s margin, with rr_next)', MODEL, 1.5),
                              ('Strictly Causal (0.3s margin, no rr_next)', CAUSAL_MODEL, 0.3)]:
        if not os.path.exists(b_path): continue
        b_ = ep.load_bundle(b_path)
        beats, win, stats = ep.simulate_stream(sig, b_, chunk_s=0.5, max_s=MAX_S, margin_s=marg)
        mm = match_peaks(ref, beats['sample'].values, int(0.15 * FS))
        cs = np.zeros((n_cls, n_cls), int)
        for i, j in mm['pairs']:
            if ref_lab[i] in LABEL_MAP and beats['label'].iloc[j] in LABEL_MAP:
                cs[LABEL_MAP[ref_lab[i]], LABEL_MAP[beats['label'].iloc[j]]] += 1
        print(f"\n[{tag}] Peak Se={mm['se']:.4f} PPV={mm['ppv']:.4f} | Stats={ {k: round(v, 3) for k, v in stats.items()} }")
        beat_report(cs, f"STREAMING BEATS - {tag}")