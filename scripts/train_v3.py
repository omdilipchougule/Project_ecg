"""train_v3.py - 4-class (N/S/V/Q) training, strategy & causal ablations, S-FP diagnostic, and OOF tuning."""
import numpy as np, pandas as pd, joblib
from collections import Counter
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             balanced_accuracy_score, average_precision_score, precision_recall_curve)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler

from ecg_features import (DATA_DIR, FS, SEED, LABEL_MAP, FEATURES, NON_CAUSAL,
                          Timer, load_data)

# Best hyperparameters found in v2 random search
BEST_PARAMS = dict(
    colsample_bytree=0.75, learning_rate=0.03, max_depth=8, min_child_samples=100,
    num_leaves=255, reg_alpha=0.244, reg_lambda=0.355, subsample=0.64, subsample_freq=1
)


def build_train(X, y, st):
    sw = None
    if st.get('under'):
        cnt = Counter(y.tolist())
        tgt = int(st['under'] * max(cnt[1], cnt[2]))
        if tgt < cnt[0]:
            strat = {0: tgt, 1: cnt[1], 2: cnt[2], 3: min(cnt[3], tgt)}
            X, y = RandomUnderSampler(sampling_strategy=strat, random_state=SEED).fit_resample(X, y)
    if st.get('weight'):
        classes = np.unique(y)
        w = compute_class_weight('balanced', classes=classes, y=y) ** st['weight']
        sw = w[y]
    return X, y, sw


def fit_predict(Xtr, ytr, Xva, yva, st, params, cols):
    Xt, yt, sw = build_train(Xtr[:, cols], ytr, st)
    clf = LGBMClassifier(objective='multiclass', n_estimators=2000, random_state=SEED,
                         n_jobs=-1, verbose=-1, **params)
    clf.fit(Xt, yt, sample_weight=sw, eval_set=[(Xva[:, cols], yva)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return clf, clf.predict_proba(Xva[:, cols])


def decide(P, thr_s, thr_q=0.50):
    Pn = P.copy(); Pn[:, 1] = -1; Pn[:, 3] = -1
    pred = Pn.argmax(1)
    pred[P[:, 1] >= thr_s] = 1
    pred[P[:, 3] >= thr_q] = 3
    return pred


def tune_thr(P, y, cls_idx=1):
    p, r, th = precision_recall_curve(y == cls_idx, P[:, cls_idx]); p, r = p[:-1], r[:-1]
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(th[f1.argmax()])


def evaluate(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2, 3], zero_division=0)
    return dict(P_N=p[0], R_N=r[0], F1_N=f[0],
                P_S=p[1], R_S=r[1], F1_S=f[1],
                P_V=p[2], R_V=r[2], F1_V=f[2],
                P_Q=p[3], R_Q=r[3], F1_Q=f[3],
                macroF1_NSV=f[:3].mean(), macroF1_all=f.mean(),
                balAcc=balanced_accuracy_score(y, pred))


if __name__ == '__main__':
    with Timer("load+features_v3"):
        X, y, groups = load_data(DATA_DIR)
    cc = Counter(y.tolist())
    print(f"{len(y):,} beats | patients: {len(set(groups))} | "
          f"N={cc[0]:,} S={cc[1]:,} V={cc[2]:,} Q={cc[3]:,}")

    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    dev_idx, test_idx = next(sgkf.split(X, y, groups))
    assert not (set(groups[dev_idx]) & set(groups[test_idx]))
    np.save("test_patients.npy", np.unique(groups[test_idx]))

    Xd, yd, gd = X[dev_idx], y[dev_idx], groups[dev_idx]
    Xte, yte   = X[test_idx], y[test_idx]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1).split(Xd, yd, gd))
    tr_i, va_i = folds[0]

    ix = lambda names: [FEATURES.index(n) for n in names]
    ALL = list(range(len(FEATURES)))
    SHOW = ['name', 'AP_S', 'thr_S', 'P_S', 'R_S', 'F1_S', 'F1_V', 'F1_Q', 'macroF1_NSV', 'iters']

    def run_exp(name, st, params, cols):
        with Timer(name):
            clf, P = fit_predict(Xd[tr_i], yd[tr_i], Xd[va_i], yd[va_i], st, params, cols)
        thr_s = tune_thr(P, yd[va_i], 1); thr_q = tune_thr(P, yd[va_i], 3)
        m = evaluate(yd[va_i], decide(P, thr_s, thr_q))
        m.update(name=name, AP_S=average_precision_score(yd[va_i] == 1, P[:, 1]),
                 thr_S=thr_s, thr_Q=thr_q, iters=clf.best_iteration_)
        return m, clf, P

    # 1. Strategy Comparison (fast, leakage-free)
    STRATS = {
        'none':    {},
        'cw_sqrt': {'weight': 0.5},
        'under10': {'under': 10},
    }
    print("\n--- 1. IMBALANCE STRATEGY COMPARISON ---")
    s_rows = [run_exp(k, st, BEST_PARAMS, ALL)[0] for k, st in STRATS.items()]
    df_s = pd.DataFrame(s_rows)[SHOW]; print(df_s.round(4).to_string(index=False))
    BEST_STRAT = df_s.sort_values('F1_S', ascending=False).iloc[0]['name']
    print("Selected strategy:", BEST_STRAT)

    # 2. Feature & Causal Ablation
    v2_morph = ['r_amp', 'qrs_width_ms', 'rr_prev', 'rr_next', 'rr_med', 'prev_over_med',
                'prev_over_next', 'pause_ratio', 'p_energy', 'rr_prev2', 'rr_next2',
                'rr_sd', 'next_over_med', 'prev_over_prev2', 'qrs_area', 'max_slope', 'tmpl_corr', 't_energy']
    v2_all   = v2_morph + ['noise', 'baseline_off', 'amp_ratio']
    v3_causal = [f for f in FEATURES if f not in NON_CAUSAL]
    FSETS = {
        'v2_+morph(no_qual)': ix(v2_morph),
        'v2_all(+quality)':   ix(v2_all),
        'v3_all(+long_rr)':   ALL,
        'v3_causal(no_next)': ix(v3_causal),
    }
    print("\n--- 2. FEATURE & CAUSAL ABLATION ---")
    f_rows = [run_exp(k, STRATS[BEST_STRAT], BEST_PARAMS, cols)[0] for k, cols in FSETS.items()]
    df_f = pd.DataFrame(f_rows)[SHOW]; print(df_f.round(4).to_string(index=False))

    # 3. 5-Fold OOF Threshold Tuning for v3_all
    print("\n--- 3. 5-FOLD OOF TRAINING (v3_all) ---")
    oof = np.zeros((len(yd), 4), dtype=np.float32); iters = []
    for k, (tr, va) in enumerate(folds, 1):
        with Timer(f"fold{k}"):
            clf, P = fit_predict(Xd[tr], yd[tr], Xd[va], yd[va], STRATS[BEST_STRAT], BEST_PARAMS, ALL)
        oof[va] = P; iters.append(clf.best_iteration_)
    THR_S = tune_thr(oof, yd, 1); THR_Q = tune_thr(oof, yd, 3)
    print(f"OOF Tuned Thresholds: THR_S={THR_S:.4f}, THR_Q={THR_Q:.4f}")
    print("OOF Metrics:", {k: round(v, 4) for k, v in evaluate(yd, decide(oof, THR_S, THR_Q)).items()})
    print("OOF Confusion Matrix (N, S, V, Q):\n", confusion_matrix(yd, decide(oof, THR_S, THR_Q)))

    # 4. Final Test Evaluation & Bundle Export (Full v3 + Causal v3)
    Xt_, yt_, sw_ = build_train(Xd, yd, STRATS[BEST_STRAT])
    n_est = int(np.median(iters) * 1.1)
    final_clf = LGBMClassifier(objective='multiclass', n_estimators=n_est, random_state=SEED,
                               n_jobs=-1, verbose=-1, **BEST_PARAMS).fit(Xt_, yt_, sample_weight=sw_)
    Pte = final_clf.predict_proba(Xte); pred_te = decide(Pte, THR_S, THR_Q)
    print("\n--- 4. FINAL HELD-OUT TEST EVALUATION (v3_all) ---")
    print("TEST Metrics:", {k: round(v, 4) for k, v in evaluate(yte, pred_te).items()})
    print("TEST Confusion Matrix (rows=truth N,S,V,Q; cols=pred N,S,V,Q):\n", confusion_matrix(yte, pred_te))

    joblib.dump(dict(model=final_clf, label_map=LABEL_MAP, s_threshold=THR_S, q_threshold=THR_Q,
                     features=FEATURES, strategy=STRATS[BEST_STRAT], params=BEST_PARAMS, fs=FS),
                'ecg_improved_v3.pkl')
    print("Saved -> ecg_improved_v3.pkl")

    # 5. S False-Positive Diagnostic on Test Set
    print("\n--- 5. S FALSE-POSITIVE DIAGNOSTIC (Medians) ---")
    fp_s = (yte == 0) & (pred_te == 1)
    tp_s = (yte == 1) & (pred_te == 1)
    tn_n = (yte == 0) & (pred_te == 0)
    diag = pd.DataFrame({
        'True_N (TN)':      np.nanmedian(Xte[tn_n], axis=0),
        'False_S (N->S)':   np.nanmedian(Xte[fp_s], axis=0),
        'True_S (TP)':      np.nanmedian(Xte[tp_s], axis=0),
    }, index=FEATURES)
    print(diag.round(3).to_string())

    # 6. Train & Save Causal Model (without rr_next) for Zero-Lookahead Comparison
    c_cols = ix(v3_causal)
    Xtc, ytc, swc = build_train(Xd[:, c_cols], yd, STRATS[BEST_STRAT])
    causal_clf = LGBMClassifier(objective='multiclass', n_estimators=n_est, random_state=SEED,
                                n_jobs=-1, verbose=-1, **BEST_PARAMS).fit(Xtc, ytc, sample_weight=swc)
    joblib.dump(dict(model=causal_clf, label_map=LABEL_MAP, s_threshold=THR_S, q_threshold=THR_Q,
                     features=v3_causal, strategy=STRATS[BEST_STRAT], params=BEST_PARAMS, fs=FS),
                'ecg_improved_v3_causal.pkl')
    print("Saved -> ecg_improved_v3_causal.pkl")