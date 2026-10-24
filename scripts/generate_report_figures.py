"""generate_report_figures.py - Generates 5 publication-quality figures (300 DPI) for the ECG report & slides.
Run: python generate_report_figures.py
Outputs in ./report_figures/:
  fig1_q_leakage_and_cm.png         - v2 vs v3 Q-leakage reduction + 4-class row-normalized confusion matrices
  fig2_window_dumbbell_f1.png       - Annotated vs Detected peaks window F1 dumbbell chart (error propagation)
  fig3_threshold_sweeps.png         - Cross-fitted P/R/F1 curves across the 0.30..0.995 threshold grid
  fig4_ad8232_telemetry_strip.png   - 4-track synchronized AD8232 motion-burst & poor_quality gate timeline
  fig5_streaming_ & _feature_imp.png - Streaming Causal vs Buffered benchmark + Top-15 LightGBM features
"""
import os, numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import ecg_pipeline as ep
from ecg_features import FS, FEATURES
from arrhythmia_rules import analyse_stream, CFG

OUT_DIR = "report_figures"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'
})


# =============================================================================
# FIGURE 1: v2 vs v3 Q-Beat Leakage Reduction + 4-Class Confusion Matrices
# =============================================================================
def plot_fig1_cm_and_leakage():
    cm_ann = np.array([
        [156004, 376, 104, 6588],
        [101,    642,  60,   40],
        [68,      10, 728,   65],
        [18833,  133, 646, 37162]
    ])
    cm_det = np.array([
        [139083, 177,  49, 23692],
        [123,    561,  52,   107],
        [63,      12, 650,   146],
        [12529,   73, 252, 43167]
    ])
    classes = ['N (Normal)', 'S (PAC)', 'V (PVC)', 'Q (Noise)']
    short_c = ['N', 'S', 'V', 'Q']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={'width_ratios': [1.15, 1.15, 0.9]})

    for ax, cm, title in [(axes[0], cm_ann, 'A. Annotated Peaks (Classifier Only)'),
                          (axes[1], cm_det, 'B. Pan-Tompkins Peaks (Full Pipeline)')]:
        row_pct = cm / cm.sum(axis=1, keepdims=True) * 100.0
        annot = np.empty(cm.shape, dtype=object)
        for i in range(4):
            for j in range(4):
                annot[i, j] = f"{row_pct[i, j]:.1f}%\n({cm[i, j]:,})"
        sns.heatmap(row_pct, annot=annot, fmt='', cmap='Blues', vmin=0, vmax=100,
                    xticklabels=short_c, yticklabels=classes if ax == axes[0] else short_c,
                    cbar=False, ax=ax, linewidths=0.8, linecolor='white',
                    annot_kws={'size': 8.5, 'weight': 'bold'})
        ax.set_title(title)
        ax.set_xlabel('Predicted Beat Class')
        if ax == axes[0]:
            ax.set_ylabel('True Beat Class (Ground Truth)')

    # Panel C: Q-beat False Ectopy Leakage (v2 vs v3)
    ax3 = axes[2]
    labels = ['Q -> False V\n(PVC Leakage)', 'Q -> False S\n(PAC Leakage)', 'Total False Ectopy\nfrom Q Beats']
    v2_rates = [7387 / 84097 * 100, 3262 / 84097 * 100, (7387 + 3262) / 84097 * 100]
    v3_rates = [732 / 70489 * 100,  163 / 70489 * 100,  (732 + 163) / 70489 * 100]

    x = np.arange(len(labels)); w = 0.35
    b1 = ax3.bar(x - w/2, v2_rates, w, label='v2 (3-Class: N/S/V)', color='#e74c3c', alpha=0.88)
    b2 = ax3.bar(x + w/2, v3_rates, w, label='v3 (4-Class: N/S/V/Q)', color='#2ecc71', alpha=0.92)
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2, h + 0.25, f"{h:.2f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax3.set_xticks(x); ax3.set_xticklabels(labels)
    ax3.set_ylabel('% of True Q Beats Misclassified as Ectopic')
    ax3.set_title('C. 90% Reduction in Noise -> Ectopy Leakage')
    ax3.set_ylim(0, 15); ax3.grid(axis='y', alpha=0.3); ax3.legend(loc='upper left')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_q_leakage_and_cm.png')
    plt.savefig(path); plt.close()
    print("Saved:", path)


# =============================================================================
# FIGURE 2: Dumbbell Plot - Window-Level F1 (Annotated Ceiling vs Full Pipeline)
# =============================================================================
def plot_fig2_window_dumbbell():
    if os.path.exists('window_metrics.csv'):
        df = pd.read_csv('window_metrics.csv')
        df = df[df['support'] >= 9]
        piv = df.pivot(index='type', columns='variant', values=['f1', 'support'])
        types = piv.index.tolist()
        f1_ann = piv[('f1', 'annotated_peaks')].values
        f1_det = piv[('f1', 'detected_peaks')].values
        sup    = piv[('support', 'annotated_peaks')].values.astype(int)
    else:
        data = [
            ('bradycardia', 1019, 0.991, 0.955),
            ('tachycardia', 402, 0.955, 0.940),
            ('frequent_PVC', 62, 0.866, 0.810),
            ('frequent_PAC', 124, 0.802, 0.757),
            ('isolated_PVC', 333, 0.804, 0.702),
            ('isolated_PAC', 231, 0.671, 0.682),
            ('ventricular_bigeminy', 11, 0.667, 0.609),
            ('irregular_rhythm_suspected', 278, 0.742, 0.556),
            ('PAC_couplet', 72, 0.500, 0.408),
        ]
        types  = [d[0] for d in data]; sup = np.array([d[1] for d in data])
        f1_ann = np.array([d[2] for d in data]); f1_det = np.array([d[3] for d in data])

    order = np.argsort(f1_det)
    types = [f"{types[i]} (n={sup[i]})" for i in order]
    f1_ann, f1_det = f1_ann[order], f1_det[order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(types))
    for i in range(len(types)):
        ax.plot([f1_det[i], f1_ann[i]], [y[i], y[i]], color='#95a5a6', lw=3.5, zorder=1)
        gap = f1_ann[i] - f1_det[i]
        if abs(gap) >= 0.03:
            ax.text((f1_ann[i] + f1_det[i]) / 2, y[i] + 0.18, f"Δ={gap:+.2f}",
                    ha='center', va='bottom', fontsize=8, color='#555555')

    ax.scatter(f1_ann, y, color='#2980b9', s=110, label='Annotated Peaks + Classifier (Ceiling: 91.6% Primary Acc)', zorder=3)
    ax.scatter(f1_det, y, color='#e67e22', s=110, label='Pan-Tompkins + Classifier (End-to-End: 77.1% Primary Acc)', zorder=3)

    for i in range(len(types)):
        ax.text(f1_det[i] - 0.018, y[i], f"{f1_det[i]:.3f}", va='center', ha='right', fontsize=8.5, color='#d35400', fontweight='bold')
        ax.text(f1_ann[i] + 0.018, y[i], f"{f1_ann[i]:.3f}", va='center', ha='left', fontsize=8.5, color='#1f618d', fontweight='bold')

    ax.set_yticks(y); ax.set_yticklabels(types, fontweight='bold')
    ax.set_xlim(0.25, 1.08)
    ax.set_xlabel('Window-Level F1 Score (30 s Windows, Held-Out Test Patients)')
    ax.set_title('Arrhythmia Detection F1 & R-Peak Error Propagation Across Pattern Types')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_window_dumbbell_f1.png')
    plt.savefig(path); plt.close()
    print("Saved:", path)


# =============================================================================
# FIGURE 3: Extended Threshold Sweep (0.30..0.995) Precision / Recall / F1 Curves
# =============================================================================
def plot_fig3_threshold_sweeps():
    thr = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995]
    # Fold B representative sweep curves from run_pattern_tuning.py
    v_p = [0.789, 0.795, 0.802, 0.810, 0.812, 0.870, 0.894, 0.924, 0.965, 0.982, 1.000, 1.000]
    v_r = [0.670, 0.664, 0.659, 0.648, 0.616, 0.581, 0.440, 0.240, 0.150, 0.101, 0.040, 0.002]
    v_f = [0.725, 0.724, 0.723, 0.720, 0.700, 0.697, 0.590, 0.381, 0.259, 0.183, 0.076, 0.004]

    s_p = [0.497, 0.554, 0.637, 0.676, 0.736, 0.763, 0.783, 0.884, 0.900, 0.944, 1.000, 1.000]
    s_r = [0.794, 0.742, 0.742, 0.711, 0.660, 0.598, 0.485, 0.392, 0.278, 0.175, 0.052, 0.021]
    s_f = [0.611, 0.634, 0.686, 0.693, 0.696, 0.671, 0.599, 0.543, 0.425, 0.296, 0.098, 0.040]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, p, r, f, opt_t, title in [
        (axes[0], v_p, v_r, v_f, 0.30, 'A. Ventricular (PVC) Family Threshold Sweep'),
        (axes[1], s_p, s_r, s_f, 0.70, 'B. Supraventricular (PAC) Family Threshold Sweep')
    ]:
        ax.plot(thr, p, 'o--', color='#2980b9', label='Micro Precision', lw=1.8)
        ax.plot(thr, r, 's--', color='#27ae60', label='Micro Recall', lw=1.8)
        ax.plot(thr, f, 'D-',  color='#c0392b', label='Micro F1 Score', lw=2.5)
        ax.axvline(opt_t, color='#c0392b', linestyle=':', lw=1.8, label=f'Optimal Threshold ({opt_t:.2f})')
        ax.axvline(0.95, color='gray', linestyle='-.', alpha=0.5, label='v2 Grid Saturation (0.95)')
        ax.set_title(title); ax.set_xlabel('Probability Threshold')
        ax.set_ylabel('Score'); ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.3); ax.legend(loc='lower left', fontsize=8.5)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_threshold_sweeps.png')
    plt.savefig(path); plt.close()
    print("Saved:", path)


# =============================================================================
# FIGURE 4: 4-Track AD8232 Hardware Emulation & Poor-Quality Suppression Strip
# =============================================================================
def plot_fig4_ad8232_telemetry():
    from test_ad8232_emulation import emulate_ad8232_esp32, DATA_DIR, MODEL, TEST_PATIENTS
    bundle = ep.load_bundle(MODEL)
    test_pids = set(np.load(TEST_PATIENTS, allow_pickle=True).tolist())
    recs = [h for h in ep.select_records(DATA_DIR) if os.path.basename(h).split('_')[0] in test_pids][:20]
    for h in recs:
        sig, smp, truth = ep.read_record(h)
        if sig is not None and len(smp) > 500:
            rec_name = os.path.basename(h)[:-4]; break

    MAX_S = 180
    sig_hw = emulate_ad8232_esp32(sig[:int(MAX_S * FS)])
    beats_hw, _, _ = ep.simulate_stream(sig_hw, bundle, chunk_s=0.5, max_s=MAX_S)
    wins = analyse_stream(beats_hw['label'].values, beats_hw['t'].values, MAX_S, 30.0, 30.0)

    t = np.arange(len(sig_hw)) / FS
    probs = np.vstack(beats_hw['probs'].values)
    bt = beats_hw['t'].values

    fig, axes = plt.subplots(4, 1, figsize=(14, 7.5), sharex=True,
                             gridspec_kw={'height_ratios': [1.4, 1.1, 0.9, 0.45]})

    # Track 1: Emulated AD8232 ECG Signal
    axes[0].plot(t, sig_hw, color='#2c3e50', lw=0.6, label='AD8232 Emulated Stream (12-bit ADC + 50Hz + Wander)')
    axes[0].axvspan(60, 72, color='#e74c3c', alpha=0.22, label='Electrode Motion / Lead-Off Saturation (60–72 s)')
    axes[0].set_ylabel('Amplitude (mV)'); axes[0].set_ylim(-3.4, 3.4)
    axes[0].set_title(f'Real-Time Streaming Telemetry & Artifact Gate on {rec_name} (0.5 s Chunks, 17.3 ms/hop)')
    axes[0].legend(loc='upper right', fontsize=8.5); axes[0].grid(alpha=0.25)

    # Track 2: Per-Beat Class Probabilities P(N), P(S), P(V), P(Q)
    axes[1].scatter(bt, probs[:, 0], s=14, color='#2ecc71', alpha=0.7, label='P(N: Normal)')
    axes[1].scatter(bt, probs[:, 3], s=22, color='#e74c3c', alpha=0.9, label='P(Q: Noise/Artifact)')
    axes[1].scatter(bt, probs[:, 1] + probs[:, 2], s=14, color='#9b59b6', alpha=0.7, label='P(S+V: Ectopic)')
    axes[1].axhline(bundle.get('q_threshold', 0.75), color='#c0392b', ls='--', lw=1.2, label='Q Gate (0.75)')
    axes[1].set_ylabel('Beat Prob.'); axes[1].set_ylim(-0.05, 1.08)
    axes[1].legend(loc='upper right', ncol=4, fontsize=8); axes[1].grid(alpha=0.25)

    # Track 3: Window Q-Burden vs max_q_frac Gate
    w_mid = [w['t0'] + 15.0 for w in wins]
    w_q   = [w.get('burden_Q', 0.0) for w in wins]
    bars = axes[2].bar(w_mid, w_q, width=26.0, color=['#e74c3c' if q > CFG['max_q_frac'] else '#3498db' for q in w_q], alpha=0.8)
    axes[2].axhline(CFG['max_q_frac'], color='#c0392b', ls='--', lw=1.5, label=f"Poor-Quality Gate (max_q_frac={CFG['max_q_frac']})")
    for b, q in zip(bars, w_q):
        axes[2].text(b.get_x() + b.get_width()/2, q + 0.03, f"{q*100:.1f}%", ha='center', fontsize=8.5, fontweight='bold')
    axes[2].set_ylabel('Window Q-Burden'); axes[2].set_ylim(0, 0.80)
    axes[2].legend(loc='upper right', fontsize=8.5); axes[2].grid(alpha=0.25)

    # Track 4: Window Decision Ribbon
    for w in wins:
        c = '#e67e22' if w['primary_type'] == 'poor_quality' else '#27ae60'
        axes[3].axvspan(w['t0'] + 0.3, w['t1'] - 0.3, color=c, alpha=0.85)
        lbl = 'POOR QUALITY\n(Alarm Suppressed)' if w['primary_type'] == 'poor_quality' else 'NO ARRHYTHMIA'
        axes[3].text((w['t0'] + w['t1'])/2, 0.5, lbl, ha='center', va='center', color='white', fontweight='bold', fontsize=8)
    axes[3].set_yticks([]); axes[3].set_ylabel('30s Output')
    axes[3].set_xlabel('Stream Time (seconds)'); axes[3].set_xlim(0, MAX_S)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_ad8232_telemetry_strip.png')
    plt.savefig(path); plt.close()
    print("Saved:", path)


# =============================================================================
# FIGURE 5: Streaming Causal vs Buffered + LightGBM Feature Importance
# =============================================================================
def plot_fig5_streaming_and_features():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), gridspec_kw={'width_ratios': [1.0, 1.25]})

    # Panel A: Buffered (1.5s margin) vs Strictly Causal (0.3s margin) on p00486_s00
    metrics = ['Normal (N) F1', 'PAC (S) F1', 'PVC (V) F1', 'Noise (Q) F1', 'Peak PPV']
    buf_scores = [0.900, 0.775, 1.000, 0.474, 0.922]
    cau_scores = [0.896, 0.756, 0.500, 0.407, 0.921]
    x = np.arange(len(metrics)); w = 0.35
    b1 = axes[0].bar(x - w/2, buf_scores, w, label='Buffered (1.5s margin, 2.52s latency)', color='#2980b9')
    b2 = axes[0].bar(x + w/2, cau_scores, w, label='Strictly Causal (0.3s margin, 1.32s latency)', color='#95a5a6')
    for b in list(b1) + list(b2):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.015, f"{b.get_height():.2f}",
                     ha='center', fontsize=8, fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels(metrics, rotation=15)
    axes[0].set_ylim(0, 1.15); axes[0].set_ylabel('Score on p00486_s00 (600s Stream)')
    axes[0].set_title('A. Streaming Trade-Off: 1.5s Lookahead vs Zero-Lookahead')
    axes[0].grid(axis='y', alpha=0.3); axes[0].legend(loc='lower right', fontsize=8.5)

    # Panel B: Top 15 LightGBM Features (from ecg_improved_v3.pkl)
    if os.path.exists('ecg_improved_v3.pkl'):
        b = joblib.load('ecg_improved_v3.pkl')
        imp = b['model'].booster_.feature_importance(importance_type='gain')
        imp = imp / imp.sum() * 100.0
        f_names = np.array(b['features'])
        top_ix = np.argsort(imp)[-15:]
        colors = ['#e74c3c' if f in {'rr_med20', 'rr_cv20', 'prev_over_med20', 'comp_deficit', 'local_premature_rate', 'rr_diff_norm'}
                  else ('#27ae60' if f in {'noise', 'baseline_off', 'amp_ratio'} else '#34495e')
                  for f in f_names[top_ix]]
        axes[1].barh(np.arange(15), imp[top_ix], color=colors, alpha=0.9)
        axes[1].set_yticks(np.arange(15)); axes[1].set_yticklabels(f_names[top_ix], fontweight='bold')
        axes[1].set_xlabel('Relative Information Gain (%)')
        axes[1].set_title('B. Top-15 LightGBM Features (Red = New v3 Long-RR, Green = Quality)')
        axes[1].grid(axis='x', alpha=0.3)
        patches = [
            mpatches.Patch(color='#34495e', label='Core RR & QRS Morphology'),
            mpatches.Patch(color='#e74c3c', label='New v3 Long-Range & Ectopy Context'),
            mpatches.Patch(color='#27ae60', label='Restored Signal Quality Features')
        ]
        axes[1].legend(handles=patches, loc='lower right', fontsize=8.5)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig5_streaming_and_feature_imp.png')
    plt.savefig(path); plt.close()
    print("Saved:", path)


if __name__ == '__main__':
    plot_fig1_cm_and_leakage()
    plot_fig2_window_dumbbell()
    plot_fig3_threshold_sweeps()
    plot_fig4_ad8232_telemetry()
    plot_fig5_streaming_and_features()
    print(f"\nAll 5 figures generated in: {os.path.abspath(OUT_DIR)}")