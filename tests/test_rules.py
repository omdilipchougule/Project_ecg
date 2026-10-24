"""Unit tests for arrhythmia_rules.py. Run: python test_rules.py   (or: pytest test_rules.py)"""
import numpy as np
from arrhythmia_rules import analyse_window, NONE_TYPE


def run(s, rr=0.8, **kw):
    s = s.replace(' ', '')
    rr = np.full(len(s), rr) if np.isscalar(rr) else np.asarray(rr, float)
    return analyse_window(list(s), np.cumsum(rr), **kw)


def types(r): return set(r['types'])


def test_normal():
    r = run('N' * 40); assert r['primary_type'] == NONE_TYPE, r['types']

def test_isolated_pvc():
    r = run('N' * 10 + 'V' + 'N' * 10); assert r['primary_type'] == 'isolated_PVC'

def test_isolated_pac():
    r = run('N' * 10 + 'S' + 'N' * 10); assert r['primary_type'] == 'isolated_PAC'

def test_couplet():
    r = run('N' * 8 + 'VV' + 'N' * 8); assert r['primary_type'] == 'PVC_couplet'

def test_triplet_slow_vs_nsvt_fast():
    assert run('N' * 8 + 'VVV' + 'N' * 8, 0.8)['primary_type'] == 'PVC_triplet'
    assert run('N' * 8 + 'VVV' + 'N' * 8, [0.8] * 8 + [0.5] * 3 + [0.8] * 8)['primary_type'] == 'NSVT'

def test_sustained_vt():
    rr = [0.8] * 5 + [0.45] * 80 + [0.8] * 5   # 80 beats * 0.45 s = 36 s
    r = run('N' * 5 + 'V' * 80 + 'N' * 5, rr); assert r['primary_type'] == 'sustained_VT'

def test_aivr():
    rr = [0.8] * 8 + [1.2] * 6 + [0.8] * 8     # 6 V beats at 50 bpm
    r = run('N' * 8 + 'V' * 6 + 'N' * 8, rr); assert r['primary_type'] == 'AIVR'

def test_bigeminy():
    r = run('NNNN' + 'VN' * 5 + 'NNNN'); assert 'ventricular_bigeminy' in types(r) and 'isolated_PVC' not in types(r)

def test_trigeminy():
    r = run('NN' + 'NNV' * 4 + 'NN'); assert 'ventricular_trigeminy' in types(r)

def test_two_alternating_is_not_bigeminy():
    r = run('N' * 6 + 'VNV' + 'N' * 10); assert 'ventricular_bigeminy' not in types(r) and 'isolated_PVC' in types(r)

def test_atrial_patterns():
    assert run('N' * 8 + 'SS' + 'N' * 8)['primary_type'] == 'PAC_couplet'
    assert 'atrial_bigeminy' in types(run('NNNN' + 'SN' * 4 + 'NNNN'))
    assert run('N' * 8 + 'SSSS' + 'N' * 8, [0.8] * 8 + [0.5] * 4 + [0.8] * 8)['primary_type'] == 'SVT_run'

def test_frequent_pvc_burden():
    r = run('N' * 25 + ('NV' * 3) + 'N' * 10 + 'V' * 0 + 'NNV' * 2)
    assert 'frequent_PVC' in types(r), r['burden_V']

def test_rate_flags():
    assert 'tachycardia' in types(run('N' * 30, 0.5)) and 'bradycardia' in types(run('N' * 30, 1.2))
    rr = np.full(30, 0.8); rr[15] = 2.5
    assert 'pause' in types(run('N' * 30, rr))

def test_irregular_rhythm_flag():
    rng = np.random.default_rng(0)
    r = run('N' * 40, rng.uniform(0.45, 1.1, 40)); assert 'irregular_rhythm_suspected' in types(r)
    assert 'irregular_rhythm_suspected' not in types(run('N' * 40, 0.8 + rng.normal(0, 0.01, 40)))

def test_priority_picks_most_severe():
    s = 'N' * 6 + 'V' + 'N' * 6 + 'VV' + 'N' * 6
    assert run(s)['primary_type'] == 'PVC_couplet'

def test_confidence_gate_drops_weak_runs():
    s = 'N' * 8 + 'VVV' + 'N' * 8; rr = [0.8] * 8 + [0.5] * 3 + [0.8] * 8
    P = np.zeros((len(s), 3)); P[:, 0] = 1; P[8:11] = [0.2, 0.0, 0.55]
    r = run(s, rr, probs=P, cfg={**__import__('arrhythmia_rules').CFG, 'min_event_conf': 0.7})
    assert 'NSVT' not in types(r)

def test_too_few_beats():
    assert run('NNN')['primary_type'] == 'insufficient_data'

def test_q_beat_does_not_create_false_pause():
    s = 'N' * 10 + 'Q' + 'N' * 10
    r = run(s, 1.1)
    assert 'pause' not in types(r)

def test_q_heavy_window_is_poor_quality():
    s = 'N' * 15 + 'Q' * 10
    r = run(s, 0.8)
    assert r['primary_type'] == 'poor_quality'


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for f in fns:
        f(); print('PASS', f.__name__)
    print(f'{len(fns)} tests passed')