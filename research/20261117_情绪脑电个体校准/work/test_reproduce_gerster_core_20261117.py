"""Pure-logic tests for reproduce_gerster_core_20261117.py (no network, deterministic).

Covers the label mapping, feature/smoothing helpers, fetch_events BOM handling, and the
loso() 3-tuple return (balanced accuracy + majority baseline). Run: python test_...py
"""
import csv
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import reproduce_gerster_core_20261117 as R


def _check(name, fn):
    fn()
    print(f'  PASS {name}')


def test_label_mapping():
    assert len(R.VIDEO_LABELS_9CLASS) == 28
    assert sorted(set(R.VIDEO_LABELS_9CLASS)) == list(range(9))
    assert R.VIDEO_LABELS_9CLASS.count(4) == 4  # neutral has 4 videos, others 3
    assert R.SCORE_INDEX_TO_CLASS == [7, 8, 6, 5, 0, 1, 2, 3]


def test_band_logpower():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 250))
    out = R.band_logpower(x, 250.0)
    assert out.shape == (5, len(R.BANDS))
    assert np.isfinite(out).all()


def test_lds_smooth():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 10))
    Y = R.lds_smooth(X)
    assert Y.shape == X.shape and np.isfinite(Y).all()


def test_fetch_events_bom_and_rows():
    tmp = Path(tempfile.mkdtemp(prefix='hermes-verify-cache-'))
    R.CACHE = tmp
    cols = ['onset', 'duration', 'trial_type', 'value', 'video_index'] + R.SCORE_NAMES
    rows = ['\t'.join(cols)]
    for vi in range(1, 29):
        rows.append('\t'.join([f'{vi * 10}.0', '5', f'vid{vi}', '1', str(vi)] + ['1'] * 8))
    rows.append('\t'.join(['999.0', '1', 'fixation', '0', 'n/a'] + [''] * 8))
    (tmp / 'sub-008_events.tsv').write_text('\ufeff' + '\n'.join(rows), encoding='utf-8')

    sid, t = R.fetch_events(8)
    assert sid == 'sub-008' and not t.startswith('\ufeff')
    parsed = list(csv.DictReader(io.StringIO(t), delimiter='\t'))
    vids = [r for r in parsed if (r.get('video_index') or '').strip() not in ('', 'n/a', 'None')]
    assert len(vids) == 28


def test_loso_returns_balanced_and_majority():
    rng = np.random.default_rng(0)
    n_subj, NF = 3, 5
    F = {s: rng.normal(size=(28, R.WPT, NF)) for s in range(n_subj)}
    labels = {s: [dict(vi=vi, stim=R.VIDEO_LABELS_9CLASS[vi - 1],
                       subj=(R.VIDEO_LABELS_9CLASS[vi - 1] + 3) % 9) for vi in range(1, 29)]
              for s in range(n_subj)}
    acc, bacc, maj = R.loso(F, labels, list(range(n_subj)), 'stim')
    assert acc.shape == bacc.shape == maj.shape == (n_subj,)
    assert (bacc >= 0).all() and (bacc <= 1).all()
    assert (maj >= 1 / 9 - 1e-9).all()  # majority baseline >= chance


if __name__ == '__main__':
    for name, fn in [('label mapping', test_label_mapping),
                     ('band_logpower', test_band_logpower),
                     ('lds_smooth', test_lds_smooth),
                     ('fetch_events BOM+28 rows', test_fetch_events_bom_and_rows),
                     ('loso balanced+majority', test_loso_returns_balanced_and_majority)]:
        _check(name, fn)
    print('ALL PASS')
