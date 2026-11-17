"""Reproduce Gerster et al. core analysis (cross-subject FACED) using NEMAR raw BDF.

Reduced re-implementation (network-throttled S3):
  - 250 Hz subjects sub-008..023 (16 subjects)
  - first 10 s per video (10 x 1 s windows), 5-band log power (DE-equivalent)
  - mastoids A1/A2 dropped by name (mid-list in NEMAR BDF)
  - LOSO LinearSVC(C=1), per-subject z-score

This is NOT a bit-for-bit replication of the Synapse DE pipeline; it tests whether
the qualitative confound (stimulus labels decode better than subjective labels) holds.
"""
import csv, io, sys, base64, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import requests
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score

COMMIT = 'b19270dcf75170fc7588ddba3b1e9129a01e9e43'
S3 = 'https://nemar.s3.us-east-2.amazonaws.com/nm000112/objects/'
CACHE = Path('/tmp/faced_repro'); CACHE.mkdir(exist_ok=True)
PROXIES = {'http': 'http://127.0.0.1:8890', 'https': 'http://127.0.0.1:8890'}

CLASS_NAMES = ["Anger", "Disgust", "Fear", "Sadness", "Neutral",
               "Amusement", "Inspiration", "Joy", "Tenderness"]
SCORE_NAMES = ["Joy", "Tenderness", "Inspiration", "Amusement",
               "Anger", "Disgust", "Fear", "Sadness"]
SCORE_INDEX_TO_CLASS = [CLASS_NAMES.index(n) for n in SCORE_NAMES]  # [7,8,6,5,0,1,2,3]
VIDEO_LABELS_9CLASS = ([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 4
                       + [5] * 3 + [6] * 3 + [7] * 3 + [8] * 3)
BANDS = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]
MASTOID_NAMES = {'A1', 'A2'}
SINGLE_VIDEOS = {1, 4, 7, 10, 13, 17, 20, 23, 26}  # first video per class
N_SUBJECTS = 123
SECONDS_PER_VIDEO = 10
WPT = SECONDS_PER_VIDEO
SUBJECTS = list(range(8, 24))  # 250 Hz block


def get(url, headers=None, proxy_first=False, timeout=120, retries=5):
    order = [{'proxies': PROXIES}, {}] if proxy_first else [{}, {'proxies': PROXIES}]
    last = None
    for attempt in range(retries):
        for kw in order:
            try:
                r = requests.get(url, headers=headers, timeout=timeout, **kw)
                r.raise_for_status()
                return r
            except Exception as e:
                last = e
        time.sleep(2)
    raise last


def read_range(url, start, stop):
    r = get(url, headers={'Range': f'bytes={start}-{stop - 1}'}, timeout=120)
    assert r.status_code == 206 and len(r.content) == stop - start, (r.status_code, len(r.content), stop - start)
    return r.content


# ---------------- labels ----------------

def fetch_events(i):
    sid = f'sub-{i:03}'
    p = CACHE / f'{sid}_events.tsv'
    if p.exists():
        return sid, p.read_text(encoding='utf-8-sig')
    gh = get(f'https://api.github.com/repos/nemarDatasets/nm000112/contents/{sid}/eeg/{sid}_task-watchingVideoClips_events.tsv?ref={COMMIT}',
             headers={'User-Agent': 'h', 'Accept': 'application/vnd.github+json'})
    t = base64.b64decode(gh.json()['content']).decode('utf-8', 'replace').lstrip('\ufeff')
    p.write_text(t, encoding='utf-8')
    return sid, t


def load_labels():
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(fetch_events, range(N_SUBJECTS)))
    labels = {}
    for sid, text in results:
        rows = list(csv.DictReader(io.StringIO(text), delimiter='\t'))
        vids = [r for r in rows if (r.get('video_index') or '').strip() not in ('', 'n/a', 'None')]
        vids = sorted(vids, key=lambda r: int(float(r['video_index'])))
        arr = []
        for r in vids:
            scores = np.array([float(r[n]) for n in SCORE_NAMES])
            subj_cls = int(SCORE_INDEX_TO_CLASS[int(np.argmax(scores))])
            vi = int(float(r['video_index']))
            stim_cls = VIDEO_LABELS_9CLASS[vi - 1]
            arr.append(dict(vi=vi, onset=float(r['onset']), stim=stim_cls, subj=subj_cls))
        assert len(arr) == 28, (sid, len(arr))
        labels[int(sid.split('-')[1])] = arr
    return labels


# ---------------- BDF features ----------------

def parse_header(head0, head):
    n = int(head0[252:256])
    head_size = int(head0[184:192])
    record_s = float(head0[244:252])

    def field(offset, width):
        return [head[256 + offset * n + j * width:256 + offset * n + (j + 1) * width]
                .decode('ascii', 'replace').strip() for j in range(n)]

    names = field(0, 16)
    pmin = np.array(field(104, 8), float)
    pmax = np.array(field(112, 8), float)
    dmin = np.array(field(120, 8), float)
    dmax = np.array(field(128, 8), float)
    ns = np.array(field(216, 8), int)
    offs = np.r_[0, np.cumsum(ns[:-1]) * 3]
    width = int(ns.sum() * 3)
    fs = ns[0] / record_s
    keep = [j for j, nm in enumerate(names) if nm not in MASTOID_NAMES]
    return dict(head_size=head_size, record_s=record_s, fs=fs, pmin=pmin, pmax=pmax,
                dmin=dmin, dmax=dmax, ns=ns, offs=offs, width=width, keep=keep)


def band_logpower(x, fs):
    nch, n = x.shape
    x = x - x.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    out = np.zeros((nch, len(BANDS)))
    for bi, (a, b) in enumerate(BANDS):
        m = (freqs >= a) & (freqs < b)
        out[:, bi] = np.log10(np.maximum(spec[:, m].sum(axis=1), 1e-20))
    return out


def _read_video(url, H, onset):
    """Read one video's SECONDS_PER_VIDEO segment, return (nch, WPT, nbands) log power."""
    fs, offs, ns, width = H['fs'], H['offs'], H['ns'], H['width']
    keep, rec, hs = H['keep'], H['record_s'], H['head_size']
    pmin, pmax, dmin, dmax = H['pmin'], H['pmax'], H['dmin'], H['dmax']
    nch = len(keep)
    spw = int(round(fs))
    lo = int(np.floor(onset / rec))
    hi = int(np.ceil((onset + SECONDS_PER_VIDEO) / rec))
    raw = read_range(url, hs + lo * width, hs + hi * width)
    records = np.frombuffer(raw, np.uint8).reshape(hi - lo, width)
    x = np.zeros((nch, (hi - lo) * ns[0]))
    for k, j in enumerate(keep):
        b = records[:, offs[j]:offs[j] + ns[j] * 3].reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | b[:, 1] << 8 | b[:, 2] << 16
        v = np.where(v & 0x800000, v - 0x1000000, v)
        v = (v - dmin[j]) * (pmax[j] - pmin[j]) / (dmax[j] - dmin[j]) + pmin[j]
        x[k] = v
    s0 = int(round((onset - lo * rec) * fs))
    seg = x[:, s0:s0 + SECONDS_PER_VIDEO * spw]
    feat = np.zeros((WPT, nch * len(BANDS)))
    for w in range(WPT):
        ws = seg[:, w * spw:(w + 1) * spw]
        feat[w] = band_logpower(ws, fs).ravel()
    return feat


def compute_subject_features(i, labels_i):
    sid = f'sub-{i:03}'
    feat_cache = CACHE / f'{sid}_feat{SECONDS_PER_VIDEO}s.npy'
    if feat_cache.exists():
        return np.load(feat_cache)
    stem = f'{sid}_task-watchingVideoClips'
    gh = get(f'https://api.github.com/repos/nemarDatasets/nm000112/contents/{sid}/eeg/{stem}_eeg.bdf?ref={COMMIT}',
             headers={'User-Agent': 'h', 'Accept': 'application/vnd.github+json'})
    key = base64.b64decode(gh.json()['content']).decode('utf-8', 'replace').strip().split('/')[-1]
    url = S3 + key
    h0 = read_range(url, 0, 256)
    head = read_range(url, 0, int(h0[184:192]))
    H = parse_header(h0, head)
    if H['fs'] != 250.0:
        print(f'{sid}: fs={H["fs"]} (skip, not 250 Hz)', flush=True)
        return None
    feat = np.zeros((28, WPT, len(H['keep']) * len(BANDS)))
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(_read_video, url, H, row['onset']): vi for vi, row in enumerate(labels_i)}
        for fut in futs:
            vi = futs[fut]
            feat[vi] = fut.result()
    np.save(feat_cache, feat)
    print(f'{sid}: fs={H["fs"]:g}Hz feat={feat.shape}', flush=True)
    return feat


# ---------------- classification ----------------

def zscore_per_subject(F_sub):
    mu = F_sub.mean(axis=0, keepdims=True)
    sd = F_sub.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (F_sub - mu) / sd


def lds_smooth(X):
    """Gerster's scalar Kalman LDS smoother along time axis; X: (n_time, n_feat)."""
    u0 = X.mean(axis=0)
    Xt = X.T
    m, n = Xt.shape
    V0, A, T, C, sig = 0.01, 1.0, 0.0001, 1.0, 1.0
    P = np.zeros((m, n)); u = np.zeros((m, n)); V = np.zeros((m, n)); K = np.zeros((m, n))
    K[:, 0] = (V0 * C / (C * V0 * C + sig)) * np.ones(m)
    u[:, 0] = u0 + K[:, 0] * (Xt[:, 0] - C * u0)
    V[:, 0] = (np.ones(m) - K[:, 0] * C) * V0
    for i in range(1, n):
        P[:, i - 1] = A * V[:, i - 1] * A + T
        K[:, i] = P[:, i - 1] * C / (C * P[:, i - 1] * C + sig)
        u[:, i] = A * u[:, i - 1] + K[:, i] * (Xt[:, i] - C * A * u[:, i - 1])
        V[:, i] = (np.ones(m) - K[:, i] * C) * P[:, i - 1]
    return u.T


def loso(F, labels, subjects, label_key='stim', video_mask=None):
    accs, baccs, majs = [], [], []
    for held in subjects:
        tr = [s for s in subjects if s != held]
        Xtr, ytr, Xte, yte = [], [], [], []
        for s in tr:
            for vi, row in enumerate(labels[s]):
                if video_mask is not None and row['vi'] not in video_mask:
                    continue
                Xtr.append(F[s][vi].reshape(WPT, -1))
                ytr.append(row[label_key])
        for vi, row in enumerate(labels[held]):
            if video_mask is not None and row['vi'] not in video_mask:
                continue
            Xte.append(F[held][vi].reshape(WPT, -1))
            yte.append(row[label_key])
        Xtr = np.concatenate(Xtr); Xte = np.concatenate(Xte)
        ytr = np.repeat(ytr, WPT); yte = np.repeat(yte, WPT)
        clf = LinearSVC(C=1.0, random_state=7, max_iter=5000)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        accs.append(np.mean(pred == yte))
        baccs.append(balanced_accuracy_score(yte, pred))
        vals, counts = np.unique(ytr, return_counts=True)
        maj_cls = vals[np.argmax(counts)]
        majs.append(np.mean(yte == maj_cls))
    return np.array(accs), np.array(baccs), np.array(majs)


def main():
    print(f'=== Phase A: labels (all {N_SUBJECTS}) ===', flush=True)
    labels = load_labels()
    conc = np.array([np.mean([r['stim'] == r['subj'] for r in labels[s]]) for s in range(N_SUBJECTS)])
    print(f'concordance mean {conc.mean():.3f}  range [{conc.min():.3f},{conc.max():.3f}]', flush=True)

    subjects = [s for s in SUBJECTS]
    print(f'=== Phase B: features for subjects {subjects[0]}..{subjects[-1]} ({SECONDS_PER_VIDEO}s/video) ===', flush=True)
    F = {}
    for s in subjects:
        f = compute_subject_features(s, labels[s])
        if f is not None:
            F[s] = f
    subjects = sorted(F.keys())
    print(f'kept {len(subjects)} subjects at 250 Hz', flush=True)
    for s in subjects:
        flat = F[s].reshape(-1, F[s].shape[-1])
        F[s] = zscore_per_subject(lds_smooth(flat)).reshape(F[s].shape)

    print('=== Phase C: LOSO LinearSVC (C=1) ===', flush=True)
    a2, b2, m2 = loso(F, labels, subjects, 'stim')
    ac, bc, mc = loso(F, labels, subjects, 'subj')
    ad, bd, md = loso(F, labels, subjects, 'stim', video_mask=SINGLE_VIDEOS)
    def fmt(a):
        return f'{100*a.mean():.1f}% +/- {100*a.std()/np.sqrt(len(a)):.1f}%'
    print(f'2a stimulus labels  : acc {fmt(a2)}  balanced {fmt(b2)}  majority-baseline {fmt(m2)}', flush=True)
    print(f'2c subjective labels : acc {fmt(ac)}  balanced {fmt(bc)}  majority-baseline {fmt(mc)}', flush=True)
    print(f'2d single video      : acc {fmt(ad)}  balanced {fmt(bd)}  majority-baseline {fmt(md)}', flush=True)
    print(f'n_subjects={len(subjects)}  sec/video={SECONDS_PER_VIDEO}', flush=True)
    np.savez(CACHE / f'repro_{SECONDS_PER_VIDEO}s.npz', a2=a2, b2=b2, m2=m2,
             ac=ac, bc=bc, mc=mc, ad=ad, bd=bd, md=md,
             subjects=np.array(subjects), conc=conc[np.array(subjects)])


if __name__ == '__main__':
    main()
