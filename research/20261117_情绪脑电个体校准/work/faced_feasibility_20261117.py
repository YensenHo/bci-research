"""Tiny exploratory FACED pipeline check; not an estimate of model performance."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import requests
from scipy.signal import welch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COMMIT = 'b19270dcf75170fc7588ddba3b1e9129a01e9e43'
ROOT = Path(__file__).parent / 'faced_audit_v1.1.4'
CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
TRAIN_VIDEOS = {1, 7, 17, 23}
TEST_VIDEOS = {4, 10, 20, 26}

def read_range(url, start, stop):
    for attempt in range(4):
        try:
            response = requests.get(url, headers={'Range': f'bytes={start}-{stop - 1}'}, timeout=60)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 3: raise
            time.sleep(attempt + 1)
    if response.status_code != 206 or len(response.content) != stop - start or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{stop - 1}/'):
        raise ValueError('Unexpected response to range request')
    return response.content

def subject(i):
    sid = f'sub-{i:03}'; stem = f'{sid}_task-watchingVideoClips'
    cache = ROOT / f'feasibility_{sid}.json'
    if cache.exists(): return json.loads(cache.read_text(encoding='utf-8'))
    base = f'https://raw.githubusercontent.com/nemarDatasets/nm000112/{COMMIT}/{sid}/eeg/{stem}_eeg.bdf'
    pointer = requests.get(base, timeout=30); pointer.raise_for_status()
    key = pointer.text.strip().split('/')[-1]
    if not key.startswith('SHA256E-s'): raise ValueError('Unexpected BDF pointer')
    url = f'https://nemar.s3.us-east-2.amazonaws.com/nm000112/objects/{key}'
    head0 = read_range(url, 0, 256); n = int(head0[252:256]); head_size = int(head0[184:192]); record_s = float(head0[244:252])
    head = read_range(url, 0, head_size)
    def field(offset, width):
        return [head[256 + offset*n + j*width:256 + offset*n + (j+1)*width].decode('ascii').strip() for j in range(n)]
    names = field(0, 16); ids = [names.index(c) for c in CHANNELS]
    unit = field(96, 8); phys_min = np.array(field(104, 8), float); phys_max = np.array(field(112, 8), float)
    dig_min = np.array(field(120, 8), float); dig_max = np.array(field(128, 8), float)
    samples = np.array(field(216, 8), int); offsets = np.r_[0, np.cumsum(samples[:-1])*3]; record_bytes = int(samples.sum()*3)
    if any(unit[j] != 'uV' for j in ids) or len(set(samples[ids])) != 1: raise ValueError('Unexpected units or sampling rate')
    fs = samples[ids[0]] / record_s
    def features(start, stop):
        lo = int(np.floor(start / record_s)); hi = int(np.ceil(stop / record_s))
        if lo < 0: raise ValueError('Negative recording time')
        raw = read_range(url, head_size + lo*record_bytes, head_size + hi*record_bytes)
        records = np.frombuffer(raw, np.uint8).reshape(hi-lo, record_bytes)
        x = []
        for j in ids:
            b = records[:, offsets[j]:offsets[j]+samples[j]*3].reshape(-1, 3).astype(np.int32)
            v = b[:,0] | b[:,1]<<8 | b[:,2]<<16; v = np.where(v & 0x800000, v - 0x1000000, v)
            v = (v-dig_min[j])*(phys_max[j]-phys_min[j])/(dig_max[j]-dig_min[j])+phys_min[j]
            x.append(v)
        x = np.asarray(x)[:, round((start-lo*record_s)*fs):round((stop-lo*record_s)*fs)]
        if x.shape[1] != round((stop-start)*fs): raise ValueError('Unexpected segment length')
        f, psd = welch(x, fs=fs, nperseg=min(x.shape[1], round(fs*2)), axis=-1)
        return np.log(np.maximum(np.stack([psd[:,(f>=lo)&(f<hi)].mean(axis=1) for lo,hi in [(1,4),(4,8),(8,13),(13,30)]], axis=1), 1e-20)).ravel()
    events = pd.read_csv(ROOT / f'{stem}_events.tsv', sep='\t')
    events = events[events.video_index.notna()].sort_values('onset')
    first = float(events.iloc[0].onset); baseline = features(first-4, first-1)
    chosen = TRAIN_VIDEOS if i < 6 else TEST_VIDEOS
    rows = []
    for event in events.itertuples():
        if int(event.video_index) in chosen:
            x = features(float(event.onset), float(event.onset)+4)
            rows.append({'subject':sid, 'video':int(event.video_index), 'arousal':float(event.Arousal), 'x':x.tolist(), 'calibrated':(x-baseline).tolist()})
    if len(rows) != 4: raise ValueError('Missing trial')
    cache.write_text(json.dumps(rows), encoding='utf-8')
    print(f'{sid}: {len(rows)} video segments, {fs:g} Hz', flush=True)
    return rows

def main():
    with ThreadPoolExecutor(max_workers=4) as pool: rows = [r for group in pool.map(subject, range(8)) for r in group]
    train = [r for r in rows if int(r['subject'][-3:]) < 6]; test = [r for r in rows if int(r['subject'][-3:]) >= 6]
    assert not ({r['subject'] for r in train} & {r['subject'] for r in test})
    assert not ({r['video'] for r in train} & {r['video'] for r in test})
    ytrain = np.array([r['arousal'] for r in train]); ytest = np.array([r['arousal'] for r in test])
    results = {'training_mean':np.full(len(test), ytrain.mean()).tolist()}
    for key in ('x', 'calibrated'):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10))
        model.fit(np.array([r[key] for r in train]), ytrain)
        results[key] = model.predict(np.array([r[key] for r in test])).tolist()
    output = {'dataset_commit':COMMIT,'purpose':'pipeline-only; 6 train subjects, 2 test subjects; 4 training and 4 unseen test videos; no model selection',
      'limitations':'baseline unverified; ratings from converted events not independently matched to source; tiny sample; 4-second EEG against retrospective whole-video rating; no inference about efficacy',
      'channels':CHANNELS, 'train_subjects':sorted(set(r['subject'] for r in train)), 'train_videos':sorted(TRAIN_VIDEOS),
      'test_subjects':sorted(set(r['subject'] for r in test)), 'test_videos':sorted(TEST_VIDEOS),
      'rows':[{'subject':r['subject'],'video':r['video'],'arousal':r['arousal']} for r in test],
      'predictions':results, 'mae':{k:float(np.mean(np.abs(np.asarray(v)-ytest))) for k,v in results.items()}}
    dest = ROOT / 'feasibility.json'; dest.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output['mae']))

if __name__ == '__main__': main()
