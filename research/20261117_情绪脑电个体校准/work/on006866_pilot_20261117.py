"""Single-participant chronological pipeline feasibility only; no per-trial output."""
import csv
import json
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from on006866_verify_read_20261117 import fetch

CHANNELS = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2']
DATA_MATRIX = 25156088

def trials_from_events(path, count=30):
    with open(path, encoding='utf-8-sig', newline='') as f:
        events = list(csv.DictReader(f, delimiter='\t'))
    trials = []
    prompt = None
    for event in events:
        value = int(event['value'])
        if 18 <= value <= 23:
            trials.append({'onset': float(event['onset']), 'condition': value, 'arousal': [], 'valence': [], 'order': []})
            prompt = None
        elif trials and value in (48, 49):
            prompt = 'arousal' if value == 49 else 'valence'
            trials[-1]['order'].append(prompt)
        elif trials and 1 <= value <= 9 and prompt:
            trials[-1][prompt].append(value)
            prompt = None
    if len(trials) < count: raise ValueError('too few trials')
    chosen = trials[:count]
    if any(len(t['arousal']) != 1 or len(t['valence']) != 1 or t['order'] != ['arousal','valence'] for t in chosen):
        raise ValueError('ambiguous/missing rating in selected trials')
    if any(b['onset'] <= a['onset'] + 2 for a,b in zip(chosen,chosen[1:])):
        raise ValueError('nonchronological or overlapping trials')
    return chosen

def main(events_path, channels_path, full=False):
    ntrain = 120 if full else 20
    ntrials = 240 if full else 30
    with open(channels_path, encoding='utf-8-sig', newline='') as f:
        channels = list(csv.DictReader(f, delimiter='\t'))
    names = [c['name'] for c in channels]
    ids = [names.index(x) for x in CHANNELS]
    if any(channels[i]['units'] != 'uV' for i in ids): raise ValueError('unexpected channel unit')
    header = fetch(DATA_MATRIX, DATA_MATRIX+56)
    words = struct.unpack('<14I',header)
    if (words[0],words[8],words[9],words[10],words[11],words[12],words[13]) != (14,69,3144334,262145,1635017060,7,867836184):
        raise ValueError('unrecognized MATLAB data layout')
    trials = trials_from_events(events_path, ntrials)
    def extract(t):
        sample = round(t['onset']*1000)
        start = DATA_MATRIX+56+sample*69*4
        raw = fetch(start,start+2000*69*4)
        x = np.frombuffer(raw,dtype='<f4').reshape(2000,69)[:,ids].T
        if not np.isfinite(x).all(): raise ValueError('nonfinite samples')
        f,p = welch(x,fs=1000,nperseg=2000,axis=-1)
        return np.log(np.maximum(np.stack([p[:,(f>=a)&(f<b)].mean(axis=1) for a,b in [(1,4),(4,8),(8,13),(13,30)]],axis=1),1e-20)).ravel()
    with ThreadPoolExecutor(max_workers=4) as pool:
        features = []
        for feature in pool.map(extract,trials):
            features.append(feature)
            if len(features) % 30 == 0: print(f"processed {len(features)}/{ntrials}", flush=True)
        X = np.array(features)
    y = np.array([t['arousal'][0] for t in trials], dtype=float)
    cond = np.array([t['condition'] for t in trials])
    mean = y[:ntrain].mean()
    condition_means = {c:y[:ntrain][cond[:ntrain]==c].mean() for c in set(cond[:ntrain])}
    base = np.array([condition_means.get(c,mean) for c in cond])
    model = lambda: make_pipeline(StandardScaler(),Ridge(alpha=10))
    eeg = model().fit(X[:ntrain],y[:ntrain]).predict(X[ntrain:])
    residual = model().fit(X[:ntrain],y[:ntrain]-base[:ntrain]).predict(X[ntrain:]) + base[ntrain:]
    preds = {'training_mean':np.full(ntrials-ntrain,mean), 'condition_mean':base[ntrain:], 'eeg_ridge':eeg, 'condition_plus_eeg_residual':residual}
    print(json.dumps({'purpose':'single-subject pipeline feasibility; not generalizable', 'train_trials':ntrain,'test_trials':ntrials-ntrain,
       'mae':{k:round(float(np.mean(abs(v-y[ntrain:]))),4) for k,v in preds.items()},
       'unseen_test_condition_count':int(sum(c not in condition_means for c in cond[ntrain:])), 'channels':len(CHANNELS)}))

if __name__=='__main__': main(sys.argv[1],sys.argv[2], '--full' in sys.argv[3:])
