"""Two-subject exploratory read check, not a model or data-quality certification."""
from pathlib import Path
import json
import numpy as np
import requests
from scipy.signal import welch
from scipy.io import loadmat
from io import BytesIO
import hashlib
from faced_feasibility_20261117 import read_range

ROOT=Path(__file__).parent/'dens_audit_v1.0.0';COMMIT='beb73da6fde5c043d42145637db9f5bce7e3fe12'

def main():
    manifest=json.loads((ROOT/'trial_manifest.json').read_text()); rows=[]
    for sid in ('sub-mit003','sub-mit004'):
        subj=next(s for s in manifest['subjects_data'] if s['subject']==sid)
        set_url=f'https://raw.githubusercontent.com/nemarDatasets/on003751/{COMMIT}/{sid}/eeg/{sid}_task-Emotion_eeg.set'
        set_ref=requests.get(set_url,timeout=30);set_ref.raise_for_status();set_key=set_ref.text.strip().split('/')[-1]
        src=requests.get(f'https://nemar.s3.us-east-2.amazonaws.com/on003751/objects/{set_key}',timeout=60);src.raise_for_status()
        if hashlib.md5(src.content).hexdigest()!=set_key.split('--')[-1].split('.')[0]:raise ValueError('set checksum mismatch')
        eeg=loadmat(BytesIO(src.content),squeeze_me=True,struct_as_record=False)['EEG']
        if eeg.nbchan!=132 or eeg.srate!=250 or [str(c.labels) for c in eeg.chanlocs[:8]]!=[f'E{i}' for i in range(1,9)]:raise ValueError('unexpected EEG layout')
        url=f'https://raw.githubusercontent.com/nemarDatasets/on003751/{COMMIT}/{sid}/eeg/{sid}_task-Emotion_eeg.fdt'
        r=requests.get(url,timeout=30);r.raise_for_status();key=r.text.strip().split('/')[-1]
        if not key.startswith('MD5E-s'):raise ValueError('unexpected source object')
        url=f'https://nemar.s3.us-east-2.amazonaws.com/on003751/objects/{key}'
        for trial in subj['trials']:
            # EEGLAB uses 1-based latency; the original .set was independently checked for these two people.
            start=int(round(trial['latency_sample']-1)); count=250*5; channels=132
            raw=read_range(url,start*channels*4,(start+count)*channels*4)
            x=np.frombuffer(raw,dtype='<f4').reshape(count,channels)[:,:8].T
            f,p=welch(x,fs=250,nperseg=500,axis=-1)
            bands=[float(np.median(p[:,(f>=a)&(f<b)])) for a,b in [(1,4),(4,8),(8,13),(13,30)]]
            rows.append({'subject':sid,'stimulus':trial['stimulus'],'arousal':trial['arousal'],
              'median_mean_abs_uv':float(np.median(abs(x.mean(axis=1)))),
              'median_centered_std_uv':float(np.median(x.std(axis=1))), 'median_band_psd':bands})
        print(f'{sid}: {len(subj["trials"])} trial segments checked',flush=True)
    out={'warning':'two people, first five seconds, first 8 EGI channels; no artifact correction; not emotion decoding','rows':rows}
    (ROOT/'two_subject_signal_probe.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps({'subjects':2,'trials':len(rows),'finite':all(np.isfinite(v) for row in rows for v in row['median_band_psd'])}))

if __name__=='__main__':main()
