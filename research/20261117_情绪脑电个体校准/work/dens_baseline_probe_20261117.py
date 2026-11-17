"""Two-subject unlabeled baseline stability probe, no prediction or fitting."""
import json
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.signal import welch
from faced_feasibility_20261117 import read_range

COMMIT='beb73da6fde5c043d42145637db9f5bce7e3fe12'; ROOT=Path(__file__).parent/'dens_audit_v1.0.0'

def main():
    output=[]
    for sid in ('sub-mit003','sub-mit004'):
        base=f'https://raw.githubusercontent.com/nemarDatasets/on003751/{COMMIT}/{sid}/eeg/{sid}_task-'
        r=requests.get(base+'emotion_events.tsv',timeout=30);r.raise_for_status();events=pd.read_csv(StringIO(r.text),sep='\t')
        start=float(events[events.trial_type.eq('base')].iloc[0].onset); stop=float(events[(events.trial_type.eq('bend')) & (events.onset>start)].iloc[0].onset)
        if (stop-start)/250 < 75:raise ValueError('insufficient baseline')
        r=requests.get(base+'Emotion_eeg.fdt',timeout=30);r.raise_for_status();key=r.text.strip().split('/')[-1]
        if not key.startswith('MD5E-s'):raise ValueError('unexpected file pointer')
        url=f'https://nemar.s3.us-east-2.amazonaws.com/on003751/objects/{key}';begin=round(start-1);samples=75*250;channels=132
        raw=read_range(url,begin*channels*4,(begin+samples)*channels*4)
        x=np.frombuffer(raw,dtype='<f4').reshape(samples,channels)[:,:8].T
        def features(sig):
            f,p=welch(sig,fs=250,nperseg=500,axis=-1)
            return np.log(np.maximum(np.stack([p[:,(f>=a)&(f<b)].mean(axis=1) for a,b in [(1,4),(4,8),(8,13),(13,30)]],axis=1),1e-20)).ravel()
        durations={str(s):features(x[:,:s*250]) for s in (3,10,30,60,75)}
        late=np.stack([features(x[:,i*3*250:(i+1)*3*250]) for i in range(25)])
        result={'subject':sid,'logpower_median_abs_difference_from_75s':{s:float(np.median(abs(v-durations['75']))) for s,v in durations.items()},
          'nonoverlapping_early_late_logpower_difference':{str(s):float(np.median(abs(features(x[:,:s*250])-features(x[:,-s*250:])))) for s in (3,10,30)},
          'median_feature_std_across_3s_windows':float(np.median(np.std(late,axis=0))),
          'early_to_late_3s_median_abs_difference':float(np.median(abs(late[0]-late[-1])))}
        output.append(result);print(json.dumps(result),flush=True)
    (ROOT/'baseline_stability_probe.json').write_text(json.dumps({'warning':'two subjects; same 75s used as reference; 8 EGI channels; signal quality and eye state unverified; not evidence of emotion or calibration performance','results':output},indent=2),encoding='utf8')

if __name__=='__main__':main()
