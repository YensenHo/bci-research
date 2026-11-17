"""Development-only, fixed-center, non-overlapping baseline comparison."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO, StringIO
from pathlib import Path
import hashlib, json, time
import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat
from scipy.signal import welch
from faced_feasibility_20261117 import read_range

COMMIT='beb73da6fde5c043d42145637db9f5bce7e3fe12';ROOT=Path(__file__).parent/'dens_audit_v1.0.0';CACHE=ROOT/'baseline_batch';CACHE.mkdir(exist_ok=True)

def get(url):
    for attempt in range(3):
        try:r=requests.get(url,timeout=60);r.raise_for_status();return r
        except requests.RequestException:
            if attempt==2:raise
            time.sleep(attempt+1)

def subject(sid):
    cache=CACHE/f'{sid}.json'
    if cache.exists():return json.loads(cache.read_text(encoding='utf8'))
    base=f'https://raw.githubusercontent.com/nemarDatasets/on003751/{COMMIT}/{sid}/eeg/{sid}_task-'
    events=pd.read_csv(StringIO(get(base+'emotion_events.tsv').text),sep='\t')
    starts=events[events.trial_type.eq('base')]; ends=events[events.trial_type.eq('bend')]
    if starts.empty or ends.empty:raise ValueError('missing baseline markers')
    start=float(starts.iloc[0].onset);ends=ends[ends.onset>start]
    if ends.empty or (float(ends.iloc[0].onset)-start)/250<75:raise ValueError('short baseline')
    if ((events.trial_type=='boundary') & (events.onset>=start) & (events.onset<start+75*250)).any():raise ValueError('boundary in baseline')
    set_key=get(base+'Emotion_eeg.set').text.strip().split('/')[-1]
    src=get(f'https://nemar.s3.us-east-2.amazonaws.com/on003751/objects/{set_key}')
    if hashlib.md5(src.content).hexdigest()!=set_key.split('--')[-1].split('.')[0]:raise ValueError('set checksum mismatch')
    eeg=loadmat(BytesIO(src.content),squeeze_me=True,struct_as_record=False)['EEG']
    if eeg.nbchan!=132 or eeg.srate!=250 or [str(c.labels) for c in eeg.chanlocs[:8]]!=[f'E{i}' for i in range(1,9)]:raise ValueError('unexpected channel layout')
    if round(start-1)+75*250>int(eeg.pnts):raise ValueError('baseline exceeds recording')
    key=get(base+'Emotion_eeg.fdt').text.strip().split('/')[-1]
    if not key.startswith('MD5E-s'):raise ValueError('bad FDT pointer')
    first=round(start-1);raw=read_range(f'https://nemar.s3.us-east-2.amazonaws.com/on003751/objects/{key}',first*132*4,(first+75*250)*132*4)
    x=np.frombuffer(raw,dtype='<f4').reshape(75*250,132)[:,:8].T
    def power(segment):
        f,p=welch(segment,fs=250,nperseg=500,axis=-1)
        return np.log(np.maximum(np.stack([p[:,(f>=a)&(f<b)].mean(axis=1) for a,b in [(1,4),(4,8),(8,13),(13,30)]],axis=1),1e-20))
    diff={}
    for duration in (3,10,30):
        half=duration*250//2
        left=x[:,15*250-half:15*250+half];right=x[:,60*250-half:60*250+half]
        diff[str(duration)]=float(np.median(abs(power(left)-power(right))))
    result={'subject':sid,'set_md5':hashlib.md5(src.content).hexdigest(),'median_abs_difference_logpower':diff,
      'median_centered_std_uv':float(np.median(x.std(axis=1))), 'raw_mean_abs_median_uv':float(np.median(abs(x.mean(axis=1))))}
    tmp=cache.with_suffix('.tmp');tmp.write_text(json.dumps(result),encoding='utf8');tmp.replace(cache)
    print(sid,'completed',flush=True);return result

def main():
    audit=json.loads((ROOT/'metadata_audit.json').read_text());ids=[r['subject'] for r in audit['details'] if r['ordered_names_match'] is True and r['sampling_rate'] is not None]
    rows=[]; errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(subject,s):s for s in ids}
        for future in as_completed(futures):
            try:rows.append(future.result())
            except Exception as e:errors.append({'subject':futures[future],'error':f'{type(e).__name__}: {e}'})
    rows.sort(key=lambda r:r['subject']);subset=[r for r in rows if r['subject'] not in ('sub-mit003','sub-mit004')]
    def summary(rs):return {str(s):float(np.median([r['median_abs_difference_logpower'][str(s)] for r in rs])) for s in (3,10,30)} if rs else {}
    out={'warning':'exploratory; nested windows, possible nonstationarity and artifacts, no prediction or clinical conclusion','expected':len(ids),'completed':len(rows),'errors':errors,'medians_all':summary(rows),'medians_excluding_initial_2':summary(subset),'subjects':rows}
    (CACHE/'results.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in out.items() if k!='subjects'}))

if __name__=='__main__':main()
