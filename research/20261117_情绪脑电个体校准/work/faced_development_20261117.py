"""Pre-specified exploratory development set; no held-out test subjects are read."""
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
from faced_feasibility_20261117 import read_range, COMMIT, CHANNELS, ROOT

TRAIN_SUBJECTS = list(range(8, 20))
VALIDATION_SUBJECTS = list(range(20, 24))
TRAIN_VIDEOS = [1, 4, 7, 10, 13, 17, 20, 23, 26]
VALIDATION_VIDEOS = [2, 5, 8, 11, 14, 18, 21, 24, 27]
CACHE = ROOT / 'development_v1'
CACHE.mkdir(exist_ok=True)

def subject(i):
    sid=f'sub-{i:03}'; stem=f'{sid}_task-watchingVideoClips'; cache=CACHE/f'{sid}.json'
    if cache.exists(): return json.loads(cache.read_text(encoding='utf-8'))
    url=f'https://raw.githubusercontent.com/nemarDatasets/nm000112/{COMMIT}/{sid}/eeg/{stem}_eeg.bdf'
    for attempt in range(4):
        try:
            pointer=requests.get(url,timeout=30); pointer.raise_for_status(); break
        except requests.RequestException:
            if attempt==3: raise
            time.sleep(attempt+1)
    key=pointer.text.strip().split('/')[-1]
    if not key.startswith('SHA256E-s'): raise ValueError(f'{sid}: invalid object pointer')
    url=f'https://nemar.s3.us-east-2.amazonaws.com/nm000112/objects/{key}'
    head0=read_range(url,0,256); n=int(head0[252:256]); size=int(head0[184:192]); record_s=float(head0[244:252]); head=read_range(url,0,size)
    def field(offset,width): return [head[256+offset*n+j*width:256+offset*n+(j+1)*width].decode('ascii').strip() for j in range(n)]
    names=field(0,16); ids=[names.index(c) for c in CHANNELS]; unit=field(96,8)
    pmin=np.array(field(104,8),float); pmax=np.array(field(112,8),float); dmin=np.array(field(120,8),float); dmax=np.array(field(128,8),float)
    ns=np.array(field(216,8),int); offs=np.r_[0,np.cumsum(ns[:-1])*3]; width=int(ns.sum()*3)
    if any(unit[j]!='uV' for j in ids) or len(set(ns[ids]))!=1: raise ValueError(f'{sid}: unexpected units or rate')
    fs=ns[ids[0]]/record_s
    def extract(start,stop):
        lo=int(np.floor(start/record_s)); hi=int(np.ceil(stop/record_s))
        if lo<0: raise ValueError('negative start')
        data=read_range(url,size+lo*width,size+hi*width); records=np.frombuffer(data,np.uint8).reshape(hi-lo,width)
        x=[]; clipped=[]
        for j in ids:
            b=records[:,offs[j]:offs[j]+ns[j]*3].reshape(-1,3).astype(np.int32)
            v=b[:,0]|b[:,1]<<8|b[:,2]<<16; v=np.where(v&0x800000,v-0x1000000,v)
            v=v[round((start-lo*record_s)*fs):round((stop-lo*record_s)*fs)]
            clipped.append(float(np.mean((v<=dmin[j])|(v>=dmax[j]))))
            x.append((v-dmin[j])*(pmax[j]-pmin[j])/(dmax[j]-dmin[j])+pmin[j])
        x=np.asarray(x)
        if x.shape[1]!=round((stop-start)*fs): raise ValueError('bad segment length')
        f,p=welch(x,fs=fs,nperseg=round(fs*2),axis=-1)
        feature=np.log(np.maximum(np.stack([p[:,(f>=a)&(f<b)].mean(axis=1) for a,b in [(1,4),(4,8),(8,13),(13,30)]],axis=1),1e-20)).ravel()
        quality={'dc_abs_median_uv':float(np.median(np.abs(np.mean(x,axis=1)))), 'std_median_uv':float(np.median(np.std(x,axis=1))), 'clipped_fraction_max':max(clipped)}
        return feature,quality
    events=pd.read_csv(ROOT/f'{stem}_events.tsv',sep='\t'); events=events[events.video_index.notna()].sort_values('onset')
    first=float(events.iloc[0].onset); base,base_quality=extract(first-4,first-1)
    chosen=TRAIN_VIDEOS if i in TRAIN_SUBJECTS else VALIDATION_VIDEOS; rows=[]
    for event in events.itertuples():
        if int(event.video_index) in chosen:
            if event.duration<30: raise ValueError('video shorter than 30 seconds')
            feature,quality=extract(float(event.onset),float(event.onset)+30)
            rows.append({'subject':sid,'video':int(event.video_index),'arousal':float(event.Arousal),'x':feature.tolist(),'calibrated':(feature-base).tolist(),'quality':quality})
    if len(rows)!=len(chosen): raise ValueError(f'{sid}: expected {len(chosen)} trials')
    result={'subject':sid,'fs':fs,'baseline_quality':base_quality,'rows':rows}
    tmp=cache.with_suffix('.tmp'); tmp.write_text(json.dumps(result),encoding='utf-8'); tmp.replace(cache)
    print(f'{sid}: {len(rows)} trials, {fs:g} Hz',flush=True); return result

def main():
    assert not(set(TRAIN_SUBJECTS)&set(VALIDATION_SUBJECTS)) and not(set(TRAIN_VIDEOS)&set(VALIDATION_VIDEOS))
    with ThreadPoolExecutor(max_workers=4) as pool: subjects=list(pool.map(subject,TRAIN_SUBJECTS+VALIDATION_SUBJECTS))
    train=[r for s in subjects if int(s['subject'][-3:]) in TRAIN_SUBJECTS for r in s['rows']]
    val=[r for s in subjects if int(s['subject'][-3:]) in VALIDATION_SUBJECTS for r in s['rows']]
    yt=np.array([r['arousal'] for r in train]); y=np.array([r['arousal'] for r in val]); pred={'training_mean':np.full(len(val),yt.mean()).tolist()}
    for key in ['x','calibrated']:
        model=make_pipeline(StandardScaler(),Ridge(alpha=10)); model.fit([r[key] for r in train],yt); pred[key]=model.predict([r[key] for r in val]).tolist()
    per_subject={}
    for sid in sorted({r['subject'] for r in val}):
        m=np.array([r['subject']==sid for r in val]); per_subject[sid]={k:float(np.mean(np.abs(np.asarray(v)[m]-y[m]))) for k,v in pred.items()}
    result={'commit':COMMIT,'split':{'train_subjects':TRAIN_SUBJECTS,'validation_subjects':VALIDATION_SUBJECTS,'train_videos':TRAIN_VIDEOS,'validation_videos':VALIDATION_VIDEOS},
      'counts':{'train_trials':len(train),'validation_trials':len(val)},'mae':{k:float(np.mean(np.abs(np.asarray(v)-y))) for k,v in pred.items()},
      'per_subject_mae':per_subject,'validation_rows':[{'subject':r['subject'],'video':r['video'],'arousal':r['arousal'],'quality':r['quality']} for r in val],
      'predictions':pred,'sampling_rates':{s['subject']:s['fs'] for s in subjects},'baseline_quality':{s['subject']:s['baseline_quality'] for s in subjects}}
    (CACHE/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps({'counts':result['counts'],'mae':result['mae'],'per_subject':per_subject},ensure_ascii=False))

if __name__=='__main__': main()
