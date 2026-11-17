"""Fixed exploratory cross-subject pilot; only aggregate metrics are printed."""
import csv, io, json, re, struct, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import requests
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader
from scipy.signal import welch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COMMIT='7350e08e5e5e4cb7d2c96b3933a298b3ef67daa3'
BASE=f'https://raw.githubusercontent.com/nemarDatasets/on006866/{COMMIT}/'
CHANNELS=['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2']
TRAIN=range(2,8); TARGET=range(8,10)
CACHE=Path(__file__).parent/'on006866_cross_cache'

def request(url,start=None,stop=None,size=None):
    for attempt in range(5):
        try:
            kw={} if start is None else {'headers':{'Range':f'bytes={start}-{stop-1}'}}
            r=requests.get(url,timeout=(12,60),**kw);r.raise_for_status()
            if start is not None and (r.status_code!=206 or len(r.content)!=stop-start or r.headers.get('Content-Range')!=f'bytes {start}-{stop-1}/{size}'):
                raise ValueError('bad Content-Range')
            return r.content
        except requests.RequestException:
            if attempt==4: raise
            time.sleep(attempt+1)

class Remote(io.RawIOBase):
    def __init__(self,url,size): self.url=url;self.size=size;self.pos=0;self.blocks={}
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=(0 if whence==0 else self.pos if whence==1 else self.size)+offset
        if self.pos<0: raise ValueError('negative seek')
        return self.pos
    def read(self,n=-1):
        n=min(self.size-self.pos,self.size-self.pos if n<0 else n);out=bytearray()
        while n>0:
            i,j=divmod(self.pos,65536)
            if i not in self.blocks:self.blocks[i]=request(self.url,i*65536,min((i+1)*65536,self.size),self.size)
            piece=self.blocks[i][j:j+n]
            if not piece:raise EOFError('empty block')
            out.extend(piece);self.pos+=len(piece);n-=len(piece)
        return bytes(out)

def parse_trials(rows):
    trials=[];prompt=None
    for e in rows:
        v=int(e['value'])
        if 18<=v<=23:
            trials.append({'onset':float(e['onset']),'condition':v,'arousal':[],'valence':[],'order':[]});prompt=None
        elif trials and v in (48,49):
            prompt='arousal' if v==49 else 'valence';trials[-1]['order'].append(prompt)
        elif trials and 1<=v<=9 and prompt:
            trials[-1][prompt].append(v);prompt=None
    if len(trials)<30:raise ValueError('fewer than 30 trials')
    chosen=trials[:30]
    if any(len(t['arousal'])!=1 or len(t['valence'])!=1 or t['order']!=['arousal','valence'] for t in chosen):raise ValueError('ambiguous rating')
    if any(b['onset']<=a['onset']+2 for a,b in zip(chosen,chosen[1:])):raise ValueError('overlapping trials')
    return chosen

def subject(i,metadata_only=False):
    sid=f'sub-{i:03}';stem=f'{sid}_task-EmotionProcessingandRegulationtask';p=f'{sid}/eeg/{stem}'
    ptr=request(BASE+p+'_eeg.set').decode().strip().split('/')[-1]
    match=re.fullmatch(r'SHA256E-s(\d+)--[0-9a-f]{64}\.set',ptr)
    if not match:raise ValueError('invalid annex object')
    size=int(match.group(1));url='https://nemar.s3.us-east-2.amazonaws.com/on006866/objects/'+ptr
    def tsv(suffix): return list(csv.DictReader(io.StringIO(request(BASE+p+suffix).decode('utf-8-sig')),delimiter='\t'))
    events=tsv('_events.tsv');channels=tsv('_channels.tsv');trials=parse_trials(events)
    remote=Remote(url,size);info=loadmat(remote,variable_names=['nbchan','pnts','srate','chanlocs','event'],simplify_cells=True)
    labels=[c['labels'] for c in info['chanlocs']];names=[c['name'] for c in channels]
    if labels!=names or float(info['srate'])!=1000:raise ValueError('channel/rate mismatch')
    nchan=int(info['nbchan']);npoints=int(info['pnts']);ids=[names.index(n) for n in CHANNELS]
    if len(names)!=nchan or any(channels[j]['units']!='uV' for j in ids):raise ValueError('channel units mismatch')
    if len(events)!=len(info['event']):raise ValueError('event count mismatch')
    for e,m in zip(events,info['event']):
        if int(float(e['value']))!=int(float(m['type'])) or abs(float(m['latency'])-1-float(e['onset'])*1000)>0.05:raise ValueError('event mismatch')
    reader=MatFile5Reader(remote);reader.initialize_read();reader.read_file_header()
    while not reader.end_of_stream():
        hdr,next_pos=reader.read_var_header()
        if hdr.name==b'data':
            if list(hdr.dims)!=[nchan,npoints] or hdr.mclass!=7:raise ValueError('data dimensions/type mismatch')
            offset=remote.tell();tag=remote.read(8)
            if len(tag)!=8 or struct.unpack('<II',tag)!=(7,nchan*npoints*4):raise ValueError('compressed or unsupported data block')
            data_start=offset+8;break
        remote.seek(next_pos)
    else:raise ValueError('missing data matrix')
    if data_start+nchan*npoints*4>size:raise ValueError('data extends past file')
    if metadata_only:return {'subject':sid,'nchan':nchan,'npoints':npoints,'offset':data_start,'event_count':len(events)}
    CACHE.mkdir(exist_ok=True);cache=CACHE/(sid+'.npz')
    if cache.exists():
        with np.load(cache,allow_pickle=False) as saved:
            if saved['object'].item()!=ptr or saved['commit'].item()!=COMMIT:raise ValueError('cache provenance mismatch')
            X=saved['features']
        if X.shape!=(30,40) or not np.isfinite(X).all():raise ValueError('cache features invalid')
    else:
        def one(t):
            sample=round(t['onset']*1000);start=data_start+sample*nchan*4;stop=start+2000*nchan*4
            if sample<0 or sample+2000>npoints:raise ValueError('EEG window out of bounds')
            b=request(url,start,stop,size);x=np.frombuffer(b,'<f4').reshape(2000,nchan)[:,ids].T
            if not np.isfinite(x).all():raise ValueError('nonfinite EEG')
            f,psd=welch(x,fs=1000,nperseg=2000,axis=-1)
            return np.log(np.maximum(np.stack([psd[:,(f>=a)&(f<b)].mean(axis=1) for a,b in [(1,4),(4,8),(8,13),(13,30)]],axis=1),1e-20)).ravel()
        with ThreadPoolExecutor(max_workers=4) as pool:X=np.array(list(pool.map(one,trials)))
        np.savez_compressed(cache,features=X,object=ptr,commit=COMMIT)
    print(f'{sid}: metadata verified and 30 features ready',flush=True)
    return {'X':X,'y':np.array([t['arousal'][0] for t in trials],float),'condition':np.array([t['condition'] for t in trials])}

def evaluate(data):
    X=np.concatenate([data[i]['X'] for i in TRAIN]);y=np.concatenate([data[i]['y'] for i in TRAIN]);conditions=np.concatenate([data[i]['condition'] for i in TRAIN]);avg=y.mean()
    cm={c:y[conditions==c].mean() for c in set(conditions)}
    model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(X,y)
    errors={k:[] for k in ['source_mean','source_condition','source_eeg','cal_mean','cal_condition','cal_eeg']};unseen=0
    for i in TARGET:
        v=data[i];base={'mean':np.full(30,avg),'condition':np.array([cm.get(c,avg) for c in v['condition']]),'eeg':model.predict(v['X'])}
        unseen+=sum(c not in cm for c in v['condition'][10:])
        for name,pred in base.items():
            correction=(10/20)*np.mean(v['y'][:10]-pred[:10])
            errors['source_'+name].append(float(np.mean(abs(pred[10:]-v['y'][10:]))))
            errors['cal_'+name].append(float(np.mean(abs(pred[10:]+correction-v['y'][10:]))))
    return {'source_subjects':len(TRAIN),'target_subjects':len(TARGET),'train_trials':len(y),'calibration_trials_per_target':10,'test_trials_per_target':20,'target_macro_mae':{k:round(float(np.mean(v)),4) for k,v in errors.items()},'unknown_condition_test_trials':int(unseen)}

if __name__=='__main__':
    import sys
    if '--metadata' in sys.argv:
        print(json.dumps(subject(2,metadata_only=True)))
    else:
        data={i:subject(i) for i in list(TRAIN)+list(TARGET)}
        print(json.dumps(evaluate(data)))
