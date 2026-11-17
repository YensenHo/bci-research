"""Development-only sensitivity run with film-disjoint clips and mixed rates."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import numpy as np
import requests
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import faced_development_20261117 as dev

OLD=dev.ROOT/'development_v1'; NEW=dev.ROOT/'development_v2'; NEW.mkdir(exist_ok=True)
TRAIN=list(range(8,20))+list(range(61,65)); VALID=list(range(20,24))+[65,66]
TV=[1,4,10,17,20,23,26]; VV=[2,5,11,18,21,24,27]
CATEGORY={1:'anger',2:'anger',4:'disgust',5:'disgust',10:'sadness',11:'sadness',17:'amusement',18:'amusement',20:'inspiration',21:'inspiration',23:'joy',24:'joy',26:'tenderness',27:'tenderness'}

def subject(i):
    old=OLD/f'sub-{i:03}.json'
    if old.exists():
        result=json.loads(old.read_text(encoding='utf8'))
        wanted=TV if i in TRAIN else VV
        result['rows']=[r for r in result['rows'] if r['video'] in wanted]
        if len(result['rows'])!=len(wanted):raise ValueError('old cache missing required videos')
        return result
    return dev.subject(i)

def main():
    assert not set(TRAIN)&set(VALID) and not set(TV)&set(VV)
    url=f'https://raw.githubusercontent.com/nemarDatasets/nm000112/{dev.COMMIT}/stimuli/stimuli.tsv'
    from io import StringIO
    response=requests.get(url,timeout=30);response.raise_for_status();stim=pd.read_csv(StringIO(response.text),sep='\t').set_index('stim_id')
    titles_train=set(stim.loc[TV,'source_title']); titles_valid=set(stim.loc[VV,'source_title'])
    if titles_train & titles_valid: raise ValueError('shared source film')
    dev.TRAIN_SUBJECTS=TRAIN;dev.VALIDATION_SUBJECTS=VALID;dev.TRAIN_VIDEOS=TV;dev.VALIDATION_VIDEOS=VV;dev.CACHE=NEW
    with ThreadPoolExecutor(max_workers=4) as pool:subjects=list(pool.map(subject,TRAIN+VALID))
    tr=[r for s in subjects if int(s['subject'][-3:]) in TRAIN for r in s['rows']]
    va=[r for s in subjects if int(s['subject'][-3:]) in VALID for r in s['rows']]
    yt=np.array([r['arousal'] for r in tr]);yv=np.array([r['arousal'] for r in va])
    preds={'training_mean':np.full(len(va),yt.mean())}
    for key in ['x','calibrated']:
        m=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit([r[key] for r in tr],yt)
        preds[key]=m.predict([r[key] for r in va])
    ctrain=np.array([CATEGORY[r['video']] for r in tr]);cval=[CATEGORY[r['video']] for r in va]
    means={c:float(yt[ctrain==c].mean()) for c in set(ctrain)}
    metadata=np.array([means[c] for c in cval]);preds['known_stimulus_category']=metadata
    residual=yt-np.array([means[c] for c in ctrain])
    for key in ['x','calibrated']:
        m=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit([r[key] for r in tr],residual)
        preds['category_plus_'+key]=metadata+m.predict([r[key] for r in va])
    def metrics(mask):return {k:float(np.mean(abs(v[mask]-yv[mask]))) for k,v in preds.items()}
    subject_rates={s['subject']:s['fs'] for s in subjects};per={}
    for i in VALID:
        sid=f'sub-{i:03}';mask=np.array([r['subject']==sid for r in va]);per[sid]=metrics(mask)
    rates={}
    for fs in (250,1000):
        mask=np.array([subject_rates[r['subject']]==fs for r in va]);rates[str(fs)]=metrics(mask)
    result={'warning':'development only; same-film sources excluded; participants of earlier pilot were previously examined; no independent test',
      'train_subjects':TRAIN,'valid_subjects':VALID,'train_videos':TV,'valid_videos':VV,'counts':[len(tr),len(va)],
      'all':metrics(np.ones(len(va),bool)),'by_subject':per,'by_sampling_rate':rates,'subject_rates':subject_rates,
      'rows':[{'subject':r['subject'],'video':r['video'],'arousal':r['arousal'],'quality':r['quality']} for r in va],
      'predictions':{k:v.tolist() for k,v in preds.items()}}
    (NEW/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'counts':result['counts'],'all':result['all'],'by_sampling_rate':rates},ensure_ascii=False))

if __name__=='__main__':main()
