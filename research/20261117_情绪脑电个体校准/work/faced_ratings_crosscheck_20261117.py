from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from io import BytesIO
import hashlib, json, time
import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat

SOURCE='68d43642c8d8af886d40c542e15c947397bda9f5'
ROOT=Path(__file__).parent / 'faced_audit_v1.1.4'
KEYS=['Joy','Tenderness','Inspiration','Amusement','Anger','Disgust','Fear','Sadness','Arousal','Valence','Familiarity','Liking']

def check(i):
    sid=f'sub-{i:03}'
    url=f'https://raw.githubusercontent.com/jianjiez100-sys/AMA-EEG/{SOURCE}/After_remarks/sub{i:03}/After_remarks.mat'
    for attempt in range(3):
        try:
            response=requests.get(url,timeout=30); response.raise_for_status(); break
        except requests.RequestException:
            if attempt == 2: raise
            time.sleep(attempt+1)
    items=loadmat(BytesIO(response.content),squeeze_me=True,struct_as_record=False)['After_remark']
    records={int(x.vid):np.asarray(x.score,dtype=float).reshape(-1) for x in items}
    if len(items)!=28 or set(records)!=set(range(1,29)) or any(len(v)!=12 for v in records.values()):
        raise ValueError(f'{sid} source does not contain exactly 28 unique 12-score records')
    events=pd.read_csv(ROOT / f'{sid}_task-watchingVideoClips_events.tsv',sep='\t')
    events=events[events.video_index.notna()]
    errors=[]
    for row in events.itertuples(index=False):
        wanted=records[int(row.video_index)]; observed=np.asarray([getattr(row,k) for k in KEYS],float)
        errors.extend(np.abs(wanted-observed).tolist())
    return {'subject':sid,'mat_sha256':hashlib.sha256(response.content).hexdigest(),
            'n_scores':len(errors),'max_abs_difference':float(max(errors)),'over_1e_10':sum(v>1e-10 for v in errors)}

def main():
    with ThreadPoolExecutor(max_workers=8) as pool: rows=list(pool.map(check,range(123)))
    result={'source_commit':SOURCE,'nemar_commit':'b19270dcf75170fc7588ddba3b1e9129a01e9e43',
            'subjects':len(rows),'scores':sum(x['n_scores'] for x in rows),
            'max_abs_difference':max(x['max_abs_difference'] for x in rows),'mismatches_1e_10':sum(x['over_1e_10'] for x in rows),'details':rows}
    (ROOT/'ratings_crosscheck.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='details'}))

if __name__=='__main__': main()
