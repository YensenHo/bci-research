"""Paired subject-level aggregation; multiple seeds are not independent subjects."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('inputs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
data=[json.loads(path.read_text(encoding='utf8')) for path in args.inputs]
if len(data)!=3 or {d['seed'] for d in data}!={42,123,2024}:raise ValueError('expected exactly seeds 42, 123 and 2024')
reference=data[0]
if len(set(reference['subjects']))!=len(reference['subjects']): raise ValueError('duplicate subject identifiers')
for d in data:
    if d.get('status')!='completed' or any(d[key]!=reference[key] for key in ('task','subjects','frequencies','source_sha256','device','epochs','patience','protocol')):
        raise ValueError('incomplete or incompatible runs')
    if set(d['results'])!={str(s) for s in d['subjects']}: raise ValueError('missing subjects')
result={'task':reference['task'],'subjects':reference['subjects'],'seeds':[42,123,2024],
        'source_sha256':reference['source_sha256'],'comparisons':{},
        'warning':'Exploratory development gate; repeated seeds are averaged within subject, not independent people.'}
for variant,control in (('B','A'),('C','A'),('D','A'),('C','D')):
    diffs=[]; values=[]
    for sid in reference['subjects']:
        subject=[d['results'][str(sid)]['models'] for d in data]
        diffs.append(float(np.mean([m[variant]['test']['accuracy']-m[control]['test']['accuracy'] for m in subject])))
        values.append(float(np.mean([m[variant]['test']['accuracy'] for m in subject])))
    n=len(diffs); change=float(np.mean(diffs)); accuracy=float(np.mean(values)); improved=sum(x>0 for x in diffs)
    result['comparisons'][variant+'_vs_'+control]={'n_subjects':n,'mean_difference_pp':100*change,'mean_accuracy':accuracy,
        'improved':improved,'declined':sum(x<0 for x in diffs),'tied':sum(x==0 for x in diffs),
        'per_subject_difference_pp':(100*np.asarray(diffs)).tolist(),
        'development_gate':(n==10 and improved>=7 and change>=.05 and accuracy>=.10) if reference['task']=='real_40' and variant in ('B','C') and control=='A' else None}
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
