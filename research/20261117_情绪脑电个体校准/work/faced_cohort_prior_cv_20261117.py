"""Exploratory leave-person-out EEG increment over same-video cohort ratings."""
from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).parent/'faced_audit_v1.1.4'
PEOPLE=list(range(8,20))+list(range(61,65));VIDEOS=[1,4,10,17,20,23,26]

def main():
    rows=[]
    for i in PEOPLE:
        p=ROOT/('development_v1' if i<20 else 'development_v2')/f'sub-{i:03}.json'
        data=json.loads(p.read_text(encoding='utf8'))['rows']
        rows.extend(r for r in data if r['video'] in VIDEOS)
    assert len(rows)==len(PEOPLE)*len(VIDEOS)
    predictions=[]
    for i in PEOPLE:
        sid=f'sub-{i:03}'; train=[r for r in rows if r['subject']!=sid];test=[r for r in rows if r['subject']==sid]
        assert len(train)==15*7 and len(test)==7
        video_means={v:np.mean([r['arousal'] for r in train if r['video']==v]) for v in VIDEOS}
        residual=[]
        for item in train:
            peers=[r['arousal'] for r in train if r['video']==item['video'] and r['subject']!=item['subject']]
            assert len(peers)==14
            residual.append(item['arousal']-np.mean(peers))
        baseline=np.array([video_means[r['video']] for r in test]);pred={'cohort_mean':baseline}
        for key in ('x','calibrated'):
            model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit([r[key] for r in train],residual)
            pred['cohort_plus_'+key]=baseline+model.predict([r[key] for r in test])
        for n,item in enumerate(test):
            predictions.append({'subject':sid,'video':item['video'],'truth':item['arousal'],
              'predictions':{name:float(y[n]) for name,y in pred.items()}})
    def metrics(items):return {key:float(np.mean([abs(r['truth']-r['predictions'][key]) for r in items])) for key in ('cohort_mean','cohort_plus_x','cohort_plus_calibrated')}
    per={f'sub-{i:03}':metrics([r for r in predictions if r['subject']==f'sub-{i:03}']) for i in PEOPLE}
    result={'warning':'development leave-person-out only; other participants rated each test video; no claim about unseen stimuli',
      'n_people':len(PEOPLE),'n_videos':len(VIDEOS),'n_predictions':len(predictions),'overall_mae':metrics(predictions),
      'per_person_mae':per,'predictions':predictions}
    (ROOT/'development_v2'/'cohort_prior_cv_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_person_mae','predictions')}))

if __name__=='__main__':main()
