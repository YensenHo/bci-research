"""Development-side crossed leave-person-and-film-out check; not a final test."""
from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).parent/'faced_audit_v1.1.4'
PEOPLE=list(range(8,20))+list(range(61,65))
VIDEOS=[1,4,10,17,20,23,26]

def main():
    rows=[]
    for i in PEOPLE:
        path=ROOT/('development_v1' if i<20 else 'development_v2')/f'sub-{i:03}.json'
        records=json.loads(path.read_text(encoding='utf8'))['rows']
        chosen=[r for r in records if r['video'] in VIDEOS]
        assert len(chosen)==7 and {r['video'] for r in chosen}==set(VIDEOS)
        rows.extend(chosen)
    assert len(rows)==112
    predictions=[]
    for i in PEOPLE:
        person=f'sub-{i:03}'
        for video in VIDEOS:
            train=[r for r in rows if r['subject']!=person and r['video']!=video]
            test=[r for r in rows if r['subject']==person and r['video']==video]
            assert len(train)==90 and len(test)==1
            yt=np.array([r['arousal'] for r in train]);pred={'mean':float(yt.mean())}
            for key in ('x','calibrated'):
                model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit([r[key] for r in train],yt)
                pred[key]=float(model.predict([test[0][key]])[0])
            predictions.append({'subject':person,'video':video,'truth':test[0]['arousal'],'predictions':pred})
    def metrics(items):
        return {k:float(np.mean([abs(r['truth']-r['predictions'][k]) for r in items])) for k in ('mean','x','calibrated')}
    per_person={f'sub-{i:03}':metrics([r for r in predictions if r['subject']==f'sub-{i:03}']) for i in PEOPLE}
    per_video={str(v):metrics([r for r in predictions if r['video']==v]) for v in VIDEOS}
    output={'warning':'exploratory crossed CV, 16 people and 7 films, not 112 independent samples or a held-out test',
      'overall_mae':metrics(predictions),'per_person_mae':per_person,'per_video_mae':per_video,
      'calibration_worse_person_count':sum(v['calibrated']>v['x'] for v in per_person.values()),'predictions':predictions}
    dest=ROOT/'development_v2'/'crossed_cv_results.json';dest.write_text(json.dumps(output,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in output.items() if k not in ('per_person_mae','per_video_mae','predictions')}))

if __name__=='__main__':main()
