"""Exploratory crossed-CV fixed baseline weights, with a mismatched control."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from faced_crossed_cv_20261117 import ROOT, PEOPLE, VIDEOS

WEIGHTS=(0.,0.25,0.5,0.75,1.)

def main():
    rows=[]; baselines={}
    for i in PEOPLE:
        sid=f'sub-{i:03}';p=ROOT/('development_v1' if i<20 else 'development_v2')/f'{sid}.json'
        person=[r for r in json.loads(p.read_text(encoding='utf8'))['rows'] if r['video'] in VIDEOS]
        assert len(person)==7 and {r['video'] for r in person}==set(VIDEOS)
        base=np.array(person[0]['x'])-np.array(person[0]['calibrated'])
        assert all(np.allclose(np.array(r['x'])-np.array(r['calibrated']),base,rtol=1e-10,atol=1e-10) for r in person)
        baselines[sid]=base;rows.extend(person)
    predictions=[]
    for i in PEOPLE:
        sid=f'sub-{i:03}';others=[f'sub-{j:03}' for j in PEOPLE if j!=i];wrong=others[PEOPLE.index(i)%len(others)]
        assert sid!=wrong and wrong in baselines
        for v in VIDEOS:
            train=[r for r in rows if r['subject']!=sid and r['video']!=v]
            test=[r for r in rows if r['subject']==sid and r['video']==v]
            assert len(train)==90 and len(test)==1
            target=np.array([r['arousal'] for r in train]);pred={'mean':float(target.mean())}
            for w in WEIGHTS:
                xtrain=[np.array(r['x'])-w*baselines[r['subject']] for r in train]
                model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(xtrain,target)
                pred[f'w{w:g}']=float(model.predict([np.array(test[0]['x'])-w*baselines[sid]])[0])
                if w==1.:
                    pred['wrong_baseline']=float(model.predict([np.array(test[0]['x'])-baselines[wrong]])[0])
            predictions.append({'subject':sid,'video':v,'truth':test[0]['arousal'],'predictions':pred})
    keys=['mean']+[f'w{w:g}' for w in WEIGHTS]+['wrong_baseline']
    def scores(items):return {k:float(np.mean([abs(r['truth']-r['predictions'][k]) for r in items])) for k in keys}
    per_person={f'sub-{i:03}':scores([r for r in predictions if r['subject']==f'sub-{i:03}']) for i in PEOPLE}
    per_video={str(v):scores([r for r in predictions if r['video']==v]) for v in VIDEOS}
    result={'warning':'exploratory fixed weights, 16 development subjects and 7 films, not an independent test',
      'overall_mae':scores(predictions),'per_person_mae':per_person,'per_video_mae':per_video,'predictions':predictions}
    (ROOT/'development_v2'/'baseline_weight_cv_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'overall_mae':result['overall_mae'],'n_people':len(per_person),'n_films':len(per_video)}))

if __name__=='__main__':main()
