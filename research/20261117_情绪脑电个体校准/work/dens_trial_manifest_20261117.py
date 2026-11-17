from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).parent/'dens_audit_v1.0.0'
COMMIT='beb73da6fde5c043d42145637db9f5bce7e3fe12'

def get(url):
    for i in range(3):
        try:
            r=requests.get(url,timeout=30);r.raise_for_status();return r.text
        except requests.RequestException:
            if i==2:raise
            time.sleep(i+1)

def extract(rec):
    sid=rec['subject'];base=f'https://raw.githubusercontent.com/nemarDatasets/on003751/{COMMIT}/{sid}/';fs=rec['sampling_rate']
    e=pd.read_csv(StringIO(get(base+f'eeg/{sid}_task-emotion_events.tsv')),sep='\t');b=pd.read_csv(StringIO(get(base+f'beh/{sid}_task-Emotion_beh.tsv')),sep='\t')
    s=e[e.trial_type.eq('stm')];names=[str(x).rsplit('_',1)[0] for x in s.label]
    if names!=[Path(str(x)).stem for x in b.stimuliName]:raise ValueError(f'{sid}: stimuli mismatch')
    bas=e[e.trial_type.eq('base')];end=e[e.trial_type.eq('bend')]
    baseline_s=None
    if len(bas)>0 and len(end)>0:
        start=float(bas.iloc[0].onset);ends=end[end.onset>start]
        if len(ends):baseline_s=(float(ends.iloc[0].onset)-start)/fs
    rows=[]
    for n,(_,ev) in enumerate(s.iterrows()):
        onset=float(ev.onset);idx=ev.name;later=e.iloc[idx+1:];next_stim=later[later.trial_type.eq('stm')];next_stim_sample=float(next_stim.iloc[0].onset) if len(next_stim) else float('inf')
        rating=later[(later.trial_type.eq('vlnc')) & (later.onset<next_stim_sample)];rating_sample=float(rating.iloc[0].onset) if len(rating) else None
        boundaries=e[(e.trial_type.eq('boundary')) & (e.onset>=onset) & (e.onset<onset+30*fs)]
        rows.append({'subject':sid,'stimulus':names[n],'latency_sample':onset,'candidate_time_s':(onset-1)/fs,
          'arousal':float(b.iloc[n].arousal),'rating_marker_delay_s':(rating_sample-onset)/fs if rating_sample is not None else None,
          'boundary_in_first30':len(boundaries)>0,'first30_within_recording':(onset-1)/fs+30<=rec['recording_duration_s']})
    return {'subject':sid,'baseline_duration_s':baseline_s,'trials':rows}

def main():
    audit=json.loads((ROOT/'metadata_audit.json').read_text());eligible=[r for r in audit['details'] if r['ordered_names_match'] is True and r['sampling_rate'] is not None]
    with ThreadPoolExecutor(max_workers=8) as pool:subjects=list(pool.map(extract,eligible))
    rows=[r for s in subjects for r in s['trials']]
    result={'warning':'candidate manifest, sampled onset conversion, alignment and permissions not fully verified',
      'subjects':len(subjects),'trials':len(rows),'baseline_at_least_10s':sum(s['baseline_duration_s'] is not None and s['baseline_duration_s']>=10 for s in subjects),
      'boundary_within_first30':sum(r['boundary_in_first30'] for r in rows),
      'first30_exceeds_recording':sum(not r['first30_within_recording'] for r in rows),
      'rating_marker_before_30s':sum(r['rating_marker_delay_s'] is not None and r['rating_marker_delay_s']<30 for r in rows),
      'missing_rating_marker':sum(r['rating_marker_delay_s'] is None for r in rows),'subjects_data':subjects}
    (ROOT/'trial_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='subjects_data'}))

if __name__=='__main__':main()
