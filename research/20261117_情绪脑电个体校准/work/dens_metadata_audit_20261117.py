from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
import json, time
import pandas as pd
import requests

COMMIT='beb73da6fde5c043d42145637db9f5bce7e3fe12'
ROOT=Path(__file__).parent/'dens_audit_v1.0.0';ROOT.mkdir(exist_ok=True)

def get(url):
    for i in range(3):
        try:
            r=requests.get(url,timeout=30);r.raise_for_status();return r
        except requests.RequestException:
            if i==2:raise
            time.sleep(i+1)

def check(sid, paths):
    base=f'https://raw.githubusercontent.com/nemarDatasets/on003751/{COMMIT}/{sid}/'
    evt=pd.read_csv(StringIO(get(base+f'eeg/{sid}_task-emotion_events.tsv').text),sep='\t')
    beh_path=f'{sid}/beh/{sid}_task-Emotion_beh.tsv'
    beh=pd.read_csv(StringIO(get(base+f'beh/{sid}_task-Emotion_beh.tsv').text),sep='\t') if beh_path in paths else None
    meta_path=f'{sid}/eeg/{sid}_task-emotion_eeg.json'
    meta=get(base+f'eeg/{sid}_task-emotion_eeg.json').json() if meta_path in paths else None
    fs=float(meta['SamplingFrequency']) if meta else None;duration=float(meta['RecordingDuration']) if meta else None
    on=pd.to_numeric(evt.onset,errors='coerce');stim=evt[evt.trial_type.eq('stm')]
    # Marker label = source stimulus name + '_' + 1-based presentation number.
    labels=[str(x).rsplit('_',1)[0] for x in stim.label];ratings=[Path(str(x)).stem for x in beh.stimuliName] if beh is not None else []
    return {'subject':sid,'sampling_rate':fs,'recording_duration_s':duration,'max_event_onset_field':float(on.max()),
      'events_past_recording_if_seconds':int((on>duration).sum()) if meta else None,'events_past_recording_if_samples':int((on/fs>duration+1).sum()) if meta else None,
      'stimulus_events':len(stim),'behaviour_rows':len(beh) if beh is not None else 0,'ordered_names_match':labels==ratings if beh is not None else None,
      'valid_arousal':int(pd.to_numeric(beh.arousal,errors='coerce').notna().sum()) if beh is not None else 0}

def main():
    url=f'https://api.github.com/repos/nemarDatasets/on003751/git/trees/{COMMIT}?recursive=1';tree=get(url).json()
    if tree.get('truncated'):raise ValueError('incomplete tree')
    subjects=sorted(x['path'] for x in tree['tree'] if x['type']=='tree' and x['path'].startswith('sub-') and '/' not in x['path'])
    paths={x['path'] for x in tree['tree']}
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(lambda sid:check(sid,paths),subjects))
    report={'commit':COMMIT,'subjects':len(rows),'all_ordered_match':sum(x['ordered_names_match'] is True for x in rows),
      'missing_eeg_metadata':sum(x['sampling_rate'] is None for x in rows),
      'any_time_past_seconds':sum((x['events_past_recording_if_seconds'] or 0)>0 for x in rows),
      'any_time_past_samples':sum((x['events_past_recording_if_samples'] or 0)>0 for x in rows),'details':rows}
    (ROOT/'metadata_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='details'}))
    print(json.dumps([r for r in rows if r['ordered_names_match'] is not True or r['events_past_recording_if_samples']]))

if __name__=='__main__':main()
