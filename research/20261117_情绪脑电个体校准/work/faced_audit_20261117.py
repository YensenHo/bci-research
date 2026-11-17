from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from io import BytesIO
import hashlib, json, time
import pandas as pd
import requests

COMMIT = 'b19270dcf75170fc7588ddba3b1e9129a01e9e43'
ROOT = Path(__file__).parent / 'faced_audit_v1.1.4'
ROOT.mkdir(exist_ok=True)

def fetch(i):
    sid = f'sub-{i:03}'
    name = f'{sid}_task-watchingVideoClips_events.tsv'
    path = ROOT / name
    url = f'https://raw.githubusercontent.com/nemarDatasets/nm000112/{COMMIT}/{sid}/eeg/{name}'
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.content
            frame = pd.read_csv(BytesIO(data), sep='\t')
            assert {'video_index', 'onset', 'duration', 'Arousal', 'Valence'} <= set(frame.columns)
            path.write_bytes(data)
            return sid, hashlib.sha256(data).hexdigest(), frame
        except Exception:
            if attempt == 2: raise
            time.sleep(attempt + 1)

def main():
    with ThreadPoolExecutor(max_workers=8) as executor:
        data = list(executor.map(fetch, range(123)))
    results = []
    for sid, sha, frame in data:
        video = frame.loc[frame.video_index.notna()].copy()
        index = pd.to_numeric(video.video_index, errors='coerce')
        score = video[['Arousal', 'Valence']].apply(pd.to_numeric, errors='coerce')
        onset = pd.to_numeric(video.onset, errors='coerce')
        duration = pd.to_numeric(video.duration, errors='coerce')
        results.append({'subject': sid, 'sha256': sha, 'videos':len(video),
            'unique_video_ids':int(index.nunique()), 'missing_ids': sorted(set(range(1,29))-set(index)),
            'missing_arousal':int(score.Arousal.isna().sum()), 'missing_valence':int(score.Valence.isna().sum()),
            'out_of_range':int(((score < 0) | (score > 7)).any(axis=1).sum()),
            'invalid_time':int((onset.isna() | duration.isna() | (onset < 0) | (duration <= 0)).sum())})
    totals = {key:sum(r[key] for r in results) for key in ('missing_arousal','missing_valence','out_of_range','invalid_time')}
    totals['subjects'] = len(results)
    totals['subjects_28_unique'] = sum(r['videos']==28 and r['unique_video_ids']==28 and not r['missing_ids'] for r in results)
    report = {'version':'v1.1.4', 'commit':COMMIT, 'summary':totals, 'subjects':results}
    (ROOT / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(totals, ensure_ascii=False))
    for row in results:
        if row['missing_ids'] or row['missing_arousal'] or row['missing_valence'] or row['out_of_range'] or row['invalid_time']:
            print(json.dumps(row, ensure_ascii=False))

if __name__ == '__main__':
    main()
