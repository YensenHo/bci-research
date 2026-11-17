"""Independent SciPy MAT metadata and byte-range signal cross-check; no ratings output."""
import csv
import io
import json
import re
import time
import numpy as np
import requests
from scipy.io import loadmat

KEY = 'SHA256E-s893431584--6340852e04b548cddebfa8ea59e1771277cef36b83bac432c9cefe35554ee97b.set'
URL = 'https://nemar.s3.us-east-2.amazonaws.com/on006866/objects/' + KEY
SIZE = int(re.search(r'-s(\d+)--', KEY).group(1))


def fetch(start, stop):
    for attempt in range(4):
        try:
            r = requests.get(URL, headers={'Range': f'bytes={start}-{stop-1}'}, timeout=(10, 45))
            r.raise_for_status()
            if (r.status_code != 206 or len(r.content) != stop-start or
                r.headers.get('Content-Range') != f'bytes {start}-{stop-1}/{SIZE}'):
                raise ValueError('invalid HTTP byte range')
            return r.content
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


class RemoteMat(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.blocks = {}
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = (0 if whence == 0 else self.pos if whence == 1 else SIZE) + offset
        if self.pos < 0: raise ValueError('negative seek')
        return self.pos
    def read(self, n=-1):
        n = min(SIZE - self.pos, SIZE - self.pos if n < 0 else n)
        out = bytearray()
        while n > 0:
            i, j = divmod(self.pos, 65536)
            if i not in self.blocks:
                self.blocks[i] = fetch(i * 65536, min((i+1) * 65536, SIZE))
            piece = self.blocks[i][j:j+n]
            if not piece: raise EOFError('empty block')
            out.extend(piece); self.pos += len(piece); n -= len(piece)
        return bytes(out)


def main(events_path, channels_path):
    mat = RemoteMat()
    info = loadmat(mat, variable_names=['nbchan','pnts','srate','ref','chanlocs','event'], simplify_cells=True)
    with open(channels_path, encoding='utf-8-sig', newline='') as f:
        channels = list(csv.DictReader(f, delimiter='\t'))
    with open(events_path, encoding='utf-8-sig', newline='') as f:
        events = list(csv.DictReader(f, delimiter='\t'))
    labels = [x['labels'] for x in info['chanlocs']]
    file_labels = [x['name'] for x in channels]
    same_labels = labels == file_labels
    if not same_labels: raise ValueError('channel order mismatch')
    fs = float(info['srate'])
    if (int(info['nbchan']), int(info['pnts']), fs) != (69, 3144334, 1000): raise ValueError('unexpected dimensions')
    me = info['event']
    if len(events) != len(me): raise ValueError('event counts differ')
    # EEGLAB latency is one-based; BIDS onset is seconds, allowing fractional samples.
    offsets = np.array([float(m['latency'])-1-float(e['onset'])*fs for m,e in zip(me,events)])
    code_match = all(str(int(float(m['type']))) == str(int(float(e['value']))) for m,e in zip(me,events))
    if not code_match or np.max(np.abs(offsets)) > 0.05: raise ValueError('events mismatch')
    print(json.dumps({'channel_count':len(labels), 'channel_order_equal':same_labels,
      'first_64_then_aux':labels[64:] == ['HEO','VEO','EKG','GSR','Trigger'], 'event_count':len(events),
      'event_codes_equal':code_match, 'max_sample_offset':float(np.max(np.abs(offsets))),
      'sampling_hz':fs, 'reference':str(info['ref']), 'blocks':len(mat.blocks)}))

if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])
