"""Offline representation diagnosis; per-subject blocks stay separate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from gate1_ssvep import cca_predict, fit_threshold, encode, shuffle_events


def normalize_trials(x):
    x = x - x.mean(axis=-1, keepdims=True)
    return x / np.maximum(np.sqrt(np.mean(x*x, axis=-1, keepdims=True)), 1e-8)


def ridge(train, y, val, vy, test, classes):
    """Dual ridge with training-only scaling and validation-only penalty choice."""
    tr = train.reshape(len(train), -1).astype('float64')
    va = val.reshape(len(val), -1).astype('float64')
    te = test.reshape(len(test), -1).astype('float64')
    mean, std = tr.mean(0), tr.std(0)
    std = np.where(std > 1e-8, std, 1.0)
    tr, va, te = ((z - mean)/std for z in (tr, va, te))
    # Intercept in feature space, independent of the regularised feature scaling.
    tr, va, te = (np.c_[z, np.ones(len(z))] for z in (tr, va, te))
    target = np.eye(classes)[y]
    gram = tr@tr.T
    best = None
    for penalty in (.1, 1., 10., 100.):
        weights = np.linalg.solve(gram + penalty*np.eye(len(tr)), target)
        train_pred = (gram@weights).argmax(1)
        val_pred = (va@tr.T@weights).argmax(1)
        score = float(np.mean(val_pred == vy))
        if best is None or score > best[0]:
            best = (score, penalty, float(np.mean(train_pred == y)), weights)
    return dict(train_accuracy=best[2], validation_accuracy=best[0], penalty=best[1],
                test_predictions=(te@tr.T@best[3]).argmax(1))


def run_subject(data, frequencies, train_blocks, validation_block, test_blocks, seed):
    """data: blocks, targets, channels, samples."""
    blocks, classes, channels, samples = data.shape
    if samples != 250 or classes != len(frequencies):
        raise ValueError('expected one-second windows and aligned frequencies')
    labels = np.tile(np.arange(classes), blocks)
    x = data.reshape(-1, channels, samples).astype('float32')
    def ids(block_ids): return np.concatenate([np.arange(b*classes,(b+1)*classes) for b in block_ids])
    train_i, val_i, test_i = ids(train_blocks), ids(validation_block), ids(test_blocks)
    if len(set(train_i)&set(val_i)) or len(set(train_i)&set(test_i)) or len(set(val_i)&set(test_i)):
        raise ValueError('overlapping splits')
    threshold=fit_threshold(x[train_i])
    events=encode(x,threshold)
    signed=(events.reshape(len(x),samples,channels,2)[...,0]-events.reshape(len(x),samples,channels,2)[...,1]).transpose(0,2,1)
    scrambled=shuffle_events(events[test_i],seed)
    shuffled_signed=(scrambled.reshape(len(test_i),samples,channels,2)[...,0]-scrambled.reshape(len(test_i),samples,channels,2)[...,1]).transpose(0,2,1)
    test_y=labels[test_i]
    row={'test_trials':len(test_i), 'threshold':threshold.tolist(),
         'event_count':float(events[test_i].sum()/len(test_i)),
         'cca_raw':float(np.mean(cca_predict(normalize_trials(x[test_i]),frequencies)==test_y)),
         'cca_signed_events':float(np.mean(cca_predict(normalize_trials(signed[test_i]),frequencies)==test_y)),
         'cca_shuffled_events':float(np.mean(cca_predict(normalize_trials(shuffled_signed),frequencies)==test_y))}
    for name,features in [('raw_flat',x),('signed_events_flat',signed),('event_counts',events.sum(axis=1))]:
        r=ridge(features[train_i],labels[train_i],features[val_i],labels[val_i],features[test_i],classes)
        predictions=r.pop('test_predictions')
        r['test_accuracy']=float(np.mean(predictions==test_y))
        row[name]=r
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--freq-phase',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--subjects',type=int,nargs='+',default=list(range(1,11)))
    args=p.parse_args()
    with np.load(args.input,allow_pickle=False) as f: all_data=np.asarray(f['eeg_1'])
    if all_data.shape!=(len(args.subjects),6,40,6,250): raise ValueError(f'unexpected benchmark shape {all_data.shape}')
    freqs=np.asarray(loadmat(args.freq_phase)['freqs']).ravel()
    if len(freqs)!=40 or len(np.unique(freqs))!=40: raise ValueError('invalid frequency mapping')
    data_iter=((str(i),v) for i,v in zip(args.subjects,all_data))
    train,val,test=[0,1,2,3],[4],[5]
    info='64-channel benchmark, 6 selected channels; blocks 1–4/5/6; IDs must match NPZ order'
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.input,args.freq_phase)}
    result={'format':'benchmark','protocol':info,'seed':args.seed,'source_sha256':source_hashes,'results':{},
            'warning':'CCA and flattening consume whole windows offline; model/time control is diagnostic only; CCA on transformed data does not prove SNN can use it.'}
    for sid,data in data_iter:
        result['results'][sid]=run_subject(data,freqs,train,val,test,args.seed)
        print('completed subject',sid,flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    result['status']='completed'
    args.output.write_text(json.dumps(result,indent=2),encoding='utf8')


if __name__=='__main__': main()
