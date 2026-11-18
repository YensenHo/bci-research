"""Development-only fixed 6/2/2 subject split; no individual data written."""
import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier

def main(root):
    z = loadmat(root / 'Impedance.mat')['Impedance']
    assert z.shape == (8, 10, 2, 102) and np.isfinite(z).all()
    assert z[:, :, 0, :].mean() / z[:, :, 1, :].mean() > 10
    t = np.arange(500) / 250
    freqs = 9.25 + np.arange(12) * .5
    basis = np.exp(-2j * np.pi * np.outer(np.r_[freqs, 2 * freqs], t))
    data = {}
    for i in range(1, 11):
        eeg = loadmat(root / f'S{i:03}.mat')['data']
        assert eeg.shape == (8, 710, 2, 10, 12) and np.isfinite(eeg).all()
        X, q, imp, y = [], [], [], []
        for e in range(2):
            for b in range(10):
                for label in range(12):
                    x = eeg[:, 125:625, e, b, label]
                    X.append(np.abs(x @ basis.T))
                    q.append(np.std(x, axis=1)); imp.append(z[:, b, e, i-1]); y.append(label)
        data[i] = tuple(np.asarray(a) for a in (X,q,imp,y))
    train = [data[i] for i in range(1, 7)]
    X = np.concatenate([d[0] for d in train]); q = np.concatenate([d[1] for d in train]); imp = np.concatenate([d[2] for d in train]); y = np.concatenate([d[3] for d in train])
    qth, ith = np.quantile(q,.8), np.quantile(imp,.8)
    model = make_pipeline(StandardScaler(),RidgeClassifier()).fit(X.reshape(len(X),-1),y)
    for group, ids in [('validation',(7,8)),('test',(9,10))]:
        scores = {k: [] for k in ('none','impedance','waveform')}; rates = {k: [] for k in ('impedance','waveform')}
        for i in ids:
            x,q,z,y = data[i]
            masks = {'none':np.zeros(q.shape,dtype=bool),'impedance':z>ith,'waveform':q>qth}
            for k,m in masks.items():
                pred = model.predict(np.where(m[:,:,None],0,x).reshape(len(x),-1))
                scores[k].append(float(np.mean(pred==y)))
                if k in rates: rates[k].append(float(m.mean()))
        print(group,'accuracy macro', {k:round(float(np.mean(v)),4) for k,v in scores.items()},'mask fraction', {k:round(float(np.mean(v)),4) for k,v in rates.items()})

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('data_dir',type=Path);main(p.parse_args().data_dir)
