import unittest
import numpy as np
import torch
from gate1_ssvep import cca_predict, fit_threshold, encode, shuffle_events, StreamNet, fit


class Checks(unittest.TestCase):
    def test_cca_and_encoding(self):
        t=np.arange(250)/250
        frequencies=np.linspace(8,15.8,40)
        signals=np.stack([np.sin(2*np.pi*frequencies[7]*t + phase) for phase in np.arange(6)/3])
        self.assertEqual(int(cca_predict(signals[None],frequencies)[0]),7)
        rng=np.random.default_rng(3)
        train=rng.normal(size=(8,6,250)).astype('float32')
        threshold=fit_threshold(train)
        x=encode(train,threshold)
        shuffled=shuffle_events(x,42)
        self.assertEqual(x.shape,(8,250,12))
        self.assertTrue(np.array_equal(x.sum(1),shuffled.sum(1)))
        self.assertTrue(np.all(shuffled.reshape(8,250,6,2).sum(-1)<=1))

    def test_bd_reference_forward_reset_and_training(self):
        torch.manual_seed(3)
        model=StreamNet('bd')
        x=torch.randint(0,2,(2,3,12)).float()
        with torch.no_grad():
            mem=torch.zeros(2,16); spikes=torch.zeros_like(mem)
            for i in range(3):
                previous=mem
                current=.25*previous+model.input(x[:,i])
                change=torch.zeros_like(current) if i==0 else (current-previous).abs()/.1
                threshold=.5-model.alpha*torch.tanh(change-1.5)
                spike=(current>threshold).float()-(current < -threshold).float()
                mem=current*(1-spike)
                spikes=spikes+spike
            expected=model.output(mem+spikes/3)
            self.assertTrue(torch.allclose(expected,model(x)))
            self.assertTrue(torch.equal(model(x),model(x)))
        z=np.zeros((8,3,12),dtype='float32'); z[:,1,0]=1
        labels=np.arange(8,dtype='int64')
        _,info=fit(z,labels,z,labels,'bd',2,1,1,torch.device('cpu'))
        self.assertEqual(info['epochs_run'],1)


if __name__=='__main__': unittest.main()
