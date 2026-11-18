import unittest
import numpy as np
import torch
from gate1_ssvep import StreamNet, encode, fit_threshold, shuffle_events
from diagnose_gate1_information import normalize_trials, ridge
from diagnose_gate1_lif import DiagnosticLIF, run_model


class Checks(unittest.TestCase):
    def test_representation_and_ridge(self):
        rng=np.random.default_rng(7)
        train=rng.normal(size=(8,6,250)).astype('float32')
        threshold=fit_threshold(train)
        event=encode(train,threshold)
        shuffled=shuffle_events(event,42)
        self.assertTrue(np.array_equal(event.sum(1),shuffled.sum(1)))
        self.assertTrue(np.isfinite(normalize_trials(train)).all())
        x=np.eye(4).repeat(2,axis=0);y=np.arange(4).repeat(2)
        result=ridge(x,y,x,y,x,4)
        self.assertEqual(result['test_predictions'].tolist(),y.tolist())

    def test_rate_matches_original_and_temporal_reset(self):
        torch.manual_seed(9); old=StreamNet('lif')
        torch.manual_seed(9); new=DiagnosticLIF('rate')
        x=torch.randint(0,2,(2,250,12)).float()
        self.assertTrue(torch.allclose(old(x),new(x),atol=1e-6))
        temporal=DiagnosticLIF('temporal')
        output=temporal(x); temporal(x[:1])
        self.assertTrue(torch.equal(output,temporal(x)))
        output.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in temporal.parameters()))


if __name__=='__main__': unittest.main()
