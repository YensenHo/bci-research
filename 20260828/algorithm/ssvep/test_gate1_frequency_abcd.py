import unittest
import numpy as np
import torch
from gate1_frequency_abcd import synthetic_data, select_frequencies, build_models, prepare
from diagnose_gate1_lif import DiagnosticLIF


class Checks(unittest.TestCase):
    def test_synthetic_balance_unique_and_counts(self):
        frequencies,splits=synthetic_data(42)
        self.assertEqual(frequencies.tolist(),[8.,10.,12.,14.])
        seen=set()
        for x,y in splits:
            self.assertTrue(np.all(np.bincount(y,minlength=4)==len(y)//4))
            for row in x:
                key=row.tobytes();self.assertNotIn(key,seen);seen.add(key)
        events,threshold=prepare(splits)
        self.assertEqual(events[0][0].shape,(512,250,12))
        self.assertTrue(np.all(np.isfinite(threshold)))

    def test_mapping_parameters_state_and_gradients(self):
        freq=np.linspace(8.,15.8,40)[::-1]
        ix,selected=select_frequencies(freq,'4')
        self.assertTrue(np.allclose(selected,[8,10,12,14]))
        self.assertTrue(np.allclose(freq[ix],selected))
        models=build_models([8.,10.,12.,14.],42)
        count=lambda m:sum(p.numel() for p in m.parameters())
        self.assertEqual(count(models['C']),count(models['D']))
        self.assertGreater(count(models['B']),count(models['A']))
        torch.manual_seed(42);old=DiagnosticLIF('rate')
        torch.manual_seed(42);new=build_models(np.linspace(8.,15.8,40),42)['A']
        x=torch.randint(0,2,(2,250,12)).float()
        self.assertTrue(torch.allclose(old(x),new(x),atol=1e-6))
        model=models['C']; output=model(x);model(x[:1]);self.assertTrue(torch.equal(output,model(x)))
        output.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=='__main__':unittest.main()
