import unittest
import numpy as np
import torch
from gate0_temporal_diagnostic import make_data, TemporalLIF


class TestGate0(unittest.TestCase):
    def test_balanced_counts_and_disjoint_splits(self):
        for task in ('timing', 'polarity'):
            splits = make_data(task)
            seen = set()
            for x, y in splits:
                self.assertEqual(int((y == 0).sum()), int((y == 1).sum()))
                self.assertTrue(torch.equal(x.sum(1), x.sum(1)[:1].expand(len(x), -1)))
                for row in x.numpy():
                    key = row.tobytes()
                    self.assertNotIn(key, seen)
                    seen.add(key)
                early = x[:, :24].sum(1)
                if task == 'timing':
                    self.assertTrue(torch.equal((early[:, 2] > early[:, 0]).long(), y))
                else:
                    self.assertTrue(torch.equal((early[:, 1] > early[:, 0]).long(), y))

    def test_reset_order_and_gradients(self):
        torch.manual_seed(10)
        model = TemporalLIF()
        x = torch.randint(0, 2, (3, 48, 4)).float()
        output = model(x)
        self.assertTrue(torch.allclose(output[0], model(x[:1])[0], atol=1e-6))
        model(x[1:]); self.assertTrue(torch.equal(output, model(x)))
        output.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()), 0)


if __name__ == '__main__':
    unittest.main()
