"""Deterministic synthetic checks; no research data are needed."""
import unittest
import torch
from eeg_tcnet_stca import AdaptiveSignedLIF, TemporalChannelAttention, EEGTCNetAttention
from lif_eegnet import SignedLIFActivation

class Checks(unittest.TestCase):
    def test_attention(self):
        torch.manual_seed(1)
        x = torch.randn(2, 12, 15, requires_grad=True)
        y, w = TemporalChannelAttention(12)(x, return_weights=True)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(w.sum(-1), torch.ones(2, 15), atol=1e-6))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_bipolar_and_gradient(self):
        m = AdaptiveSignedLIF(threshold=.5, adapt_gain=.1)
        x = torch.tensor([-1., 0., 1.], requires_grad=True)
        y = m(x)
        self.assertLess(y[0].item(), 0); self.assertGreater(y[2].item(), 0)
        self.assertGreater(m.last_threshold, .5)
        self.assertGreater(m.last_firing_rate, 0); self.assertLess(m.last_firing_rate, 1)
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_model(self):
        for attention, adaptive in [(False, False), (True, False), (True, True)]:
            model = EEGTCNetAttention(attention=attention, adaptive=adaptive)
            y = model(torch.randn(2, 1, 22, 500))
            self.assertEqual(y.shape, (2, 4))
            self.assertTrue(torch.isfinite(y).all())
            self.assertGreater(sum(p.numel() for p in model.parameters()), 0)

    def test_fixed_signed_matches_zero_adaptation(self):
        x = torch.linspace(-2, 2, 31)
        fixed = SignedLIFActivation(threshold=.5)(x)
        adaptive = AdaptiveSignedLIF(threshold=.5, adapt_gain=0)(x)
        self.assertTrue(torch.equal(fixed, adaptive))
        self.assertLess(fixed[0].item(), 0)
        self.assertGreater(fixed[-1].item(), 0)
        for kwargs in ({'signed': True}, {'adaptive': True}):
            self.assertEqual(EEGTCNetAttention(**kwargs)(torch.randn(2, 1, 22, 500)).shape, (2, 4))

if __name__ == '__main__':
    unittest.main()
