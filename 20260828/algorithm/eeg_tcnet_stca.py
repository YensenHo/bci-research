"""Exploratory temporal-channel attention and adaptive signed LIF for EEG-TCNet.

Attention operates on EEG feature time, not the internal LIF simulation axis;
these are inspired by STCA/BD-LIF and are not reproductions of those papers.
"""
import math
import torch
from torch import nn
from eeg_tcnet_lif import EEGTCNetLIF
from lif_eegnet import LIFActivation, SurrogateStep


class TemporalChannelAttention(nn.Module):
    """Dense offline attention on (batch, feature channels, EEG feature time)."""
    def __init__(self, channels, hidden=8):
        super().__init__()
        self.q = nn.Linear(channels, hidden, bias=False)
        self.k = nn.Linear(channels, hidden, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.channel_gate = nn.Linear(channels, channels)
        self.scale = hidden ** -0.5

    def forward(self, x, return_weights=False):
        u = x.transpose(1, 2)
        weights = torch.softmax((self.q(u) @ self.k(u).transpose(-1, -2)) * self.scale, dim=-1)
        gate = torch.sigmoid(self.channel_gate(u.mean(dim=1))).unsqueeze(1)
        y = u + (weights @ self.v(u)) * gate
        y = y.transpose(1, 2)
        return (y, weights) if return_weights else y


class AdaptiveSignedLIF(nn.Module):
    """Per-forward bipolar rate output; adaptation resets for every input batch."""
    def __init__(self, threshold=1.0, tau=2.0, timesteps=8, alpha=4.0,
                 adapt_decay=0.9, adapt_gain=0.1):
        super().__init__()
        if threshold <= 0 or tau <= 0 or timesteps < 1:
            raise ValueError('invalid LIF parameters')
        self.threshold, self.decay = float(threshold), math.exp(-1.0 / tau)
        self.timesteps, self.alpha = int(timesteps), float(alpha)
        self.adapt_decay, self.adapt_gain = float(adapt_decay), float(adapt_gain)
        self.last_firing_rate = 0.0

    def forward(self, x):
        mp = torch.zeros_like(x); mn = torch.zeros_like(x)
        ap = torch.zeros_like(x); an = torch.zeros_like(x)
        pos = torch.zeros_like(x); neg = torch.zeros_like(x)
        for _ in range(self.timesteps):
            mp = self.decay * mp + torch.relu(x)
            mn = self.decay * mn + torch.relu(-x)
            sp = SurrogateStep.apply(mp - (self.threshold + ap), self.alpha)
            sn = SurrogateStep.apply(mn - (self.threshold + an), self.alpha)
            mp = mp * (1 - sp); mn = mn * (1 - sn)
            ap = self.adapt_decay * ap + self.adapt_gain * sp
            an = self.adapt_decay * an + self.adapt_gain * sn
            pos = pos + sp; neg = neg + sn
        self.last_firing_rate = float(((pos + neg) / self.timesteps).mean().detach())
        self.last_threshold = float((self.threshold + (ap + an).mean() / 2).detach())
        return (pos - neg) / self.timesteps


class EEGTCNetAttention(EEGTCNetLIF):
    def __init__(self, attention=False, adaptive=False, **kwargs):
        kwargs.setdefault('activation', 'lif')
        super().__init__(**kwargs)
        if attention:
            self.tcn.add_module('attention', TemporalChannelAttention(self.FT))
        if adaptive:
            hp = {k: kwargs[k] for k in ('threshold', 'tau', 'timesteps', 'alpha') if k in kwargs}
            def replace(module):
                for name, child in module.named_children():
                    if isinstance(child, LIFActivation):
                        setattr(module, name, AdaptiveSignedLIF(**hp))
                    else:
                        replace(child)
            replace(self)

    @property
    def tcn_dilations(self):
        return [block.dilation for block in self.tcn if hasattr(block, 'dilation')]

    def forward_features(self, x):
        result = super().forward_features(x)
        for name, module in self.named_modules():
            if isinstance(module, AdaptiveSignedLIF):
                self.firing_rates[name] = module.last_firing_rate
        return result
