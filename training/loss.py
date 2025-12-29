import torch
import numpy as np
import scipy.stats as st
from torch_utils import persistence
import torch.nn.functional as F
import torch.fft as fft
import math
from torch import nn



@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y= augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, )
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------

@torch.jit.script  
def is_all_zero(x: torch.Tensor) -> bool:

    return torch.allclose(x, torch.zeros_like(x))

@torch.jit.script
def create_zero_complex_tensor_like(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros(x.shape, dtype=torch.complex64, device=x.device)




@persistence.persistent_class
class UnifiedLoss(nn.Module):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,
                 weight_base=0.8, weight_phase=0.12, weight_amplitude=0.08,
                 use_cpu_fft=True):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.weight_base = weight_base
        self.weight_phase = weight_phase
        self.weight_amplitude = weight_amplitude
        self.use_cpu_fft = use_cpu_fft

    def safe_fft2(self, x):
        if is_all_zero(x):
            return create_zero_complex_tensor_like(x)
        if self.use_cpu_fft:
            return fft.fft2(x.detach().cpu(), norm='ortho').to(x.device)
        else:
            return fft.fft2(x, norm='ortho')

    def compute_amplitude_phase(self, x):
        x = x - x.mean(dim=(-2, -1), keepdim=True)
        x_fft = self.safe_fft2(x)
        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        return amplitude, phase

    def forward(self, net, images, lr_images, ref_images, labels=None, augment_pipe=None,
                current_step=0, total_steps=10000):
        

        progress = current_step / total_steps
        if progress < 0.5:
            w_phase = 0.0
            w_amplitude = 0.0
            w_base = 1.0
        else:
            factor = (progress - 0.5) / 0.5  
            w_phase = self.weight_phase * factor
            w_amplitude = self.weight_amplitude * factor
            w_base = self.weight_phase + self.weight_amplitude + self.weight_base - w_phase - w_amplitude



        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, lr_images, ref_images, labels, augment_labels=augment_labels)
        loss_denoise = weight * ((D_yn - y) ** 2)


        sr_amp, sr_phase = self.compute_amplitude_phase(D_yn)
        _, ref_phase = self.compute_amplitude_phase(ref_images)
        gt_amp, _ = self.compute_amplitude_phase(images)

        sr_sin = torch.sin(sr_phase)
        sr_cos = torch.cos(sr_phase)
        ref_sin = torch.sin(ref_phase)
        ref_cos = torch.cos(ref_phase)
        cos_sim = sr_cos * ref_cos + sr_sin * ref_sin
        loss_phase = torch.mean(1 - cos_sim)

        sr_amp = sr_amp / (sr_amp.mean(dim=(-2,-1), keepdim=True) + 1e-8)
        gt_amp = gt_amp / (gt_amp.mean(dim=(-2,-1), keepdim=True) + 1e-8)
        loss_amplitude = F.l1_loss(sr_amp, gt_amp)


        total_loss = (
            w_base * loss_denoise +
            w_phase * loss_phase +
            w_amplitude * loss_amplitude
        )
        return total_loss
