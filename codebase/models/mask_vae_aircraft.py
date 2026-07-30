"""
CausalVAE for the Aircraft Damage dataset.

Neural Structural Causal Model (Yang et al. 2021, "CausalVAE: Disentangled
Representation Learning via Neural Structural Causal Models") adapted for
64×64 RGB aircraft-damage images and a learnable concept DAG.

Concept space is factored as z ∈ R^{z1_dim × z2_dim}:
    z1_dim — number of causal concepts (DAG nodes)
    z2_dim — features per concept
    z_dim  = z1_dim * z2_dim

Forward flow (matches the call sites in run/evaluate/inference):
    feat, skips = enc.encode(x)                  # U-Net encoder w/ skip channels
    q_m, q_v    = gaussian_parameters(feat)      # amortised posterior
    q_m         = enc_proj(q_m)                  # project into concept space
    decode_m,_  = dag.calculate_dag(q_m, q_v)    # (I - Aᵀ)⁻¹  structural propagation
    m_zm        = dag.mask_z(decode_m)           # aggregate causal parents (Aᵀ z)
    f_z         = mask_z.mix(m_zm)               # per-concept nonlinear SCM g(·)
    e_tilde     = attn.attention(decode_m, q_m)  # exogenous attention residual
    f_z1        = f_z + e_tilde                  # final concept code
    recon       = dec.decode(z, skips)           # U-Net decoder

The module also exposes ``compute_skip_gradients`` used by the inference
script for skip-connection counterfactuals.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from codebase import utils as ut
from codebase.utils import (
    gaussian_parameters,
    sample_gaussian,
    kl_normal,
    condition_prior_adaptive,
)


# ── U-Net encoder / decoder ──────────────────────────────────────────────────

class Encoder(nn.Module):
    """128×128 RGB → (2·z_dim) latent parameters, plus skip features.

    The 128-px input flows through four stride-2 blocks to an **8×8 bottleneck**
    (rather than 4×4), so four times as many spatial cells feed the latent and
    the skips are held at higher resolution — the point of moving to 128 px.
    Channel counts per skip level are unchanged, so the skip-attribution
    machinery (corr_idx) still applies.
    """

    def __init__(self, z_dim: int):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 4, 2, 1),
                nn.BatchNorm2d(co),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.c1 = block(3, 32)     # 128 → 64  (s1)
        self.c2 = block(32, 64)    # 64 → 32   (s2)
        self.c3 = block(64, 128)   # 32 → 16   (s3)
        self.c4 = block(128, 256)  # 16 →  8   (s4)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )                          #  8 →  8   (b)  — Grad-CAM target
        self.to_latent = nn.Conv2d(256, 2 * z_dim, 8, 1, 0)  # 8 → 1

    def encode(self, x):
        s1 = self.c1(x)
        s2 = self.c2(s1)
        s3 = self.c3(s2)
        s4 = self.c4(s3)
        b  = self.bottleneck(s4)
        feat = self.to_latent(b)              # (B, 2·z_dim, 1, 1)
        return feat, (s1, s2, s3, s4, b)


class Decoder(nn.Module):
    """(z_dim, 1, 1) latent + encoder skips → 128×128 RGB reconstruction."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.from_latent = nn.ConvTranspose2d(z_dim, 256, 8, 1, 0)  # 1 → 8

        def merge(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, 1, 1),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
            )

        self.m_b  = merge(512, 256)   # cat(from_latent, b)   8×8
        self.m_s4 = merge(512, 256)   # cat(·, s4)            8×8
        self.u3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 8 → 16
        self.m3 = merge(256, 128)     # cat(·, s3)
        self.u2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 16 → 32
        self.m2 = merge(128, 64)      # cat(·, s2)
        self.u1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    # 32 → 64
        self.m1 = merge(64, 32)       # cat(·, s1)
        self.u0 = nn.ConvTranspose2d(32, 32, 4, 2, 1)    # 64 → 128
        self.out = nn.Conv2d(32, 3, 3, 1, 1)

    def decode(self, z_4d, skips):
        s1, s2, s3, s4, b = skips
        d = F.relu(self.from_latent(z_4d))       # (256, 8, 8)
        d = self.m_b(torch.cat([d, b], dim=1))
        d = self.m_s4(torch.cat([d, s4], dim=1))
        d = self.u3(d)
        d = self.m3(torch.cat([d, s3], dim=1))
        d = self.u2(d)
        d = self.m2(torch.cat([d, s2], dim=1))
        d = self.u1(d)
        d = self.m1(torch.cat([d, s1], dim=1))
        d = self.u0(d)
        return self.out(d)                       # (3, 128, 128)


# ── Structural causal layers ─────────────────────────────────────────────────

class DagLayer(nn.Module):
    """Learnable concept adjacency A (A[i, j] = 1 means concept i → j)."""

    def __init__(self, z1_dim: int, z2_dim: int):
        super().__init__()
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.A = nn.Parameter(torch.zeros(z1_dim, z1_dim))

    def _inv(self):
        I = torch.eye(self.z1_dim, device=self.A.device, dtype=self.A.dtype)
        return torch.inverse(I - self.A.t())

    def calculate_dag(self, z, v):
        """Structural propagation z ← (I - Aᵀ)⁻¹ ε over the concept dimension."""
        z = z.reshape(-1, self.z1_dim, self.z2_dim)
        v = v.reshape(-1, self.z1_dim, self.z2_dim)
        inv = self._inv()
        decode_m = torch.einsum('ij,bjk->bik', inv, z)
        decode_v = torch.einsum('ij,bjk->bik', inv, v).abs() + 1e-4
        return decode_m, decode_v

    def mask_z(self, z):
        """Aggregate each node's causal parents:  node_j = Σ_i A[i, j] · z_i."""
        z = z.reshape(-1, self.z1_dim, self.z2_dim)
        return torch.einsum('ij,bjk->bik', self.A.t(), z)

    def mask_u(self, u):
        """Same parent-aggregation applied to a label/concept vector."""
        if u.dim() == 2:
            u = u.unsqueeze(-1).expand(-1, -1, self.z2_dim)
        u = u.reshape(-1, self.z1_dim, self.z2_dim)
        return torch.einsum('ij,bjk->bik', self.A.t(), u)


class MaskLayer(nn.Module):
    """Per-concept nonlinear structural equation g_i : R^{z2} → R^{z2}."""

    def __init__(self, z1_dim: int, z2_dim: int, hidden: int = 32):
        super().__init__()
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(z2_dim, hidden),
                nn.ELU(inplace=True),
                nn.Linear(hidden, z2_dim),
            )
            for _ in range(z1_dim)
        ])

    def mix(self, z):
        z = z.reshape(-1, self.z1_dim, self.z2_dim)
        outs = [self.nets[i](z[:, i, :]) for i in range(self.z1_dim)]
        return torch.stack(outs, dim=1)


class Attention(nn.Module):
    """Bilinear attention producing the exogenous residual ẽ."""

    def __init__(self, z2_dim: int):
        super().__init__()
        self.M = nn.Parameter(torch.randn(z2_dim, z2_dim) * 0.1)

    def attention(self, z, e):
        zM    = torch.einsum('bik,kl->bil', z, self.M)   # (B, z1, z2)
        score = torch.einsum('bil,bjl->bij', zM, e)      # (B, z1, z1)
        attn  = torch.softmax(score, dim=-1)
        e_out = torch.einsum('bij,bjk->bik', attn, e)    # (B, z1, z2)
        return e_out, attn


# ── CausalVAE ────────────────────────────────────────────────────────────────

class CausalVAE(nn.Module):
    def __init__(self, name: str = 'aircraft_causalvae',
                 z_dim: int = 20, z1_dim: int = 5, z2_dim: int = 4,
                 inference: bool = False, scale=None, initial: bool = True):
        super().__init__()
        assert z_dim == z1_dim * z2_dim, \
            f'z_dim ({z_dim}) must equal z1_dim*z2_dim ({z1_dim*z2_dim})'
        self.name = name
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.inference = inference
        if scale is None:
            scale = np.array([[0.5, 0.5]] * z1_dim, dtype=float)
        scale = np.asarray(scale)
        assert scale.shape == (z1_dim, 2), \
            f'scale must have shape ({z1_dim}, 2), got {tuple(scale.shape)}'
        self.scale = scale

        self.enc      = Encoder(z_dim)
        self.enc_proj = nn.Linear(z_dim, z_dim)
        self.dec      = Decoder(z_dim)
        self.dag      = DagLayer(z1_dim, z2_dim)
        self.attn     = Attention(z2_dim)
        self.mask_z   = MaskLayer(z1_dim, z2_dim)
        self.mask_u   = MaskLayer(z1_dim, z2_dim)

    # -- helpers --------------------------------------------------------------
    def _normalize_label(self, label):
        """Normalise concept labels the same way condition_prior does."""
        scale = torch.as_tensor(self.scale, dtype=label.dtype, device=label.device)
        z1 = self.z1_dim
        mean = scale[:z1, 0].unsqueeze(0)
        half = scale[:z1, 1].unsqueeze(0)
        return (label[:, :z1] - mean) / half

    # -- objective ------------------------------------------------------------
    def negative_elbo_bound(self, x, label, sample: bool = False):
        B = x.size(0)
        z1, z2 = self.z1_dim, self.z2_dim

        feat, skips = self.enc.encode(x)
        q_m_full, q_v_full = gaussian_parameters(feat, dim=1)   # (B, z_dim, 1, 1)
        q_m = self.enc_proj(q_m_full.reshape(B, -1)).reshape(B, z1, z2)
        q_v = q_v_full.reshape(B, z1, z2) + 1e-4

        decode_m, decode_v = self.dag.calculate_dag(q_m, q_v)
        decode_m = decode_m.reshape(B, z1, z2)

        m_zm    = self.dag.mask_z(decode_m).reshape(B, z1, z2)
        f_z     = self.mask_z.mix(m_zm).reshape(B, z1, z2)
        e_tilde = self.attn.attention(decode_m, q_m)[0]
        f_z1    = f_z + e_tilde

        if sample or not self.inference:
            z_dag = sample_gaussian(f_z1, decode_v)
        else:
            z_dag = f_z1

        # KL against label-conditioned structural prior (uncertainty-aware
        # variance from condition_prior_adaptive; pv varies per concept and
        # label value and is used directly, never overridden with ones).
        cp_m, cp_v = ut.condition_prior_adaptive(
            self.scale, label, self.z2_dim, tightness=1.5)
        pm, pv = cp_m.to(x.device), cp_v.to(x.device)
        kl = kl_normal(
            f_z1.reshape(B, -1), decode_v.reshape(B, -1),
            pm.reshape(B, -1), pv.reshape(B, -1),
        ).mean()

        # Reconstruction (images live in ImageNet-normalised space)
        z_4d  = z_dag.reshape(B, self.z_dim, 1, 1)
        recon = self.dec.decode(z_4d, skips)
        rec   = F.mse_loss(recon, x, reduction='none').reshape(B, -1).sum(-1).mean()

        # Structural / label consistency term  (nelbo - rec - kl == mask_l)
        u_target = self._normalize_label(label)
        u_recon  = self.mask_u.mix(f_z1).mean(-1)
        mask_l   = F.mse_loss(f_z1, decode_m) + F.mse_loss(u_recon, u_target)

        nelbo = rec + kl + mask_l
        return nelbo, kl, rec, recon, z_dag


# ── Skip-channel gradient attribution (for inference counterfactuals) ────────

def compute_skip_gradients(model, clf, dataloader, device,
                           n_observable=3, top_k=24, n_batches=50):
    """
    Identifies encoder skip channels most influential to each observable
    concept's classifier logit, using gradient-based attribution.

    Gradient attribution captures nonlinear encoder-concept relationships
    that Pearson correlation misses, giving more precise channel targeting
    for skip-channel suppression during counterfactual generation.

    For each observable concept k, backpropagates the sum of logits[:, k]
    through the classifier and latent path to the skip tensors, accumulates
    absolute gradient magnitude per channel across n_batches, then returns
    the top_k channel indices per concept per encoder level.

    Args:
        model:        CausalVAE in eval mode
        clf:          ConceptAwareClassifier
        dataloader:   validation DataLoader returning (imgs, labels)
        device:       torch device
        n_observable: number of observable concepts
        top_k:        channels to select per concept per level
        n_batches:    batches to accumulate over (50 is sufficient)

    Selection is *contrastive*: the encoder skips reach every classifier logit
    only through the single shared encoder trunk, so raw |grad| is dominated by
    the same high-activation channels for every concept (crack, dent and
    no_damage would otherwise select an almost identical channel set, making
    their counterfactual images indistinguishable). Instead each concept's
    attribution is normalised and the *other* concepts' attribution is
    subtracted, so a channel is chosen for concept k only if it drives k more
    than it drives the rest. This yields concept-specific channel sets and
    therefore visibly different counterfactuals for "remove crack" vs
    "remove dent".

    Returns:
        corr_idx: dict mapping level name to tensor (n_observable, top_k)
                  level names are 's1', 's2', 's3', 's4', 'b'
    """
    model.eval()
    clf.eval()
    level_names = ['s1', 's2', 's3', 's4', 'b']
    grad_acc    = {name: [None] * n_observable for name in level_names}
    count       = 0

    for imgs, labels, *_ in dataloader:
        if count >= n_batches:
            break
        imgs = imgs.to(device)

        feat, skips = model.enc.encode(imgs)
        skip_list   = list(skips)
        for s in skip_list:
            # Skip tensors are non-leaf (they descend from encoder params), so
            # requires_grad_ alone does not populate .grad — retain_grad() does.
            s.requires_grad_(True)
            s.retain_grad()

        q_m_full, _ = ut.gaussian_parameters(feat, dim=1)
        q_m  = model.enc_proj(q_m_full.view(imgs.size(0), -1))
        q_m  = q_m.reshape([imgs.size(0), model.z1_dim, model.z2_dim])
        dm, _ = model.dag.calculate_dag(q_m, torch.ones_like(q_m))
        dm   = dm.reshape([imgs.size(0), model.z1_dim, model.z2_dim])
        m_zm = model.dag.mask_z(dm).reshape_as(dm)
        f_z  = model.mask_z.mix(m_zm).reshape_as(dm)
        e_t  = model.attn.attention(dm, q_m)[0]
        f_z1 = f_z + e_t

        logits = clf(f_z1)

        for k in range(n_observable):
            model.zero_grad()
            clf.zero_grad()
            retain = (k < n_observable - 1)
            logits[:, k].sum().backward(retain_graph=retain)

            for name, s in zip(level_names, skip_list):
                if s.grad is None:
                    continue
                ch_grad = s.grad.abs().mean(dim=(0, 2, 3))
                if grad_acc[name][k] is None:
                    grad_acc[name][k] = ch_grad.detach().cpu()
                else:
                    grad_acc[name][k] += ch_grad.detach().cpu()

        count += 1

    corr_idx = {}
    for name in level_names:
        # Pad any concept that never received a gradient to the common width.
        width = max((g.numel() for g in grad_acc[name] if g is not None),
                    default=1)
        accumulated = [
            g if g is not None else torch.zeros(width)
            for g in grad_acc[name]
        ]
        stacked  = torch.stack(accumulated).float()          # (n_obs, C), >= 0
        k_actual = min(top_k, stacked.size(1))

        # Normalise each concept's attribution so concepts are comparable, then
        # subtract the mean attribution of the *other* concepts. Channels that
        # light up for every concept (shared trunk detail) cancel out; only the
        # concept-specific channels survive, so crack and dent select different
        # channels and their counterfactuals differ visibly.
        norm   = stacked / (stacked.sum(dim=1, keepdim=True) + 1e-8)
        n_obs  = norm.size(0)
        if n_obs > 1:
            others   = (norm.sum(dim=0, keepdim=True) - norm) / (n_obs - 1)
            contrast = norm - others
        else:
            contrast = norm
        corr_idx[name] = contrast.topk(k_actual, dim=1).indices

    return corr_idx
