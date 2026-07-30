"""
Training script for CausalVAE on the Aircraft Damage dataset.

Three mutually-exclusive observable classes (structural_crack / dent /
no_damage) predicted with single-label softmax.

Run from repo root:
    python run_aircraft.py
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from codebase import utils as ut
from codebase.models.mask_vae_aircraft import CausalVAE
from dataset.aircraft_damage import (
    get_dataloader, CLASS_NAMES, N_OBSERVABLE, SCALE
)
from dataset.aircraft_dag import A_INIT
from models.classifier_head import (
    HierarchicalConceptClassifier, ConceptHeads,
    single_label_loss, compute_metrics,
)
from plot_style import apply_journal_style
from utils import _h_A

apply_journal_style()

# ── Config ────────────────────────────────────────────────────────────────────
Z1_DIM     = 5                    # concepts: 3 observable + 2 latent root causes
Z2_DIM     = 4                    # features per concept
Z_DIM      = Z1_DIM * Z2_DIM     # = 20
EPOCHS     = 100
BATCH_SIZE = 16     # 128-px inputs + 8×8 bottleneck: small batch fits 6 GB
LR         = 1e-4
CLF_WEIGHT = 1.0    # concept labels keep DAG structure meaningful
TC_WEIGHT  = 2.0    # β-TCVAE grouped total correlation penalty
BIN_WEIGHT = 1.0    # weight on the damaged-vs-healthy binary loss
DAMAGE_POS_WEIGHT = 3.0  # cost of MISSING damage vs a false alarm (errs safe)
DATA_ROOT  = './dataset'
SAVE_DIR   = './checkpoints'
FIG_DIR    = './figs_aircraft'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ── Data ──────────────────────────────────────────────────────────────────────
train_loader = get_dataloader(DATA_ROOT, 'train', BATCH_SIZE, num_workers=0)
val_loader   = get_dataloader(DATA_ROOT, 'valid', BATCH_SIZE, num_workers=0)

# Class balance for the single-label cross-entropy (mutually-exclusive classes)
print('Computing class statistics…')
class_counts = np.zeros(N_OBSERVABLE, dtype=float)
n_total      = 0
for _, labels in train_loader:
    class_counts += labels[:, :N_OBSERVABLE].sum(0).numpy()
    n_total      += labels.size(0)
# Inverse-frequency class weights, normalised to mean 1.
class_weights = n_total / (N_OBSERVABLE * np.maximum(class_counts, 1.0))
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f'  class_weights: {class_weights.cpu().numpy().round(2)}')
print(f'  class_rates:   {(class_counts/n_total).round(3)}')

# ── Model ─────────────────────────────────────────────────────────────────────
lvae = CausalVAE(
    name='aircraft_causalvae',
    z_dim=Z_DIM,
    z1_dim=Z1_DIM,
    z2_dim=Z2_DIM,
    inference=False,
    scale=SCALE,
    initial=False,
).to(device)

with torch.no_grad():
    lvae.dag.A.copy_(A_INIT.to(device))

clf           = HierarchicalConceptClassifier(
    z2_dim=Z2_DIM, n_observable=N_OBSERVABLE, hidden=64).to(device)
concept_heads = ConceptHeads(z2_dim=Z2_DIM, n_observable=N_OBSERVABLE).to(device)

ce_weight = class_weights   # single-label cross-entropy class weights

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# ── Active configuration ──────────────────────────────────────────────────────
_bar = '═' * 46
_names_line1 = ', '.join(CLASS_NAMES[:3]) + ','
_names_line2 = ', '.join(CLASS_NAMES[3:])
print(_bar)
print('CausalADA Training Configuration')
print(_bar)
print(f'Z1_DIM       : {Z1_DIM}')
print(f'Z_DIM        : {Z_DIM}')
print(f'N_OBSERVABLE : {N_OBSERVABLE}')
print(f'CLASS_NAMES  : {_names_line1}')
print(f'               {_names_line2}')
print('DAG A_INIT   :')
print(A_INIT.cpu().numpy())
print(_bar)

# ── Optimiser ─────────────────────────────────────────────────────────────────
enc_params   = list(lvae.enc.parameters()) + list(lvae.enc_proj.parameters())
dec_params   = list(lvae.dec.parameters())
other_params = (
    list(lvae.dag.parameters())    +
    list(lvae.attn.parameters())   +
    list(lvae.mask_z.parameters()) +
    list(lvae.mask_u.parameters())
)

optimizer = torch.optim.Adam([
    {'params': enc_params,                   'lr': LR},
    {'params': dec_params,                   'lr': 5e-4},
    {'params': other_params,                 'lr': LR},
    {'params': clf.parameters(),             'lr': LR},
    {'params': concept_heads.parameters(),   'lr': LR},
], betas=(0.9, 0.999))

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
)

# ── KL annealing ──────────────────────────────────────────────────────────────
def kl_weight(epoch: int) -> float:
    return min(1.0, epoch / 50.0) * 0.12

# ── Helpers ───────────────────────────────────────────────────────────────────
def minibatch_tc(z):
    """
    Minibatch estimate of total correlation for a group of concept
    sub-vectors. Returns zero if the group contains fewer than 2 concepts.
    """
    B, z1, _ = z.shape
    if z1 < 2:
        return torch.zeros(1, device=z.device)
    z_n           = z.unsqueeze(1)
    mu_m          = z.unsqueeze(0)
    log_q_given_x = -0.5 * ((z_n - mu_m) ** 2).sum(-1)
    log_q_z       = torch.logsumexp(log_q_given_x.sum(-1), dim=1) \
                    - math.log(B)
    log_prod_q    = (torch.logsumexp(log_q_given_x, dim=1) \
                    - math.log(B)).sum(-1)
    return (log_q_z - log_prod_q).mean()


def minibatch_tc_grouped(z, n_observable=3):
    """
    Computes total correlation separately within the observable concept
    block and within the latent concept block rather than across all
    concepts jointly.

    The DAG structural equation propagates parent concept activations
    into child sub-vectors by design. Applying a joint TC penalty across
    all concepts penalises this intended dependence and works against
    the causal structure the DAG is learning. Applying TC within each
    group separately encourages diversity within groups while leaving
    inter-group dependence — imposed by the DAG — unconstrained.

    The latent group TC is weighted at 1.0 versus 0.5 for the observable
    group to encourage impact_force and metal_fatigue to develop distinct
    representations rather than converging toward each other.
    """
    tc_obs = minibatch_tc(z[:, :n_observable, :])
    tc_lat = minibatch_tc(z[:, n_observable:, :])
    return 0.5 * tc_obs + 1.0 * tc_lat


def denorm(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(t.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(t.device)
    return torch.clamp(t * std + mean, 0, 1)


def save_recon_samples(imgs, recons, epoch, n=8):
    imgs   = imgs[:n].detach().cpu()
    recons = recons[:n].detach().cpu()
    grid   = torch.cat([denorm(imgs), denorm(recons)], dim=0)
    save_image(grid, os.path.join(FIG_DIR, f'recon_epoch{epoch:04d}.png'), nrow=n)


def validate(model, clf_model, loader):
    model.eval(); clf_model.eval(); concept_heads.eval()
    all_logits, all_targets = [], []
    total_rec = total_kl = 0.0
    n_batches = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            _, kl, rec, _, z_dag = model.negative_elbo_bound(
                imgs, labels, sample=False
            )
            logits  = clf_model(z_dag)

            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())
            total_rec += rec.item()
            total_kl  += kl.item()
            n_batches += 1

    logits_all   = torch.cat(all_logits,    dim=0)
    targets_all  = torch.cat(all_targets,   dim=0)

    metrics = compute_metrics(
        logits_all, targets_all[:, :N_OBSERVABLE],
        class_names=CLASS_NAMES[:N_OBSERVABLE],
    )
    metrics['rec'] = total_rec / max(n_batches, 1)
    metrics['kl']  = total_kl  / max(n_batches, 1)
    return metrics


def save_checkpoint(epoch, val_f1, tag='best'):
    path = os.path.join(SAVE_DIR, f'aircraft_{tag}.pt')
    torch.save({
        'lvae':          lvae.state_dict(),
        'clf':           clf.state_dict(),
        'concept_heads': concept_heads.state_dict(),
        'epoch':         epoch,
        'val_f1':        val_f1,
        'config': {'z_dim': Z_DIM, 'z1_dim': Z1_DIM, 'z2_dim': Z2_DIM, 'n_observable': N_OBSERVABLE},
    }, path)
    print(f'  Saved checkpoint → {path}')


def plot_history(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('CausalVAE Training Dynamics — Aircraft Damage',
                 fontsize=18, fontweight='bold')

    specs = [
        (axes[0, 0], 'loss', 'Total Objective',                'Loss',     '#0072B2'),
        (axes[0, 1], 'rec',  'Reconstruction (MSE)',           'Loss',     '#009E73'),
        (axes[1, 0], 'clf',  'Classification (Cross-Entropy)', 'Loss',     '#B2362C'),
        (axes[1, 1], 'f1',   'Validation Macro F1',            'Macro F1', '#7B2D8E'),
    ]
    for ax, key, title, ylabel, color in specs:
        if key in history and history[key]:
            xs = range(1, len(history[key]) + 1)
            ax.plot(xs, history[key], color=color, linewidth=2.0)
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
        else:
            ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close(fig)


# ── Training loop ─────────────────────────────────────────────────────────────
history = {
    'loss': [], 'kl': [], 'rec': [], 'clf': [], 'f1': [],
}
best_val_f1 = -float('inf')

for epoch in range(EPOCHS):
    lvae.train(); clf.train(); concept_heads.train()

    kl_w       = kl_weight(epoch)
    total_loss = total_kl = total_rec = total_clf = total_tc = 0.0
    last_recon = last_imgs = None

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        nelbo, kl, rec, recon, z_dag = lvae.negative_elbo_bound(
            imgs, labels, sample=False
        )

        # DAG acyclicity penalty (NOTEARS)
        h_a       = _h_A(lvae.dag.A, lvae.dag.A.size(0))
        dag_penalty = 3 * h_a + 0.5 * h_a * h_a

        logits, bin_logit = clf(z_dag, return_binary=True)

        target_idx     = labels[:, :N_OBSERVABLE].argmax(1)
        clf_loss       = single_label_loss(logits, target_idx, weight=ce_weight)
        concept_logits = concept_heads(z_dag)
        concept_loss   = single_label_loss(concept_logits, target_idx, weight=ce_weight)
        tc_loss        = minibatch_tc_grouped(z_dag, n_observable=N_OBSERVABLE)

        # Damaged-vs-healthy binary loss; pos_weight makes missing damage costlier
        is_damaged = (target_idx != (N_OBSERVABLE - 1)).float()
        bin_loss   = F.binary_cross_entropy_with_logits(
            bin_logit.squeeze(1), is_damaged,
            pos_weight=torch.tensor(DAMAGE_POS_WEIGHT, device=device))

        mask_l = nelbo - rec - kl
        loss = (rec
                + kl_w * kl
                + 0.2 * mask_l
                + dag_penalty
                + CLF_WEIGHT * clf_loss
                + 0.5 * concept_loss
                + TC_WEIGHT * tc_loss
                + BIN_WEIGHT * bin_loss)

        loss.backward()
        nn.utils.clip_grad_norm_(
            list(lvae.parameters()) + list(clf.parameters()) +
            list(concept_heads.parameters()),
            max_norm=5.0,
        )
        optimizer.step()

        total_loss += loss.item()
        total_kl   += kl.item()
        total_rec  += rec.item()
        total_clf  += clf_loss.item()
        total_tc   += tc_loss.item()
        last_recon, last_imgs = recon, imgs

    n_batches = len(train_loader)
    avg_loss  = total_loss / n_batches
    avg_kl    = total_kl   / n_batches
    avg_rec   = total_rec  / n_batches
    avg_clf   = total_clf  / n_batches
    avg_tc    = total_tc   / n_batches

    val_metrics  = validate(lvae, clf, val_loader)
    val_f1       = val_metrics.get('f1_macro', 0.0)
    val_rec      = val_metrics.get('rec', float('inf'))
    scheduler.step(val_rec)

    history['loss'].append(avg_loss)
    history['kl'].append(avg_kl)
    history['rec'].append(avg_rec)
    history['clf'].append(avg_clf)
    history['f1'].append(val_f1)

    per_class_str = '  '.join(
        f'{n}={val_metrics.get(f"f1_{n}", 0.0):.3f}' for n in CLASS_NAMES[:N_OBSERVABLE]
    )
    cur_lr = optimizer.param_groups[0]['lr']
    print(
        f'[{epoch:03d}/{EPOCHS}] loss={avg_loss:.4f}  rec={avg_rec:.4f}  '
        f'clf={avg_clf:.4f}  tc={avg_tc:.3f}  '
        f'val_f1={val_f1:.3f}  kl_w={kl_w:.3f}  lr={cur_lr:.2e}\n'
        f'         per-class: {per_class_str}'
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        save_checkpoint(epoch, val_f1, tag='best')

    if epoch % 10 == 0:
        save_checkpoint(epoch, val_f1, tag=f'epoch{epoch:04d}')
        if last_recon is not None:
            save_recon_samples(last_imgs, last_recon, epoch)

    plot_history(history, os.path.join(FIG_DIR, 'training_metrics.png'))

# ── Final checkpoint ──────────────────────────────────────────────────────────
save_checkpoint(EPOCHS - 1, best_val_f1, tag='final')
print(f'\nTraining complete. Best val macro F1: {best_val_f1:.4f}')

# ── Precompute skip-channel gradient attribution ──────────────────────────────
print('\nComputing skip-channel gradient attribution for intervention…')
from codebase.models.mask_vae_aircraft import compute_skip_gradients
corr_idx  = compute_skip_gradients(
    lvae, clf, val_loader, device,
    n_observable=N_OBSERVABLE, top_k=24, n_batches=50)
corr_path = os.path.join(SAVE_DIR, 'corr_idx.pt')
torch.save(corr_idx, corr_path)
print(f'  Saved → {corr_path}')

summary = {
    'best_val_f1':      best_val_f1,
    'final_loss':       history['loss'][-1]   if history['loss']   else None,
    'final_rec':        history['rec'][-1]    if history['rec']    else None,
    'config': {
        'z_dim': Z_DIM, 'z1_dim': Z1_DIM, 'z2_dim': Z2_DIM, 'n_observable': N_OBSERVABLE,
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
        'clf_weight': CLF_WEIGHT,
    },
}
with open(os.path.join(SAVE_DIR, 'training_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print('Summary saved.')
