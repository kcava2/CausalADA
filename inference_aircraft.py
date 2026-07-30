"""
Aircraft Damage CausalVAE — Inference & Counterfactual Analysis

Usage:
    python inference_aircraft.py --image path/to/img.jpg --checkpoint checkpoints/aircraft_best.pt
    python inference_aircraft.py --image_dir path/to/dir/ --checkpoint checkpoints/aircraft_best.pt

Outputs per image (in --out_dir, default ./inference_results/):
    <stem>_causal_analysis.png — original + counterfactual columns (crack/dent
                                 prediction → remove both; no_damage → add both)
                                 with intervention Δ probability bars
    <stem>_causal_report.png   — learned DAG diagram
    <stem>_report.json         — full structured report
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from codebase.models.mask_vae_aircraft import CausalVAE
from codebase import utils as ut
from models.classifier_head import HierarchicalConceptClassifier
from dataset.aircraft_damage import IMAGE_SIZE, N_OBSERVABLE, CLASS_NAMES, SCALE
from plot_style import apply_journal_style, pretty

apply_journal_style()

# ── Constants ─────────────────────────────────────────────────────────────────
OBS_NAMES    = CLASS_NAMES[:N_OBSERVABLE]
LATENT_NAMES = CLASS_NAMES[N_OBSERVABLE:]  # impact_force, metal_fatigue

OBS_COLORS    = ['#DC3232', '#FFA500', '#32B432', '#508CFF']
LATENT_COLORS = ['#FF6B6B', '#4ECDC4', '#95E1D3']
BG            = 'white'

_MAINTENANCE = {
    'impact_force':  'Inspect for structural deformation. Check adjacent panels.',
    'metal_fatigue': 'Systemic issue. Review load cycle history. Inspect adjacent structure.',
}


# Causal downstream map (parent → children, from DAG)
_CAUSAL_DOWNSTREAM = {
    'structural_crack': [],
    'dent':             ['structural_crack'],
    'no_damage':        [],
    'impact_force':     ['structural_crack', 'dent'],
    'metal_fatigue':    ['structural_crack'],
}

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image',      type=str,   default=None)
parser.add_argument('--image_dir',  type=str,   default=None)
parser.add_argument('--checkpoint', type=str,   default='checkpoints/aircraft_best.pt')
parser.add_argument('--out_dir',    type=str,   default='./inference_results')
parser.add_argument('--threshold',       type=float, default=0.5)
parser.add_argument('--recompute_corr',  action='store_true',
                    help='Force recomputation of corr_idx.pt even if it already exists')
parser.add_argument('--damage_margin', type=float, default=0.0,
                    help='Logit added to the damaged score; >0 errs toward flagging damage')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
cfg  = ckpt.get('config', {'z_dim': 20, 'z1_dim': 5, 'z2_dim': 4})
Z_DIM, Z1_DIM, Z2_DIM = cfg['z_dim'], cfg['z1_dim'], cfg['z2_dim']
_N_OBS = cfg.get('n_observable', N_OBSERVABLE)

lvae = CausalVAE(
    name='aircraft_causalvae',
    z_dim=Z_DIM, z1_dim=Z1_DIM, z2_dim=Z2_DIM,
    inference=True,
    scale=SCALE,
    initial=False,
).to(device)
missing, unexpected = lvae.load_state_dict(ckpt['lvae'], strict=False)
if missing:
    print(f'  [WARN] Missing keys in checkpoint (will use random init): {missing[:4]}...')
if unexpected:
    print(f'  [WARN] Unexpected keys ignored: {unexpected[:4]}...')
lvae.eval()

clf = HierarchicalConceptClassifier(
    z2_dim=Z2_DIM, n_observable=_N_OBS, hidden=64).to(device)
clf.load_state_dict(ckpt['clf'])
if hasattr(clf, 'damage_margin'):
    clf.damage_margin.fill_(args.damage_margin)
clf.eval()

# ── Load skip-channel gradient attribution ───────────────────────────────────
_corr_path = Path(args.checkpoint).parent / 'corr_idx.pt'
if _corr_path.exists() and not args.recompute_corr:
    corr_idx = torch.load(_corr_path, map_location=device, weights_only=False)
    print(f'  Loaded corr_idx from {_corr_path}')
else:
    if args.recompute_corr:
        print(f'  [INFO] Recomputing corr_idx (--recompute_corr)…')
    else:
        print(f'  [INFO] corr_idx.pt not found; computing on the fly…')
    from dataset.aircraft_damage import get_dataloader
    from codebase.models.mask_vae_aircraft import compute_skip_gradients
    _DATA_ROOT  = './dataset'
    _tmp_loader = get_dataloader(_DATA_ROOT, 'valid', batch_size=64, num_workers=0)
    corr_idx = compute_skip_gradients(
        lvae, clf, _tmp_loader, device,
        n_observable=_N_OBS, top_k=24, n_batches=50)
    torch.save(corr_idx, _corr_path)
    print(f'  Saved corr_idx → {_corr_path}')

dag_weights = lvae.dag.A.detach().cpu().numpy()

print('Checkpoint loaded:', args.checkpoint)
print(f'  Epoch: {ckpt.get("epoch", "?")}')
print(f'  Architecture: z1_dim={Z1_DIM}, z2_dim={Z2_DIM}, z_dim={Z_DIM}, n_observable={_N_OBS}')
print('\nLearned DAG edge weights (row=cause, col=effect):')
all_names = CLASS_NAMES[:Z1_DIM]
header = '             ' + '  '.join(f'{n[:9]:>9}' for n in all_names)
print(header)
for i, row in enumerate(dag_weights):
    vals = '  '.join(f'{v:9.3f}' for v in row)
    print(f'  {all_names[i][:12]:>12}  {vals}')

# ── Image preprocessing ───────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
_denorm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_denorm_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_image(path: str) -> torch.Tensor:
    return _transform(Image.open(path).convert('RGB')).unsqueeze(0)


def to_display(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.detach().cpu().float() * _denorm_std + _denorm_mean
    return (torch.clamp(t, 0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ── Forward pass helpers ──────────────────────────────────────────────────────
def _forward_from_qm(q_m: torch.Tensor) -> torch.Tensor:
    """Run the structural-causal forward path from q_m to f_z1 (B, Z1, Z2)."""
    B = q_m.size(0)
    decode_m, _ = lvae.dag.calculate_dag(q_m, torch.ones_like(q_m))
    decode_m = decode_m.reshape([B, Z1_DIM, Z2_DIM])
    m_zm    = lvae.dag.mask_z(decode_m).reshape([B, Z1_DIM, Z2_DIM])
    f_z     = lvae.mask_z.mix(m_zm).reshape([B, Z1_DIM, Z2_DIM])
    e_tilde = lvae.attn.attention(decode_m, q_m)[0]
    return f_z + e_tilde


def _encode(img_tensor: torch.Tensor):
    """Encode image; return q_m, f_z1, z_dag, and skips for decoder."""
    with torch.no_grad():
        feat, skips = lvae.enc.encode(img_tensor)
        q_m_full, _ = ut.gaussian_parameters(feat, dim=1)   # (batch,64,1,1)
        q_m = lvae.enc_proj(q_m_full.view(img_tensor.size(0), -1))
        q_m = q_m.reshape([img_tensor.size(0), Z1_DIM, Z2_DIM])

        f_z1  = _forward_from_qm(q_m)
        z_dag = f_z1 + torch.randn_like(f_z1) * 1e-4
    return q_m, f_z1, z_dag, skips


def _latent_intervention_probs(q_m: torch.Tensor, parent_idx: int,
                               off_value: float = -3.0) -> np.ndarray:
    """
    do(parent = absent): drive the latent parent's encoded sub-vector to a
    strongly-negative value, re-propagate through the SCM so the child
    observable sub-vectors update, then re-classify. Returns (n_obs,) softmax.
    """
    with torch.no_grad():
        q_cf = q_m.clone()
        q_cf[:, parent_idx, :] = off_value
        return _classify(_forward_from_qm(q_cf))


def _decode(z_dag: torch.Tensor, skips) -> torch.Tensor:
    """Decode z_dag using stored encoder skips."""
    with torch.no_grad():
        z_4d = z_dag.reshape([z_dag.size(0), Z_DIM, 1, 1])
        return lvae.dec.decode(z_4d, skips)


def _classify(z_dag: torch.Tensor) -> np.ndarray:
    """Softmax class probabilities over the mutually-exclusive observable classes."""
    with torch.no_grad():
        logits = clf(z_dag.reshape(z_dag.size(0), -1))
        return torch.softmax(logits, dim=1).cpu().numpy()[0]


def _root_causes(z_dag: torch.Tensor) -> dict:
    """Extract latent root-cause probabilities from z_dag sub-vectors."""
    with torch.no_grad():
        rc = {}
        for i, name in enumerate(LATENT_NAMES):
            concept_idx = N_OBSERVABLE + i
            rc[name] = torch.sigmoid(z_dag[0, concept_idx, :].mean()).item()
    return rc


def _intervention_probs(f_z1: torch.Tensor, concept_idx: int,
                        target_value: float = -1.0) -> np.ndarray:
    """
    Do-operator: set concept_idx sub-vector to target_value in f_z1 and
    re-classify. target_value = -1 removes the concept, +1 forces it on.
    Returns (_N_OBS,) softmax probabilities — no image generation required.
    """
    with torch.no_grad():
        cf = f_z1.clone()
        cf[:, concept_idx, :] = target_value
        return _classify(cf)


def modulate_skips_soft(skips, concept_idx, corr_idx,
                        direction, base_strength=0.9, n_mod=24):
    """
    Modulate the encoder skip channels most associated with a concept, in the
    direction of the intervention.

    The U-Net decoder leans heavily on skip connections for spatial detail, so
    changing only the 20-D latent barely alters the output. To make a
    counterfactual visible we also act on the concept's most-attributed skip
    channels:

      direction = -1 (remove):  attenuate them, s·(1 - strength)  → erase the
                                damage-specific signal.
      direction = +1 (add):     amplify them,  s·(1 + 2·strength)  → push the
                                decoder toward rendering that concept's texture
                                on a surface that did not have it.

    Only the top-n_mod attributed channels per level are touched, so the
    background surface is preserved.
    """
    level_names = ['s1', 's2', 's3', 's4', 'b']
    strength    = base_strength
    result      = []

    for name, s in zip(level_names, skips):
        if name in corr_idx and concept_idx < corr_idx[name].size(0):
            s_mod = s.clone()
            ch    = corr_idx[name][concept_idx][:n_mod]
            if direction < 0:
                s_mod[:, ch] = s_mod[:, ch] * (1.0 - strength)
            else:
                s_mod[:, ch] = s_mod[:, ch] * (1.0 + 2.0 * strength)
            result.append(s_mod)
        else:
            result.append(s)

    return tuple(result)


def concept_counterfactual(f_z1, concept_idx, skips,
                           target_value=-1.0, base_strength=0.9):
    """
    Generate a counterfactual image for do(concept = target_value).
    target_value <= 0 removes the concept; > 0 adds it.
    """
    direction = -1 if target_value <= 0 else 1
    cf = f_z1.clone()
    cf[:, concept_idx, :] = target_value
    skips_cf = modulate_skips_soft(
        skips, concept_idx, corr_idx, direction,
        base_strength=base_strength)
    with torch.no_grad():
        z_4d = cf.reshape([1, Z_DIM, 1, 1])
        return lvae.dec.decode(z_4d, skips_cf)


def causal_consistency_score(ref_probs, latent_cf_probs, dag_weights,
                             n_observable=3, n_latent=2, threshold=0.02,
                             child_floor=0.1):
    """
    Measures whether removing a latent root cause produces the downstream
    reduction its learned DAG edges predict.

    Only latent→observable edges are testable. Observable→observable edges
    cannot be scored under single-label softmax: the class probabilities
    are normalised to sum to 1, so suppressing one class mechanically
    inflates the others. Latent causes sit outside that normalisation, so
    intervening on one can genuinely lower a child class (with the freed
    mass flowing to no_damage — the correct causal story).

    For each significant edge (child observable c ← latent parent p,
    |A[c, p]| >= 0.1), do(p = absent) should reduce P(c). The score is the
    fraction of such edges confirmed, in [0, 1], or None if none testable.

    A low score flags the prediction for human review: the model's causal
    reasoning on this image is inconsistent with its learned DAG, a
    per-image runtime assurance signal for safety-critical inspection.

    Args:
        ref_probs:       (n_obs,) baseline softmax probabilities
        latent_cf_probs: dict {parent_idx: (n_obs,) probs after do(parent=off)}
        dag_weights:     (Z1, Z1) learned adjacency (row=child, col=parent)
        threshold:       minimum probability drop counted as confirmed

    Returns:
        float in [0, 1] or None if no DAG edges are testable
    """
    confirmed = 0
    tested    = 0

    for p in range(n_observable, n_observable + n_latent):
        cf_p = latent_cf_probs[p]
        for c in range(n_observable):
            w = abs(float(dag_weights[c, p]))
            if w < 0.1:
                continue
            # Floor effect: if the child is already near zero, "reduce it
            # further" is not a meaningful test under softmax — skip it.
            if float(ref_probs[c]) < child_floor:
                continue
            delta = float(cf_p[c]) - float(ref_probs[c])
            tested += 1
            if delta < -threshold:
                confirmed += 1

    return confirmed / tested if tested > 0 else None


# ── Helpers ───────────────────────────────────────────────────────────────────
def ascii_bar(probs: np.ndarray, width: int = 30) -> str:
    lines = []
    for i, p in enumerate(probs):
        bar = '#' * int(p * width)
        lines.append(f'  {OBS_NAMES[i]:>12s} [{bar:<{width}}] {p*100:5.1f}%')
    return '\n'.join(lines)


def root_cause_bar(name: str, val: float, width: int = 10) -> str:
    filled = int(val * width)
    empty  = width - filled
    blocks = '█' * filled + '░' * empty
    return f'  {name:<14s} {blocks} {val:.2f}'


def causal_notes(probs: np.ndarray, threshold: float = 0.5) -> list:
    detected = [OBS_NAMES[i] for i, p in enumerate(probs) if p >= threshold]
    notes = []
    for src in detected:
        downstream = [d for d in _CAUSAL_DOWNSTREAM.get(src, []) if d in detected]
        if downstream:
            notes.append(f'{src} → {", ".join(downstream)}  (causal chain confirmed by DAG)')
        src_idx = CLASS_NAMES.index(src)
        for j, tgt in enumerate(CLASS_NAMES[:Z1_DIM]):
            if j != src_idx and abs(float(dag_weights[src_idx, j])) > 0.3:
                notes.append(
                    f'  DAG edge {src} → {tgt}: weight={dag_weights[src_idx, j]:.3f}'
                )
    if not notes:
        notes.append('No strong causal chains detected.')
    return notes


def _prob_bars(ax, probs, ref_probs=None, show_delta=False):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(0, 1.4)
    ax.set_ylim(-0.5, _N_OBS - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for i, p in enumerate(probs):
        y = _N_OBS - 1 - i
        # Short, capitalised label: "Crack", "Dent", "No Damage"
        lbl = pretty(OBS_NAMES[i]).replace('Structural ', '')
        ax.barh(y, float(p), height=0.55, color=OBS_COLORS[i])
        ax.text(-0.04, y, lbl, color='black', va='center', ha='right', fontsize=11)
        if show_delta and ref_probs is not None:
            delta = float(p) - float(ref_probs[i])
            sign  = '+' if delta >= 0 else ''
            dc    = '#1B7F5C' if delta < -0.05 else ('#B2362C' if delta > 0.05 else '#8593A0')
            ax.text(float(p) + 0.03, y, f'{sign}{delta:.2f}', color=dc,
                    va='center', fontsize=10, fontweight='bold')
        else:
            ax.text(float(p) + 0.03, y, f'{p*100:.0f}%', color='black',
                    va='center', fontsize=10)


def make_dag_figure(stem, dag_w, probs, save_path):
    """5-node DAG diagram showing all concepts."""
    obs_nodes = [
        ('structural_crack', 0.24, 0.28, 0),
        ('dent',             0.55, 0.28, 1),
        ('no_damage',        0.85, 0.28, 2),
    ]
    latent_nodes = [
        ('impact_force',  0.34, 0.80, 3),
        ('metal_fatigue', 0.72, 0.80, 4),
    ]
    all_nodes = obs_nodes + latent_nodes
    fig, ax = plt.subplots(figsize=(11.5, 7.2), facecolor=BG)
    ax.set_facecolor(BG); ax.set_xlim(0, 1); ax.set_ylim(0.05, 0.95); ax.axis('off')
    ax.grid(False)

    for _, sx, sy, si in all_nodes:
        for _, dx, dy, di in all_nodes:
            if di == si:
                continue
            w = float(dag_w[si, di])
            if abs(w) < 0.15:   # hide near-zero edges to keep the graph legible
                continue
            lw    = max(0.8, min(5.5, abs(w) * 8))
            alpha = 0.9 if abs(w) > 0.3 else 0.4
            color = '#C77A22' if abs(w) > 0.3 else '#B7BFC7'
            ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle='-|>', color=color,
                                        lw=lw, alpha=alpha, mutation_scale=16,
                                        connectionstyle='arc3,rad=0.08'))
            if abs(w) > 0.3:
                # Weight label near the source end, offset perpendicular + box
                t = 0.36
                mx, my = sx + t * (dx - sx), sy + t * (dy - sy)
                ddx, ddy = dx - sx, dy - sy
                norm = (ddx ** 2 + ddy ** 2) ** 0.5 or 1.0
                ox, oy = -ddy / norm * 0.05, ddx / norm * 0.05
                ax.text(mx + ox, my + oy, f'{abs(w):.2f}', fontsize=11,
                        color='#B44D18', ha='center', va='center',
                        fontweight='bold', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.18', fc='white',
                                  ec='#D9DEE4', alpha=0.95))

    pred_idx = int(np.argmax(probs[:_N_OBS]))
    for name, x, y, idx in obs_nodes:
        detected = (idx == pred_idx)
        color    = OBS_COLORS[idx] if detected else '#B7BFC7'
        ax.scatter([x], [y], s=7000, color=color, zorder=3,
                   edgecolors='black', linewidths=1.6)
        ax.text(x, y, pretty(name).replace(' ', '\n'), ha='center', va='center',
                color='white', fontsize=11.5, fontweight='bold', zorder=4)

    for name, x, y, idx in latent_nodes:
        lc = LATENT_COLORS[idx - N_OBSERVABLE]
        ax.scatter([x], [y], s=7000, color=lc, zorder=3, marker='D',
                   edgecolors='black', linewidths=1.6)
        ax.text(x, y, pretty(name).replace(' ', '\n'), ha='center', va='center',
                color='white', fontsize=11, fontweight='bold', zorder=4)

    ax.scatter([0.03], [0.07], s=150, color='#8593A0', transform=ax.transAxes)
    ax.text(0.055, 0.07, 'Observable Concept', color='black', fontsize=11.5,
            va='center', transform=ax.transAxes)
    ax.scatter([0.40], [0.07], s=150, color='#8593A0', marker='D', transform=ax.transAxes)
    ax.text(0.425, 0.07, 'Latent Root Cause', color='black', fontsize=11.5,
            va='center', transform=ax.transAxes)
    ax.set_title('Learned Causal Structure  (Predicted Class Highlighted)',
                 fontsize=16, pad=14)

    plt.savefig(save_path, dpi=200, facecolor=BG, bbox_inches='tight')
    plt.close(fig)


def make_causal_figure(stem, orig_img, cf_imgs, cf_probs,
                       ref_probs, plan, save_path, header=''):
    """
    2-row × (1 + len(plan))-col causal analysis figure. The interventions are
    keyed on the predicted class: a crack/dent prediction is probed by removing
    crack and removing dent; a no_damage prediction by adding crack and adding
    dent.
      Row 1: counterfactual images (skip-modulated)
      Row 2: causal intervention Δ probability bars
    """
    N = 1 + len(plan)

    fig = plt.figure(figsize=(N * 3.1, 7.2), facecolor=BG)
    fig.suptitle('Counterfactual Causal Analysis', fontsize=18, fontweight='bold')
    if header:
        fig.text(0.5, 0.925, header, ha='center', va='top',
                 fontsize=13, color='#43505C')
    gs = fig.add_gridspec(2, N, height_ratios=[3, 2.2],
                          hspace=0.30, wspace=0.14,
                          left=0.10, right=0.98, top=0.86, bottom=0.05)

    for c in range(N):
        ax = fig.add_subplot(gs[0, c])
        ax.set_facecolor(BG)
        if c == 0:
            ax.imshow(to_display(orig_img[0]))
            ax.set_title('Original', fontsize=13, pad=5)
        else:
            p = plan[c - 1]
            ax.imshow(to_display(cf_imgs[c - 1][0]))
            ax.set_title(p['title'], fontsize=13, pad=5)
        ax.axis('off')
        if c == 0:
            ax.set_ylabel('Counterfactual Image', color='#43505C', fontsize=12)

    for c in range(N):
        ax = fig.add_subplot(gs[1, c])
        if c == 0:
            _prob_bars(ax, ref_probs)
            ax.set_title('Baseline', fontsize=13, pad=5)
        else:
            p = plan[c - 1]
            _prob_bars(ax, cf_probs[c - 1], ref_probs=ref_probs, show_delta=True)
            ax.set_title(p['title'], fontsize=13, pad=5)

    plt.savefig(save_path, dpi=200, facecolor=BG, bbox_inches='tight')
    plt.close(fig)


# ── Per-image analysis ────────────────────────────────────────────────────────
def analyse_image(img_path: str, out_dir: str) -> dict:
    stem       = Path(img_path).stem
    img_tensor = load_image(img_path).to(device)

    q_m, f_z1, z_dag, skips = _encode(img_tensor)
    ref_probs           = _classify(z_dag)                 # softmax, sums to 1
    pred_idx            = int(np.argmax(ref_probs))
    pred_class          = OBS_NAMES[pred_idx]
    root_causes         = _root_causes(z_dag)
    dominant_cause      = max(root_causes, key=root_causes.get)
    maintenance         = _MAINTENANCE[dominant_cause]

    # Intervention plan keyed on the predicted class over the two damage
    # concepts (crack, dent):
    #   predicted crack / dent  →  remove crack, remove dent
    #   predicted no_damage     →  add crack,    add dent
    DAMAGE_IDX = [0, 1]   # structural_crack, dent
    add_mode   = (pred_class == 'no_damage')
    mode       = 'add' if add_mode else 'remove'
    plan = []
    for idx in DAMAGE_IDX:
        plan.append({
            'idx':    idx,
            'name':   OBS_NAMES[idx],
            'target': 1.0 if add_mode else -1.0,   # latent value to set
            'do':     1 if add_mode else 0,        # displayed do() value
            'mode':   mode,
            'title':  f'{mode.capitalize()} {pretty(OBS_NAMES[idx])}',  # e.g. "Remove Dent"
        })

    # Counterfactual images (skip-modulated: attenuate to remove, amplify to add)
    cf_imgs = [concept_counterfactual(f_z1, p['idx'], skips,
                   target_value=p['target'])
               for p in plan]

    # Observable interventions (for the counterfactual delta bars in the figure)
    cf_probs = [_intervention_probs(f_z1, p['idx'], target_value=p['target'])
                for p in plan]

    # Latent root-cause interventions (re-propagated) — for causal consistency
    latent_cf = {p: _latent_intervention_probs(q_m, p)
                 for p in range(_N_OBS, Z1_DIM)}
    consistency = causal_consistency_score(
        ref_probs, latent_cf, dag_weights,
        n_observable=_N_OBS, n_latent=Z1_DIM - _N_OBS)

    notes = causal_notes(ref_probs, args.threshold)

    cf_causal_indicators = []
    for k, p in enumerate(plan):
        indicators = {}
        for j, tgt in enumerate(OBS_NAMES):
            if j == p['idx']:
                continue
            delta = float(cf_probs[k][j] - ref_probs[j])
            if delta < -0.1:
                indicators[tgt] = 'downstream_reduced'
            elif delta > 0.1:
                indicators[tgt] = 'downstream_increased'
            else:
                indicators[tgt] = 'no_change'
        cf_causal_indicators.append({f"do({p['name']}={p['do']})": indicators})

    # Console output
    print(f'\n{"="*60}\nImage: {img_path}')
    print('\nClass probabilities (softmax):')
    print(ascii_bar(ref_probs))
    print(f'\nPredicted class: {pred_class}  ({ref_probs[pred_idx]*100:.1f}%)')
    print(f'Causal consistency: {consistency if consistency is not None else "n/a"}')
    print('\nRoot causes:')
    for name, val in root_causes.items():
        print(root_cause_bar(name, val))
    print(f'  → Dominant: {dominant_cause}')
    print(f'  → {maintenance}')
    print('\nCausal notes:')
    for n in notes: print(f'  {n}')

    os.makedirs(out_dir, exist_ok=True)
    causal_path   = os.path.join(out_dir, f'{stem}_causal_analysis.png')
    dag_path      = os.path.join(out_dir, f'{stem}_causal_report.png')

    _header  = (f'Predicted: {pretty(pred_class)} '
                f'({ref_probs[pred_idx]*100:.0f}% confidence)')
    make_causal_figure(stem, img_tensor.cpu(),
                       [cf.cpu() for cf in cf_imgs],
                       cf_probs, ref_probs, plan, causal_path,
                       header=_header)
    make_dag_figure(stem, dag_weights, ref_probs, dag_path)

    report = {
        'image':         img_path,
        'probabilities': {n: float(ref_probs[i]) for i, n in enumerate(OBS_NAMES)},
        'predicted_class': pred_class,
        'predicted_confidence': float(ref_probs[pred_idx]),
        'latent_root_causes': root_causes,
        'dominant_cause':     dominant_cause,
        'maintenance_recommendation': maintenance,
        'causal_notes':  notes,
        'counterfactual_deltas': {
            f"do({p['name']}={p['do']})": {
                OBS_NAMES[j]: float(cf_probs[k][j] - ref_probs[j])
                for j in range(_N_OBS)
            }
            for k, p in enumerate(plan)
        },
        'causal_indicators': cf_causal_indicators,
        'causal_consistency_score': consistency,
        'latent_intervention_deltas': {
            CLASS_NAMES[p]: {OBS_NAMES[c]: float(latent_cf[p][c] - ref_probs[c])
                             for c in range(_N_OBS)}
            for p in range(_N_OBS, Z1_DIM)
        },
        'dag_weights': {
            f'{CLASS_NAMES[i]}_to_{CLASS_NAMES[j]}': float(dag_weights[i, j])
            for i in range(Z1_DIM) for j in range(Z1_DIM)
            if abs(float(dag_weights[i, j])) > 0.05
        },
    }
    report_path = os.path.join(out_dir, f'{stem}_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f'  → {causal_path}')
    print(f'  → {dag_path}')
    print(f'  → {report_path}')
    return report


# ── Main ──────────────────────────────────────────────────────────────────────
if args.image is None and args.image_dir is None:
    parser.error('Provide --image or --image_dir')

image_paths = []
if args.image:
    image_paths.append(args.image)
if args.image_dir:
    exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    # Recursive: picks up images in --image_dir and any subfolders (e.g. the
    # per-class folders under dataset/test/).
    image_paths.extend(
        str(p) for p in sorted(Path(args.image_dir).rglob('*'))
        if p.is_file() and p.suffix in exts
    )

if not image_paths:
    print('No images found.'); sys.exit(0)

all_reports = []
for img_path in image_paths:
    try:
        all_reports.append(analyse_image(img_path, args.out_dir))
    except Exception as e:
        print(f'ERROR processing {img_path}: {e}')
        import traceback; traceback.print_exc()

if len(all_reports) > 1:
    csv_path   = os.path.join(args.out_dir, 'summary.csv')
    fieldnames = (['image', 'predicted_class', 'predicted_confidence']
                  + [f'prob_{n}' for n in OBS_NAMES]
                  + [f'rc_{n}' for n in LATENT_NAMES]
                  + ['dominant_cause', 'causal_consistency_score'])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_reports:
            row = {
                'image':                    r['image'],
                'predicted_class':          r['predicted_class'],
                'predicted_confidence':     f"{r['predicted_confidence']:.4f}",
                'dominant_cause':           r['dominant_cause'],
                'causal_consistency_score': r['causal_consistency_score'],
            }
            for n in OBS_NAMES:
                row[f'prob_{n}'] = f"{r['probabilities'][n]:.4f}"
            for n in LATENT_NAMES:
                row[f'rc_{n}'] = f"{r['latent_root_causes'][n]:.4f}"
            writer.writerow(row)
    print(f'\nBatch summary → {csv_path}')

print(f'\nDone. {len(all_reports)} image(s) processed.')
