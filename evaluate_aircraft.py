"""
Test-set evaluation for Aircraft Damage CausalVAE.

Computes:
  0. Existing metrics
     Damage type detection — F1, precision, recall per class + macro
     Reconstruction        — MSE

  1. AUC-ROC               — per-class and macro, with ROC curve plot
  2. Concept Activation    — specificity heatmap and distributions
  3. MIC / TIC             — Mutual Information Completeness / Total Information Content

Run from repo root:
    python evaluate_aircraft.py --checkpoint checkpoints/aircraft_best.pt
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score, roc_curve, mutual_info_score,
    f1_score, precision_score, recall_score,
)

from sklearn.metrics import confusion_matrix

from codebase import utils as ut
from codebase.models.mask_vae_aircraft import CausalVAE
from dataset.aircraft_damage import get_dataloader, CLASS_NAMES, N_CONCEPTS, N_OBSERVABLE, SCALE
from models.classifier_head import HierarchicalConceptClassifier, compute_metrics
from plot_style import apply_journal_style, pretty

apply_journal_style()

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='checkpoints/aircraft_best.pt')
parser.add_argument('--data_root',  default='./dataset')
parser.add_argument('--split',      default='test',
                    help='Dataset split to evaluate: test, valid, or train')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--out',        default='./eval_results.json',
                    help='Where to save the JSON report')
parser.add_argument('--damage_margin', type=float, default=0.0,
                    help='Logit added to the damaged score; >0 errs toward flagging damage')
args = parser.parse_args()

# ── Colours (colour-blind-safe, Okabe-Ito) ─────────────────────────────────────
CLASS_COLORS  = ['#0072B2', '#E69F00', '#009E73', '#56B4E9']
LATENT_COLORS = ['#CC79A7', '#56B4E9', '#0072B2']
ALL_COLORS    = CLASS_COLORS + LATENT_COLORS
OBS_NAMES     = CLASS_NAMES[:N_OBSERVABLE]
LATENT_NAMES  = CLASS_NAMES[N_OBSERVABLE:N_CONCEPTS]

os.makedirs('eval_plots', exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device     : {device}')
print(f'Checkpoint : {args.checkpoint}')
print(f'Split      : {args.split}')

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
cfg  = ckpt.get('config', {'z_dim': 20, 'z1_dim': 5, 'z2_dim': 4})
Z_DIM, Z1_DIM, Z2_DIM = cfg['z_dim'], cfg['z1_dim'], cfg['z2_dim']
_N_OBS = cfg.get('n_observable', N_OBSERVABLE)

lvae = CausalVAE(
    z_dim=Z_DIM, z1_dim=Z1_DIM, z2_dim=Z2_DIM,
    inference=True, scale=SCALE, initial=False,
).to(device)
lvae.load_state_dict(ckpt['lvae'], strict=False)
lvae.eval()

clf = HierarchicalConceptClassifier(
    z2_dim=Z2_DIM, n_observable=_N_OBS, hidden=64).to(device)
clf.load_state_dict(ckpt['clf'])
if hasattr(clf, 'damage_margin'):
    clf.damage_margin.fill_(args.damage_margin)
    print(f'Damage margin : {args.damage_margin:+.2f}  (>0 errs toward damage)')
clf.eval()

dag_weights = lvae.dag.A.detach().cpu().numpy()

# ── Data ──────────────────────────────────────────────────────────────────────
loader = get_dataloader(args.data_root, args.split, args.batch_size, num_workers=0)
print(f'Images     : {len(loader.dataset)}')


# ── Latent-intervention helpers (for causal consistency) ──────────────────────
def _encode_qm(imgs):
    """Encode a batch of images to the pre-DAG concept means q_m (B, Z1, Z2)."""
    feat, _ = lvae.enc.encode(imgs)
    q_m_full, _ = ut.gaussian_parameters(feat, dim=1)
    q_m = lvae.enc_proj(q_m_full.view(imgs.size(0), -1))
    return q_m.reshape([imgs.size(0), Z1_DIM, Z2_DIM])


def _forward_from_qm(q_m):
    """Run the structural-causal forward path from q_m to f_z1 (B, Z1, Z2)."""
    B = q_m.size(0)
    decode_m, _ = lvae.dag.calculate_dag(q_m, torch.ones_like(q_m))
    decode_m = decode_m.reshape([B, Z1_DIM, Z2_DIM])
    m_zm    = lvae.dag.mask_z(decode_m).reshape([B, Z1_DIM, Z2_DIM])
    f_z     = lvae.mask_z.mix(m_zm).reshape([B, Z1_DIM, Z2_DIM])
    e_tilde = lvae.attn.attention(decode_m, q_m)[0]
    return f_z + e_tilde


def _class_probs(f_z1):
    """Softmax class probabilities over the observable classes (B, n_obs)."""
    return torch.softmax(clf(f_z1), dim=1).cpu().numpy()


def _latent_intervention_probs(q_m, parent_idx, off_value=-3.0):
    """
    do(parent = absent): drive the parent concept's encoded sub-vector to a
    strongly-negative value, re-propagate through the SCM so the child
    observable sub-vectors update, then re-classify. Returns (B, n_obs).
    """
    q_cf = q_m.clone()
    q_cf[:, parent_idx, :] = off_value
    return _class_probs(_forward_from_qm(q_cf))


def causal_consistency_score(ref_probs, latent_cf_probs, dag_weights,
                             n_observable=3, n_latent=2, threshold=0.02,
                             child_floor=0.1):
    """
    Measures whether removing a latent root cause produces the downstream
    reduction its learned DAG edges predict.

    Only latent→observable edges are testable here. Observable→observable
    edges cannot be scored under single-label softmax: the class
    probabilities are normalised to sum to 1, so suppressing one class
    mechanically inflates the others. Latent causes sit outside that
    normalisation, so intervening on them can genuinely lower a child
    class (with the freed mass flowing to no_damage — the correct causal
    story).

    For each significant edge (child observable c ← latent parent p,
    |A[c, p]| >= 0.1), do(p = absent) should reduce P(c). The score is the
    fraction of such edges confirmed, in [0, 1], or None if none testable.

    Args:
        ref_probs:       (n_obs,) baseline softmax probabilities
        latent_cf_probs: dict {parent_idx: (n_obs,) probs after do(parent=off)}
        dag_weights:     (Z1, Z1) learned adjacency (row=child, col=parent)
        threshold:       minimum probability drop counted as confirmed
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


# ── Main evaluation loop (collect predictions + latents) ─────────────────────
all_logits   = []
all_labels   = []
all_z_dag    = []
total_rec    = 0.0
n_batches    = 0

print('\nRunning forward pass over test set…')
with torch.no_grad():
    for imgs, labels in tqdm(loader, desc='Forward pass'):
        imgs, labels = imgs.to(device), labels.to(device)

        _, _, rec, _, z_dag = lvae.negative_elbo_bound(imgs, labels, sample=False)

        logits   = clf(z_dag)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_z_dag.append(z_dag.cpu())
        total_rec += rec.item()
        n_batches += 1

logits_all   = torch.cat(all_logits)
labels_all   = torch.cat(all_labels)
z_dag_all    = torch.cat(all_z_dag)              # (N, Z1_DIM, Z2_DIM)
labels_obs   = labels_all[:, :_N_OBS].numpy()   # (N, n_obs) binary observable

# ── Single-label (softmax / argmax) metrics ───────────────────────────────────
avg_rec = total_rec / max(n_batches, 1)

probs_all = torch.softmax(logits_all, dim=1).numpy()   # (N, n_obs), rows sum to 1
preds_arg = probs_all.argmax(axis=1)                   # (N,) predicted class
y_true    = labels_obs.argmax(axis=1)                  # (N,) ground-truth class

metrics = compute_metrics(logits_all, labels_all[:, :_N_OBS], class_names=OBS_NAMES)
overall_acc = metrics['accuracy']
# Per-class recall doubles as per-class accuracy for single-label classification.
acc_per_class = {name: metrics[f'recall_{name}'] for name in OBS_NAMES}

# ── Print existing metrics ─────────────────────────────────────────────────────
SEP = '─' * 52
print(f'\n{SEP}')
print(f'  Evaluation results  ({args.split} split)')
print(SEP)

print('\n  DAMAGE TYPE CLASSIFICATION  (single-label, argmax)')
print(f'    {"Class":<12}  {"F1":>6}  {"Precision":>10}  {"Recall":>8}')
print(f'    {"-"*12}  {"------":>6}  {"----------":>10}  {"--------":>8}')
for name in OBS_NAMES:
    f1   = metrics.get(f'f1_{name}',        0.0)
    prec = metrics.get(f'precision_{name}', 0.0)
    rec  = metrics.get(f'recall_{name}',    0.0)
    print(f'    {name:<12}  {f1:6.3f}  {prec:10.3f}  {rec:8.3f}')
print(f'    {"─"*12}  {"─"*6}  {"─"*10}  {"─"*8}')
print(f'    {"MACRO":<12}  {metrics["f1_macro"]:6.3f}  '
      f'{metrics["precision_macro"]:10.3f}  {metrics["recall_macro"]:8.3f}')

print('\n  PER-CLASS RECALL  (single-label)')
print(f'    {"Class":<12}  {"Recall":>9}')
print(f'    {"-"*12}  {"─"*9}')
for name in OBS_NAMES:
    print(f'    {name:<12}  {acc_per_class[name]*100:8.1f}%')
print(f'    {"─"*12}  {"─"*9}')
print(f'    {"TOP-1 ACC":<12}  {overall_acc*100:8.1f}%')

print(f'\n  RECONSTRUCTION')
print(f'    Avg MSE         : {avg_rec:.5f}')


# ══════════════════════════════════════════════════════════════════════════════
# SEVERITY REGRESSION  (damage instance count)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  Confusion matrix')
print(SEP)

cm      = confusion_matrix(y_true, preds_arg, labels=list(range(_N_OBS)))
cm_row  = cm.sum(axis=1, keepdims=True).clip(min=1)
cm_norm = cm / cm_row
pretty_labels = [pretty(n) for n in OBS_NAMES]

print(f'    {"":16}' + ''.join(f'{p[:12]:>13}' for p in pretty_labels) + '   (predicted)')
for i, name in enumerate(pretty_labels):
    print(f'    {name:<16}' + ''.join(f'{cm[i, j]:>13d}' for j in range(_N_OBS)))

confusion_report = {
    'labels': OBS_NAMES,
    'matrix': cm.tolist(),
    'row_normalised': cm_norm.round(4).tolist(),
}

fig, ax = plt.subplots(figsize=(7.4, 6.8))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
ax.grid(False)
for i in range(_N_OBS):
    for j in range(_N_OBS):
        tc = 'white' if cm_norm[i, j] > 0.55 else '#1A2530'
        ax.text(j, i, f'{cm[i, j]}\n{cm_norm[i, j]*100:.1f}%', ha='center', va='center',
                fontsize=13.5, color=tc)
ax.set_xticks(range(_N_OBS)); ax.set_yticks(range(_N_OBS))
ax.set_xticklabels(pretty_labels, fontsize=12.5)
ax.set_yticklabels(pretty_labels, fontsize=12.5, rotation=90, va='center')
ax.set_xlabel('Predicted Class')
ax.set_ylabel('True Class')
ax.set_title('Confusion Matrix\n(counts with row-normalised percentage)', fontsize=16)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Row-normalised frequency')
plt.tight_layout()
plt.savefig('eval_plots/confusion_matrix.png')
plt.close(fig)
print('    → eval_plots/confusion_matrix.png')


# ══════════════════════════════════════════════════════════════════════════════
# EVAL 1 — AUC-ROC
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  Eval 1: AUC-ROC')
print(SEP)

auc_per_class = {}
for j, name in enumerate(OBS_NAMES):
    try:
        auc_per_class[name] = float(roc_auc_score(labels_obs[:, j], probs_all[:, j]))
    except ValueError:
        auc_per_class[name] = float('nan')

valid_aucs = [v for v in auc_per_class.values() if not math.isnan(v)]
macro_auc  = float(np.mean(valid_aucs)) if valid_aucs else float('nan')

for name, auc in auc_per_class.items():
    print(f'    {name:<12}  AUC = {auc:.3f}')
print(f'    {"MACRO":<12}  AUC = {macro_auc:.3f}')

# ROC curve plot
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], '--', color='#aaaaaa', linewidth=1, label='Random')
for j, name in enumerate(OBS_NAMES):
    try:
        fpr, tpr, _ = roc_curve(labels_obs[:, j], probs_all[:, j])
        auc_val = auc_per_class[name]
        ax.plot(fpr, tpr, color=CLASS_COLORS[j], linewidth=2.2,
                label=f'{pretty(name)}  (AUC = {auc_val:.3f})')
    except ValueError:
        pass
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves by Damage Class (one-vs-rest)')
ax.legend(fontsize=11.5, loc='lower right')
plt.tight_layout()
plt.savefig('eval_plots/roc_curves.png', dpi=200)
plt.close(fig)
print('    → eval_plots/roc_curves.png')




# ══════════════════════════════════════════════════════════════════════════════
# EVAL 3 — CONCEPT ACTIVATION CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  Eval 3: Concept Activation Consistency')
print(SEP)

# Compute per-concept mean activation per image
# z_dag_all shape: (N, Z1_DIM, Z2_DIM)
act_matrix = torch.sigmoid(z_dag_all.mean(dim=-1)).numpy()   # (N, Z1_DIM)

# 7×4 mean activation table: table[concept_i, obs_class_j]
table = np.zeros((Z1_DIM, _N_OBS))
for i in range(Z1_DIM):
    for j in range(_N_OBS):
        mask_j = labels_obs[:, j] == 1
        if mask_j.sum() > 0:
            table[i, j] = act_matrix[mask_j, i].mean()
        else:
            table[i, j] = float('nan')

# Specificity for observable concepts (diagonal vs off-diagonal)
spec_per_concept = {}
for i in range(_N_OBS):
    row     = table[i, :]
    diag    = table[i, i]
    off_mean= np.nanmean([table[i, j] for j in range(_N_OBS) if j != i])
    spec_per_concept[OBS_NAMES[i]] = float(diag - off_mean)
    print(f'    {OBS_NAMES[i]:<12}  specificity = {diag - off_mean:+.3f}'
          f'  (diag={diag:.3f}, off-diag mean={off_mean:.3f})')

mean_specificity = float(np.mean(list(spec_per_concept.values())))
print(f'    Mean specificity  : {mean_specificity:.3f}')

# ── Plot 1: Activation matrix heatmap ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 8))
ax.grid(False)

im = ax.imshow(table, cmap='viridis', vmin=0, vmax=1,
               aspect='auto', interpolation='nearest')

# Expected high-activation cells
expected = {
    0: [0],        # structural_crack → structural_crack
    1: [1],        # dent             → dent
    2: [2],        # no_damage        → no_damage
    3: [0, 1],     # impact_force     → structural_crack + dent
    4: [0],        # metal_fatigue    → structural_crack
}
for i in range(Z1_DIM):
    for j in range(_N_OBS):
        val = table[i, j]
        txt = f'{val:.2f}' if not np.isnan(val) else '-'
        star = '★' if j in expected.get(i, []) else ''
        tc  = 'white' if (not np.isnan(val) and val < 0.55) else 'black'
        ax.text(j, i, txt, ha='center', va='center', fontsize=13, color=tc)
        if star:
            ax.text(j + 0.34, i - 0.30, star, ha='center', va='center',
                    fontsize=13, color='#E69F00')

ax.set_xticks(range(_N_OBS))
ax.set_xticklabels([pretty(n) for n in OBS_NAMES], fontsize=12.5)
ax.set_yticks(range(Z1_DIM))
yticklabels = (
    [f'{pretty(n)} (observed)' for n in OBS_NAMES]
    + [f'{pretty(n)} (latent)' for n in LATENT_NAMES]
)
ax.set_yticklabels(yticklabels[:Z1_DIM], fontsize=12.5)
ax.set_xlabel('Ground-Truth Damage Class')
ax.set_ylabel('Concept Sub-Vector')
ax.set_title('Concept Activation Specificity\n(★ marks the expected high-activation cell)',
             fontsize=16)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Mean Activation (sigmoid)')
plt.tight_layout()
plt.savefig('eval_plots/concept_activation_matrix.png', dpi=200)
plt.close(fig)
print('    → eval_plots/concept_activation_matrix.png')

# ── Plot 2: Violin / distribution plots for observable concepts ───────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle('Concept Activation by Class Presence', fontsize=18, fontweight='bold')

for idx, (ax, name) in enumerate(zip(axes.flat, OBS_NAMES)):
    mask_present = labels_obs[:, idx] == 1
    mask_absent  = labels_obs[:, idx] == 0
    vals_present = act_matrix[mask_present, idx]
    vals_absent  = act_matrix[mask_absent,  idx]

    data = [vals_absent, vals_present]
    parts = ax.violinplot(data, positions=[0, 1],
                          showmeans=True, showmedians=True)
    for i_v, pc in enumerate(parts['bodies']):
        pc.set_facecolor(CLASS_COLORS[idx])
        pc.set_alpha(0.6 if i_v == 1 else 0.3)
    for key in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
        if key in parts:
            parts[key].set_color('black')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Class Absent', 'Class Present'], fontsize=12)
    ax.set_ylabel('Sub-Vector Activation')
    ax.set_title(pretty(name), color=CLASS_COLORS[idx], fontsize=14)
    ax.grid(True, axis='y')
    for sp in ax.spines.values():
        sp.set_edgecolor('#cccccc')

plt.tight_layout()
plt.savefig('eval_plots/concept_activation_distributions.png', dpi=200)
plt.close(fig)
print('    → eval_plots/concept_activation_distributions.png')


# ══════════════════════════════════════════════════════════════════════════════
# EVAL 4 — MIC / TIC (Mutual Information Completeness / Total Information Content)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  Eval 4: Disentanglement (MIC / TIC)')
print(SEP)

# MI matrix: mi_matrix[i, j] = MI(activation of z_i, label g_j)  shape (Z1_DIM, _N_OBS)
mi_matrix = np.zeros((Z1_DIM, _N_OBS))
for i in range(Z1_DIM):
    acts_i = act_matrix[:, i]
    bins_i = np.digitize(acts_i, np.percentile(acts_i, np.linspace(0, 100, 11)[1:-1]))
    for j in range(_N_OBS):
        mi_matrix[i, j] = mutual_info_score(labels_obs[:, j].astype(int), bins_i)

# MIC — completeness: for each ground-truth factor, best-matching latent
mic_per_class = {OBS_NAMES[j]: float(np.max(mi_matrix[:, j])) for j in range(_N_OBS)}
overall_mic   = float(np.mean(list(mic_per_class.values())))

# TIC — utility: for each latent, best-matching ground-truth factor
tic_per_concept = {CLASS_NAMES[i]: float(np.max(mi_matrix[i, :])) for i in range(Z1_DIM)}
overall_tic     = float(np.mean(list(tic_per_concept.values())))

# Normalise against mean binary entropy of labels (nats) for radar chart
h_vals = []
for j in range(_N_OBS):
    p = labels_obs[:, j].mean()
    if 0 < p < 1:
        h_vals.append(-(p * math.log(p) + (1 - p) * math.log(1 - p)))
h_mean  = float(np.mean(h_vals)) if h_vals else 0.693
mic_norm = min(overall_mic / h_mean, 1.0)
tic_norm = min(overall_tic / h_mean, 1.0)

def _mi_interp(mi: float) -> str:
    if mi > 0.30:  return 'Excellent'
    if mi > 0.15:  return 'Good'
    if mi > 0.05:  return 'Acceptable'
    return 'Poor'

print('  MIC (coverage — each ground-truth factor captured by best latent):')
for name, mic_j in mic_per_class.items():
    print(f'    {name:<12}  MIC = {mic_j:.3f}  ({_mi_interp(mic_j)})')
print(f'    Overall MIC = {overall_mic:.3f}  ({_mi_interp(overall_mic)})')

print('\n  TIC (utility — each latent captures something meaningful):')
for name, tic_i in tic_per_concept.items():
    print(f'    {name:<16}  TIC = {tic_i:.3f}  ({_mi_interp(tic_i)})')
print(f'    Overall TIC = {overall_tic:.3f}  ({_mi_interp(overall_tic)})')

mic_interp_str = f'MIC: {overall_mic:.3f} — {_mi_interp(overall_mic)} coverage of ground-truth factors'
tic_interp_str = f'TIC: {overall_tic:.3f} — {_mi_interp(overall_tic)} latent concept utility'


# ══════════════════════════════════════════════════════════════════════════════
# EVAL 5 — CAUSAL CONSISTENCY (latent→observable interventions, re-propagated)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  Eval 5: Causal Consistency (latent root-cause interventions)')
print(SEP)

_N_LAT = Z1_DIM - _N_OBS
cc_scores = []
with torch.no_grad():
    for imgs, labels in tqdm(loader, desc='Causal consistency'):
        imgs  = imgs.to(device)
        q_m   = _encode_qm(imgs)
        ref   = _class_probs(_forward_from_qm(q_m))             # (B, n_obs)
        cf_by_parent = {p: _latent_intervention_probs(q_m, p)   # (B, n_obs)
                        for p in range(_N_OBS, _N_OBS + _N_LAT)}
        for b in range(imgs.size(0)):
            ref_b = ref[b]
            cf_b  = {p: cf_by_parent[p][b] for p in cf_by_parent}
            s = causal_consistency_score(
                ref_b, cf_b, dag_weights,
                n_observable=_N_OBS, n_latent=_N_LAT)
            if s is not None:
                cc_scores.append(s)

if cc_scores:
    cc_arr        = np.array(cc_scores)
    cc_mean       = float(cc_arr.mean())
    cc_frac_high  = float(np.mean(cc_arr >= 0.7))
    cc_frac_flag  = float(np.mean(cc_arr < 0.3))
else:
    cc_mean = cc_frac_high = cc_frac_flag = float('nan')

print(f'    Mean consistency score       : {cc_mean:.3f}')
print(f'    Fraction with score >= 0.7   : {cc_frac_high:.3f}')
print(f'    Fraction with score <  0.3   : {cc_frac_flag:.3f}  (flagged for review)')


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY RADAR CHART
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  Summary radar chart')
print(SEP)

radar_labels  = ['Macro\nAUC-ROC', 'Top-1\naccuracy', 'Mean\nspecificity',
                 'MIC', 'TIC']
macro_auc_clipped = max(0.0, min(1.0, macro_auc if not math.isnan(macro_auc) else 0.0))
spec_norm         = max(0.0, min(1.0, (mean_specificity + 0.5) / 1.0))
radar_values      = [macro_auc_clipped, overall_acc, spec_norm, mic_norm, tic_norm]

N_axes  = len(radar_labels)
angles  = [n / N_axes * 2 * math.pi for n in range(N_axes)]
angles += angles[:1]
vals    = radar_values + radar_values[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

ax.plot(angles, vals, color='#0072B2', linewidth=2.2)
ax.fill(angles, vals, color='#0072B2', alpha=0.20)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=12)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], color='#5A6672', fontsize=10)
ax.spines['polar'].set_color('#cccccc')
ax.grid(color='#dddddd', linestyle='--', alpha=0.6)
ax.set_title('Model Evaluation Summary\n(all axes normalised to [0, 1])',
             fontsize=15, pad=26)

for angle, val in zip(angles[:-1], radar_values):
    ax.text(angle, val + 0.08, f'{val:.2f}', ha='center', va='center',
            fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('eval_plots/evaluation_summary.png', dpi=200)
plt.close(fig)
print('    → eval_plots/evaluation_summary.png')


# ══════════════════════════════════════════════════════════════════════════════
# DAG STRUCTURE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
# Topological layers: latent roots → observable intermediate → observable sink
# dent causes crack, so it sits in the middle row
_dag_nodes = {
    # idx: (label, x, y)   — no_damage sits alone on the left (it has no edges)
    3: ('impact_force',     0.42, 0.86),
    4: ('metal_fatigue',    0.74, 0.86),
    1: ('dent',             0.68, 0.52),
    2: ('no_damage',        0.16, 0.52),
    0: ('structural_crack', 0.46, 0.16),
}
_dag_edges = [
    (3, 0), (3, 1),
    (4, 0),
    (1, 0),
]

fig, ax = plt.subplots(figsize=(9, 8.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); ax.grid(False)

# Layer labels on the far right (kept clear of every node)
ax.text(0.985, 0.86, 'Latent\nRoot Causes',  color='#5A6672', fontsize=11.5,
        va='center', ha='right', transform=ax.transAxes)
ax.text(0.985, 0.52, 'Observable\n(Intermediate)', color='#5A6672', fontsize=11.5,
        va='center', ha='right', transform=ax.transAxes)
ax.text(0.985, 0.16, 'Observable\n(Sink)',    color='#5A6672', fontsize=11.5,
        va='center', ha='right', transform=ax.transAxes)

for src_i, dst_i in _dag_edges:
    sx, sy = _dag_nodes[src_i][1], _dag_nodes[src_i][2]
    dx, dy = _dag_nodes[dst_i][1], _dag_nodes[dst_i][2]
    rad = 0.32 if abs(sy - dy) < 0.05 else 0.0
    w   = abs(float(dag_weights[dst_i, src_i]))
    ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                arrowprops=dict(arrowstyle='-|>', color='#43505C',
                                lw=1.6 + 2.6 * w, mutation_scale=18,
                                connectionstyle=f'arc3,rad={rad}'))
    # Weight label placed near the source end (avoids destination / crossing
    # nodes at the midpoint) and offset perpendicular, with a solid backing box
    t = 0.32
    mx, my = sx + t * (dx - sx), sy + t * (dy - sy)
    ddx, ddy = dx - sx, dy - sy
    norm = (ddx ** 2 + ddy ** 2) ** 0.5 or 1.0
    ox, oy = -ddy / norm * 0.045, ddx / norm * 0.045
    ax.text(mx + ox, my + oy, f'{w:.2f}', fontsize=11, color='#B44D18',
            ha='center', va='center', fontweight='bold', zorder=5,
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='#D9DEE4', alpha=0.95))

for idx, (name, x, y) in _dag_nodes.items():
    marker = 'o' if idx < N_OBSERVABLE else 'D'
    color  = CLASS_COLORS[idx] if idx < N_OBSERVABLE else LATENT_COLORS[idx - N_OBSERVABLE]
    ax.scatter([x], [y], s=6400, color=color, zorder=3, marker=marker,
               edgecolors='black', linewidths=1.4)
    ax.text(x, y, pretty(name).replace(' ', '\n'), ha='center', va='center',
            color='white', fontsize=11, fontweight='bold', zorder=4)

ax.scatter([0.24], [0.035], s=140, color='#8593A0', transform=ax.transAxes)
ax.text(0.27, 0.035, 'Observable Concept', color='black', fontsize=11.5,
        va='center', transform=ax.transAxes)
ax.scatter([0.60], [0.035], s=140, color='#8593A0', marker='D', transform=ax.transAxes)
ax.text(0.63, 0.035, 'Latent Root Cause', color='black', fontsize=11.5,
        va='center', transform=ax.transAxes)
ax.set_title('Learned Causal DAG  (Edge Labels = |Adjacency Weight|)',
             fontsize=16, pad=12)

plt.tight_layout()
plt.savefig('eval_plots/dag_structure.png', bbox_inches='tight')
plt.close(fig)
print('    → eval_plots/dag_structure.png')


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT CLASS DISTRIBUTION (full dataset)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  Concept Class Distribution (full dataset)')
print(SEP)

_all_split_labels = []
for _split in ('train', 'valid', 'test'):
    _ldr = get_dataloader(args.data_root, _split, batch_size=256, num_workers=0)
    for _, _lbl in _ldr:
        _all_split_labels.append(_lbl)
_all_labels_full = torch.cat(_all_split_labels, dim=0).numpy()
_counts_full = _all_labels_full.mean(axis=0)
_n_total = len(_all_labels_full)

fig, ax = plt.subplots(figsize=(10, 5.5))
_colors = CLASS_COLORS + LATENT_COLORS
_pretty_names = [pretty(n) for n in CLASS_NAMES]
_bars = ax.bar(_pretty_names, _counts_full * 100, color=_colors, edgecolor='black', linewidth=0.8)
ax.grid(True, axis='y')
ax.set_ylabel('Share of Images (%)')
ax.set_title('Concept Prevalence Across the Full Dataset', fontsize=16)
ax.set_ylim(0, 112)
ax.set_xticklabels(_pretty_names, rotation=15, ha='right')
for _bar, _val in zip(_bars, _counts_full):
    ax.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 1.5,
            f'{_val*100:.1f}%', ha='center', va='bottom', fontsize=11.5,
            fontweight='bold')
ax.axvline(x=N_OBSERVABLE - 0.5, color='#8593A0', linestyle='--', linewidth=1.2)
ax.text((N_OBSERVABLE - 1) / 2.0, 106, 'Observable', ha='center', fontsize=11.5, color='#5A6672')
ax.text((N_OBSERVABLE + N_CONCEPTS - 1) / 2.0, 106, 'Latent Root Causes',
        ha='center', fontsize=11.5, color='#5A6672')
plt.tight_layout()
plt.savefig('eval_plots/concept_distribution.png')
plt.close(fig)
print('    → eval_plots/concept_distribution.png')


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  FINAL EVALUATION SUMMARY')
print(SEP)
print(f'\n  Top-1 accuracy    : {overall_acc:.3f}')
print(f'  Macro F1          : {metrics["f1_macro"]:.3f}')
print(f'  Macro AUC-ROC     : {macro_auc:.3f}')
print(f'  Mean specificity  : {mean_specificity:.3f}')
print(f'  Causal consistency: {cc_mean:.3f}  (frac>=0.7 {cc_frac_high:.2f}, flagged<0.3 {cc_frac_flag:.2f})')
print(f'  {mic_interp_str}')
print(f'  {tic_interp_str}')
print(f'\n{SEP}\n')

# ── Save JSON ─────────────────────────────────────────────────────────────────
report = {
    'checkpoint': args.checkpoint,
    'split':      args.split,
    'n_images':   len(loader.dataset),
    'detection': {
        'mode':            'single_label_softmax',
        'top1_accuracy':   overall_acc,
        'f1_macro':        metrics['f1_macro'],
        'precision_macro': metrics['precision_macro'],
        'recall_macro':    metrics['recall_macro'],
        'per_class': {
            name: {
                'f1':        metrics.get(f'f1_{name}', 0.0),
                'precision': metrics.get(f'precision_{name}', 0.0),
                'recall':    metrics.get(f'recall_{name}', 0.0),
            }
            for name in OBS_NAMES
        },
    },
    'reconstruction': {'avg_mse': avg_rec},
    'confusion_matrix': confusion_report,
    'auc_roc': {
        'macro':     macro_auc,
        'per_class': auc_per_class,
    },
    'concept_activation': {
        'specificity_per_concept': spec_per_concept,
        'mean_specificity':        mean_specificity,
        'activation_table': {
            CLASS_NAMES[i]: {OBS_NAMES[j]: float(table[i, j])
                             for j in range(_N_OBS)}
            for i in range(Z1_DIM)
        },
    },
    'causal_consistency': {
        'mean_score':   cc_mean,
        'frac_high':    cc_frac_high,
        'frac_flagged': cc_frac_flag,
    },
    'disentanglement': {
        'mic_overall':       overall_mic,
        'mic_per_class':     mic_per_class,
        'mic_interpretation': mic_interp_str,
        'tic_overall':       overall_tic,
        'tic_per_concept':   tic_per_concept,
        'tic_interpretation': tic_interp_str,
        'mi_matrix': {
            CLASS_NAMES[i]: {OBS_NAMES[j]: float(mi_matrix[i, j]) for j in range(_N_OBS)}
            for i in range(Z1_DIM)
        },
    },
}

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, 'w') as f:
    json.dump(report, f, indent=2)
print(f'Full report saved → {args.out}')
