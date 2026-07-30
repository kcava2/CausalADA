"""
Classification heads for aircraft damage CausalVAE.

MultiLabelHead          — flat MLP over the full latent (kept for backward compat)
ConceptAwareClassifier  — per-concept sub-vector encoders fused into logits
ConceptHeads            — one linear head per observable concept sub-vector
FocalBCELoss            — focal binary cross-entropy for imbalanced classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

from dataset.aircraft_damage import OBS_NAMES


class MultiLabelHead(nn.Module):
    """
    MLP classification head for 4 binary damage concepts.

    Args:
        z_dim:      int  input dimension (total latent dim, e.g. 32)
        n_classes:  int  number of output classes (default 4)
        hidden_dim: int  hidden layer width
    """
    def __init__(self, z_dim: int, n_classes: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), -1))


class ConceptAwareClassifier(nn.Module):
    """
    Classification head that processes each observable concept's own
    latent sub-vector through a dedicated encoder before fusing across
    concepts to produce logits.

    A flat MLP receiving the full z_dag vector must infer which latent
    dimensions correspond to which concept from gradient signal alone.
    ConceptAwareClassifier routes each concept's Z2_DIM-dimensional
    sub-vector through its own small network before fusion, directly
    reflecting the causal structure of the latent space.

    Args:
        z2_dim:       int, features per concept sub-vector
        n_observable: int, number of observable concepts to classify
        hidden:       int, hidden dimension for concept encoders
    """
    def __init__(self, z2_dim=4, n_observable=3, hidden=64):
        super().__init__()
        self.n_observable = n_observable
        self.z2_dim       = z2_dim

        self.concept_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(z2_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden // 2),
            )
            for _ in range(n_observable)
        ])

        self.fusion = nn.Sequential(
            nn.Linear(n_observable * hidden // 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_observable),
        )

    def forward(self, z_dag):
        """
        Args:
            z_dag: (B, Z1_DIM, Z2_DIM) or (B, Z_DIM) flat tensor.
                   If flat, reshapes assuming z2_dim=self.z2_dim.

        Returns:
            logits: (B, n_observable)
        """
        if z_dag.dim() == 2:
            z_dag = z_dag.reshape(z_dag.size(0), -1, self.z2_dim)

        feats = [self.concept_encoders[i](z_dag[:, i, :])
                 for i in range(self.n_observable)]
        return self.fusion(torch.cat(feats, dim=-1))


class HierarchicalConceptClassifier(nn.Module):
    """
    Two-stage single-label head that gives no_damage a POSITIVE decision.

    Softmax over three classes forces no_damage to be the residual (it only
    wins when both crack and dent logits are low), so it never learns a
    positive "healthy surface" feature. This head splits the decision:

        stage 1 (binary): damaged vs healthy, from ALL observable sub-vectors
        stage 2 (type):   crack vs dent, from the crack + dent sub-vectors

    and recombines them into calibrated 3-class probabilities:

        P(crack)     = P(damaged) · P(crack | damaged)
        P(dent)      = P(damaged) · P(dent  | damaged)
        P(no_damage) = P(healthy) = 1 − P(damaged)

    forward() returns the 3-class LOG-probabilities in [crack, dent, no_damage]
    order. Because they already normalise to 1, a downstream softmax recovers
    the probabilities exactly and cross_entropy(logprobs, target) is the exact
    hierarchical NLL — so training, argmax and probability read-out need no
    special handling.

    A safety knob, `damage_margin`, is added to the binary (damaged) logit. A
    positive margin biases the decision toward "damaged", so the model errs on
    the side of flagging damage — the safe failure mode for inspection (a false
    alarm is cheap, a missed crack is not). It can be set at train time and/or
    tuned at inference without retraining.

    Args:
        z2_dim:        features per concept sub-vector
        n_observable:  number of observable classes (assumes the last one is the
                       "healthy" / no_damage class)
        hidden:        hidden width
        damage_margin: logit added to the damaged score (>0 errs toward damage)
    """
    def __init__(self, z2_dim=4, n_observable=3, hidden=64, damage_margin=0.0):
        super().__init__()
        self.z2_dim       = z2_dim
        self.n_observable = n_observable
        self.n_damage     = n_observable - 1   # damage classes = all but healthy
        feat_dim          = hidden // 2
        self.register_buffer('damage_margin', torch.tensor(float(damage_margin)))

        self.concept_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(z2_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, feat_dim),
            )
            for _ in range(n_observable)
        ])

        # Stage 1: damaged vs healthy, from every observable concept feature.
        self.binary_head = nn.Sequential(
            nn.Linear(n_observable * feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )
        # Stage 2: which damage type, from the damage-concept features only.
        self.type_head = nn.Sequential(
            nn.Linear(self.n_damage * feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, self.n_damage),
        )

    def forward(self, z_dag, return_binary=False):
        if z_dag.dim() == 2:
            z_dag = z_dag.reshape(z_dag.size(0), -1, self.z2_dim)

        feats = [self.concept_encoders[i](z_dag[:, i, :])
                 for i in range(self.n_observable)]
        all_feats  = torch.cat(feats, dim=-1)
        dmg_feats  = torch.cat(feats[:self.n_damage], dim=-1)

        b_raw      = self.binary_head(all_feats)           # (B, 1) damaged logit
        b          = b_raw + self.damage_margin            # safety-biased logit
        log_dmg    = F.logsigmoid(b)                        # log P(damaged)
        log_healthy = F.logsigmoid(-b)                      # log P(healthy)
        type_lp    = F.log_softmax(self.type_head(dmg_feats), dim=1)  # (B, n_damage)

        # [damage classes …, healthy]  as log-probabilities
        log_probs = torch.cat([log_dmg + type_lp, log_healthy], dim=1)
        if return_binary:
            return log_probs, b_raw          # raw logit (pre-margin) for BCE loss
        return log_probs


class ConceptHeads(nn.Module):
    """One linear head per observable concept, reading only its z sub-vector."""
    def __init__(self, z2_dim: int, n_observable: int):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(z2_dim, 1) for _ in range(n_observable)
        ])

    def forward(self, z_dag: torch.Tensor) -> torch.Tensor:
        # z_dag: (B, Z1_DIM, Z2_DIM) — use only first n_observable sub-vectors
        logits = [self.heads[i](z_dag[:, i, :]) for i in range(len(self.heads))]
        return torch.cat(logits, dim=-1)  # (B, n_observable)


# ── Loss functions ─────────────────────────────────────────────────────────────

def multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor = None,
) -> torch.Tensor:
    """Binary cross-entropy for multi-label concept classification."""
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight
    )


class FocalBCELoss(nn.Module):
    """
    Focal binary cross-entropy for imbalanced multi-label classification.

    Down-weights easy negatives and focuses gradient on hard examples,
    improving performance on minority classes without architectural
    changes. Standard gamma=2.0 from the original focal loss paper.

    Args:
        gamma:      float, focusing parameter
        pos_weight: tensor or None, per-class positive weights for
                    additional imbalance correction
    """
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce     = F.binary_cross_entropy_with_logits(
                      logits, targets,
                      pos_weight=self.pos_weight,
                      reduction='none')
        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        weights = (1 - p_t) ** self.gamma
        return (weights * bce).mean()


def single_label_loss(logits, targets, weight=None):
    """
    Cross-entropy for single-label (mutually-exclusive) classification.

    The three observable classes — structural_crack / dent / no_damage —
    are mutually exclusive: every image has exactly one. Softmax
    cross-entropy models that constraint directly, unlike independent
    per-class sigmoids which allow contradictory multi-hot outputs.

    Args:
        logits:  (N, n_classes) raw logits
        targets: (N, n_classes) one-hot OR (N,) integer class indices
        weight:  optional (n_classes,) per-class weights for imbalance
    """
    if targets.dim() == 2:
        targets = targets.argmax(dim=1)
    return F.cross_entropy(logits, targets.long(), weight=weight)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: list = OBS_NAMES,
) -> dict:
    """
    Single-label (argmax) accuracy, plus per-class and macro F1, precision,
    and recall for the mutually-exclusive damage classes.

    Args:
        logits:     (N, n_classes)  raw softmax logits
        targets:    (N, n_classes) one-hot OR (N,) integer class indices
        class_names: list[str]      for named keys in output dict

    Returns:
        dict with accuracy, f1_macro, precision_macro, recall_macro,
        and per-class scores.
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1).cpu().numpy()
        tgts  = targets.cpu().numpy()
        y     = tgts.argmax(axis=1) if tgts.ndim == 2 else tgts.astype(int)

    n_classes     = logits.shape[1]
    default_names = class_names or [f'class_{i}' for i in range(n_classes)]
    labels_range  = list(range(n_classes))

    f1_per   = f1_score(y, preds, average=None, labels=labels_range, zero_division=0)
    prec_per = precision_score(y, preds, average=None, labels=labels_range, zero_division=0)
    rec_per  = recall_score(y, preds, average=None, labels=labels_range, zero_division=0)

    result = {
        'accuracy':        float(np.mean(preds == y)),
        'f1_macro':        float(np.mean(f1_per)),
        'precision_macro': float(np.mean(prec_per)),
        'recall_macro':    float(np.mean(rec_per)),
        'f1_per_class':    f1_per.tolist(),
    }
    for i, name in enumerate(default_names):
        result[f'f1_{name}']        = float(f1_per[i])
        result[f'precision_{name}'] = float(prec_per[i])
        result[f'recall_{name}']    = float(rec_per[i])

    return result
