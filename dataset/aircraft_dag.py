"""
Aircraft Damage Causal DAG — 3 observable + 2 latent root-cause concepts.

Concept indices:
    0: structural_crack   (observable)
    1: dent               (observable)
    2: no_damage          (observable — root node, no parents, no children)
    3: impact_force       (latent root cause)
    4: metal_fatigue      (latent root cause)

Adjacency convention (this file):
    A_INIT[child, parent] = 1  — row is the CHILD concept, column the PARENT.

Causal edges:
    A_INIT[0, 3] = 1   structural_crack is caused by impact_force
    A_INIT[1, 3] = 1   dent             is caused by impact_force
    A_INIT[0, 4] = 1   structural_crack is caused by metal_fatigue
    A_INIT[0, 1] = 1   structural_crack is caused by dent

no_damage (2) is a root node with no parents and no children.
impact_force (3) and metal_fatigue (4) are root nodes with no parents.
"""

import torch

N_CONCEPTS   = 5
N_OBSERVABLE = 3   # structural_crack, dent, no_damage
N_LATENT     = 2   # impact_force, metal_fatigue

CLASS_NAMES = [
    'structural_crack',  # 0 — observable
    'dent',              # 1 — observable
    'no_damage',         # 2 — observable
    'impact_force',      # 3 — latent root cause
    'metal_fatigue',     # 4 — latent root cause
]


def get_dag_init() -> torch.Tensor:
    """Return the 5×5 adjacency matrix (row=child, col=parent)."""
    A = torch.zeros(N_CONCEPTS, N_CONCEPTS)
    A[0, 3] = 1.0   # structural_crack ← impact_force
    A[1, 3] = 1.0   # dent             ← impact_force
    A[0, 4] = 1.0   # structural_crack ← metal_fatigue
    A[0, 1] = 1.0   # structural_crack ← dent
    return A


# Pre-built constant for import convenience
A_INIT: torch.Tensor = get_dag_init()
