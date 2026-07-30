"""
Aircraft Damage Dataset — CausalVAE (3 observable + 2 latent root-cause concepts)

Image-classification directory structure. Each split directory contains one
sub-folder per observable class, and the class label of an image is determined
entirely by the sub-folder it lives in:

    <root>/<split>/structural_crack/*.jpg
    <root>/<split>/dent/*.jpg
    <root>/<split>/no_damage/*.jpg

Observable concepts (directly supervised by the class folder):
    0: structural_crack
    1: dent
    2: no_damage

Latent root-cause concepts (physics-informed soft labels):
    3: impact_force
    4: metal_fatigue

The images are already cropped to the relevant region by the preprocessing
pipeline, so no further cropping or spatial augmentation is applied. no_damage
is a real labelled class with its own images, not inferred from missing
annotations.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as Data
from PIL import Image
from torchvision import transforms

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 128   # crops are natively 128×128; keep full resolution
N_CONCEPTS   = 5
N_OBSERVABLE = 3
N_LATENT     = 2

CLASS_NAMES = [
    'structural_crack',
    'dent',
    'no_damage',
    'impact_force',
    'metal_fatigue',
]

OBS_NAMES    = CLASS_NAMES[:N_OBSERVABLE]
LATENT_NAMES = CLASS_NAMES[N_OBSERVABLE:]

# scale[j] = [mean, half_range] used by the label-conditioned prior:
#   normalised = (label - mean) / half_range
SCALE = np.array([
    [0.5, 0.5],   # structural_crack  binary {0,1} maps to {-1,+1}
    [0.5, 0.5],   # dent
    [0.5, 0.5],   # no_damage
    [0.0, 1.0],   # impact_force      continuous [0,1] centred at 0
    [0.0, 1.0],   # metal_fatigue
], dtype=np.float32)

# Class folder → observable concept presence vector (structural_crack, dent, no_damage)
FOLDER_TO_OBS = {
    'structural_crack': np.array([1, 0, 0], dtype=np.float32),
    'dent':             np.array([0, 1, 0], dtype=np.float32),
    'no_damage':        np.array([0, 0, 1], dtype=np.float32),
}


# ── Physics-informed latent labels ────────────────────────────────────────────

def infer_latent_labels(p_obs: np.ndarray) -> np.ndarray:
    """
    Derives physics-informed soft labels for impact_force and metal_fatigue
    from the observable concept presence vector.

    Args:
        p_obs: (3,) binary array [structural_crack, dent, no_damage]

    Returns:
        (2,) float array [impact_force, metal_fatigue] in [0, 1]

    Physical justification:

    impact_force:
        A dent is permanent plastic deformation requiring applied force
        exceeding the material yield strength. No other failure mechanism
        produces a dent, so dent presence is the primary impact signal
        (weight 0.85). Crack and dent co-occurring is the signature of a
        high-energy impact that both deformed and fractured the surface,
        strengthening the impact attribution (additional weight 0.40).
        A crack without a visible dent carries a small residual impact
        contribution (0.15) because subsurface or oblique impacts can
        initiate cracks without producing macroscopic surface deformation.

    metal_fatigue:
        A structural crack appearing without surface deformation lacks an
        obvious single-event cause. In aviation structural analysis this
        is the textbook presentation of fatigue crack initiation: cyclic
        loading below yield strength accumulates damage at stress
        concentrations until a crack propagates to the surface. The
        primary signal is crack without dent (weight 0.90). A crack
        alongside a dent still carries a residual fatigue contribution
        (0.30) because impact-initiated cracks frequently propagate
        further by fatigue cycling, meaning both mechanisms may have
        contributed to the observed damage state.

    The residual cross-terms are deliberate. Without them the latent
    labels are near-deterministic mirrors of the observable classes,
    which causes the latent concept sub-vectors to collapse into
    redundant re-encodings of crack and dent rather than learning
    distinct causal representations.
    """
    p_crack = float(p_obs[0])
    p_dent  = float(p_obs[1])

    u_impact = min(
        p_dent  * 0.85
        + p_crack * p_dent       * 0.40
        + p_crack * (1 - p_dent) * 0.15,
        1.0
    )
    u_fatigue = min(
        p_crack * (1 - p_dent) * 0.90
        + p_crack * p_dent     * 0.30,
        1.0
    )

    return np.array([u_impact, u_fatigue], dtype=np.float32)


# ── Image-morphology latent labels ────────────────────────────────────────────
# Instead of deriving impact_force / metal_fatigue deterministically from the
# class label (which carries no information the class label does not already
# have), ground them in observable image morphology — measured automatically
# from the pixels, so they vary WITHIN a class and inject independent signal:
#
#   metal_fatigue  ← aligned, high-frequency edges (straight cracks along a
#                    stress direction): edge energy × orientation coherence
#   impact_force   ← isotropic edges + smooth deformation (dents / radial
#                    impact): isotropic edge energy + low-frequency shading
#
# All features are unsupervised and cached once to <data_root>/morph_labels.json.
MORPH_CACHE_NAME = 'morph_labels.json'
_MORPH_CACHE = {}


def _raw_morph_features(img: Image.Image):
    """Cheap grayscale descriptors: (edge energy, orientation coherence, shading)."""
    g = np.asarray(img.convert('L').resize((64, 64)), dtype=np.float32) / 255.0
    gx, gy = np.gradient(g)
    edge = float(np.hypot(gx, gy).mean())                      # high-frequency energy
    Jxx = float((gx * gx).mean()); Jyy = float((gy * gy).mean()); Jxy = float((gx * gy).mean())
    tr = Jxx + Jyy; det = Jxx * Jyy - Jxy * Jxy
    disc = max(tr * tr / 4.0 - det, 0.0) ** 0.5
    l1, l2 = tr / 2.0 + disc, tr / 2.0 - disc
    coh = float((l1 - l2) / (l1 + l2 + 1e-6))                  # edge-orientation coherence ∈ [0,1]
    shade = float((np.asarray(img.convert('L').resize((8, 8)),
                              dtype=np.float32) / 255.0).std())  # low-frequency deformation
    return edge, coh, shade


def _build_morph_labels(data_root: str) -> dict:
    """Scan every crop, normalise by TRAIN percentiles, map to [impact, fatigue]."""
    root = Path(data_root)
    raw = {}
    for split in ('train', 'valid', 'test'):
        for folder in OBS_NAMES:
            d = root / split / folder
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if p.suffix not in AircraftDamageDataset._EXTS:
                    continue
                try:
                    with Image.open(p) as im:
                        e, c, s = _raw_morph_features(im)
                except Exception:
                    continue
                raw[p.name] = (e, c, s, split)

    tr_edge  = [v[0] for v in raw.values() if v[3] == 'train'] or [0.0, 1.0]
    tr_shade = [v[2] for v in raw.values() if v[3] == 'train'] or [0.0, 1.0]
    e_lo, e_hi = np.percentile(tr_edge, [5, 95])
    s_lo, s_hi = np.percentile(tr_shade, [5, 95])

    labels = {}
    for name, (e, c, s, _) in raw.items():
        en = float(np.clip((e - e_lo) / (e_hi - e_lo + 1e-6), 0.0, 1.0))
        sn = float(np.clip((s - s_lo) / (s_hi - s_lo + 1e-6), 0.0, 1.0))
        metal_fatigue = float(np.clip(en * c, 0.0, 1.0))
        impact_force  = float(np.clip(0.5 * en * (1.0 - c) + 0.5 * sn, 0.0, 1.0))
        labels[name] = [impact_force, metal_fatigue]

    with open(root / MORPH_CACHE_NAME, 'w') as f:
        json.dump(labels, f)
    return labels


def load_morph_labels(data_root: str) -> dict:
    """Return {filename: [impact_force, metal_fatigue]}, building/caching once."""
    root = str(data_root)
    if root in _MORPH_CACHE:
        return _MORPH_CACHE[root]
    path = Path(data_root) / MORPH_CACHE_NAME
    if path.exists():
        with open(path) as f:
            labels = json.load(f)
    else:
        print('  Building image-morphology latent labels (one-time scan)…')
        labels = _build_morph_labels(data_root)
    _MORPH_CACHE[root] = labels
    return labels


# ── Transform ─────────────────────────────────────────────────────────────────
# Applied identically to all splits — the images are already cropped and are
# presented to the model as-is (no augmentation).
_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class AircraftDamageDataset(Data.Dataset):
    """
    Folder-based image classification dataset.

    Returns per sample:
        image_tensor: (3, IMAGE_SIZE, IMAGE_SIZE) float32   normalised RGB
        u_tensor:     (5,) float32  = [p_obs (3) | u_latent (2)]
    """
    _EXTS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    def __init__(self, data_root: str, split: str = 'train'):
        self.data_root = data_root
        self.split     = split
        self.transform = _TRANSFORM
        self.morph_labels = load_morph_labels(data_root)

        split_dir = Path(data_root) / split

        # Flat list of (image_path, u) where u = [p_obs (3) | u_latent (2)]
        self.samples = []
        self.class_counts = {name: 0 for name in OBS_NAMES}
        self.class_latent = {name: [] for name in OBS_NAMES}

        for folder in OBS_NAMES:
            cls_dir = split_dir / folder
            if not cls_dir.is_dir():
                print(f'  [WARN] missing class folder: {cls_dir}')
                continue

            p_obs = FOLDER_TO_OBS[folder]

            for img_path in sorted(p for p in cls_dir.iterdir()
                                   if p.suffix in self._EXTS):
                try:
                    # Cheap integrity check without decoding the full image.
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception as exc:
                    print(f'  [WARN] skipping unreadable image {img_path}: {exc}')
                    continue
                # Latent root-cause labels from image morphology (fall back to the
                # class-based proxy only if the crop is missing from the cache).
                m = self.morph_labels.get(img_path.name)
                u_latent = (np.asarray(m, dtype=np.float32) if m is not None
                            else infer_latent_labels(p_obs))
                u = np.concatenate([p_obs, u_latent]).astype(np.float32)  # (5,)
                self.samples.append((str(img_path), u))
                self.class_counts[folder] += 1
                self.class_latent[folder].append(u_latent)

        self._print_distribution()

    def _print_distribution(self):
        total = len(self.samples)
        print(f'[AircraftDamageDataset] split={self.split!r}  total={total}')
        for name in OBS_NAMES:
            c   = self.class_counts[name]
            pct = 100.0 * c / total if total else 0.0
            lat = np.array(self.class_latent[name]) if self.class_latent[name] else np.zeros((1, 2))
            print(f'    {name:<18} {c:6d}  ({pct:5.1f}%)   '
                  f'impact {lat[:, 0].mean():.2f}  fatigue {lat[:, 1].mean():.2f}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, u = self.samples[idx]
        img          = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(img)
        u_tensor     = torch.from_numpy(u)
        return image_tensor, u_tensor


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloader(data_root, split, batch_size, num_workers=0, shuffle=None):
    """
    Build a DataLoader over the given split of the dataset_final/ directory.

    Args:
        data_root:   path to dataset_final/
        split:       'train', 'valid', or 'test'
        batch_size:  int
        num_workers: int
        shuffle:     defaults to True for 'train', False otherwise.
    """
    ds = AircraftDamageDataset(data_root, split)
    if shuffle is None:
        shuffle = (split == 'train')
    return Data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=(split == 'train'),
        pin_memory=torch.cuda.is_available(),
    )
