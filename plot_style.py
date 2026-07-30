"""
Shared journal-style matplotlib configuration for all aircraft-damage figures.

Call apply_journal_style() once after importing matplotlib (with the Agg
backend already set) to get consistent, publication-appropriate typography:
larger fonts, bold descriptive titles, clean spines, tabular figures.
"""

import matplotlib as mpl

# Colour-blind-safe qualitative palette (Okabe-Ito derived), plus semantic tints.
CLASS_COLORS  = ['#0072B2', '#E69F00', '#009E73']   # crack, dent, no_damage
LATENT_COLORS = ['#CC79A7', '#56B4E9']              # impact_force, metal_fatigue
GOOD  = '#1B7F5C'
WARN  = '#C77A22'
BAD   = '#B2362C'
INK   = '#1A2530'
GRID  = '#D9DEE4'


def pretty(name: str) -> str:
    """Human-readable concept label: 'structural_crack' -> 'Structural Crack'."""
    return name.replace('_', ' ').title()


def apply_journal_style():
    mpl.rcParams.update({
        'figure.dpi':        120,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'figure.titlesize':  18,
        'figure.titleweight': 'bold',

        'font.family':       'sans-serif',
        'font.sans-serif':   ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size':         13,

        'axes.titlesize':    15,
        'axes.titleweight':  'bold',
        'axes.titlepad':     10,
        'axes.labelsize':    13,
        'axes.labelweight':  'medium',
        'axes.labelcolor':   INK,
        'axes.edgecolor':    '#5A6672',
        'axes.linewidth':    1.0,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'axes.axisbelow':    True,

        'grid.color':        GRID,
        'grid.linewidth':    0.8,
        'grid.alpha':        0.7,

        'xtick.labelsize':   11.5,
        'ytick.labelsize':   11.5,
        'xtick.color':       INK,
        'ytick.color':       INK,

        'legend.fontsize':   11.5,
        'legend.frameon':    False,
        'legend.title_fontsize': 12,

        'axes.formatter.use_mathtext': True,
    })
