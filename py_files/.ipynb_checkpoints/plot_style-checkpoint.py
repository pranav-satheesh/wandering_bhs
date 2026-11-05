"""Shared matplotlib styling for the wandering-BH figures.

The point of this module is that every figure in the project can pull the same
typography, tick and legend settings from one place, while still letting a
single plot override whatever it needs:

    import plot_style as ps

    ps.set_style()                          # project defaults
    ps.set_style('talk', font_scale=1.2)    # bigger everything for slides
    ps.set_style(grid=True, tick_direction='out',
                 **{'axes.labelweight': 'bold'})   # any rcParam by name

    ps.set_style(color_cycle=ps.OKABE_ITO,  # colorblind-safe categorical
                 linestyle_cycle=True)      # + dashes, so not color-alone

    with ps.style_context('poster'):        # temporary, restored on exit
        ...

Every knob below is a keyword argument, and anything not covered by a keyword
can be passed as a raw rcParam through **overrides, so nothing is locked in.
"""
import contextlib

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager


# Preferred text faces, best first. The first one actually installed wins, so
# this degrades gracefully on machines without the nicer fonts.
SERIF_STACK = ['STIXGeneral', 'Times New Roman', 'Nimbus Roman',
               'Liberation Serif', 'DejaVu Serif']
SANS_STACK = ['Helvetica', 'Arial', 'Nimbus Sans', 'Liberation Sans',
              'DejaVu Sans']

# Okabe-Ito: the standard colorblind-safe *categorical* palette, for series
# that are unordered (simulation runs, seed models, ...). Use it in fixed
# order; do not reorder or cycle past the eighth entry.
OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#CC79A7',
             '#D55E00', '#56B4E9', '#F0E442', '#000000']

# Dash patterns, most legible first. These carry series identity alongside
# colour, so a figure still reads in greyscale or with any colour deficiency.
LINESTYLES = [
    'solid',
    (0, (5.5, 1.6)),                    # dashed
    (0, (1.1, 1.3)),                    # dotted
    (0, (6.5, 1.6, 1.1, 1.6)),          # dash-dot
    (0, (3.2, 1.4, 1.1, 1.4, 1.1, 1.4)),  # dash-dot-dot
    (0, (9.0, 2.2)),                    # long dash
    (0, (2.4, 1.3)),                    # short dash
    (0, (1.1, 1.3, 4.5, 1.3)),          # dot-dash
]

# Size of each text element relative to the base font size, so a single
# font_scale (or preset) moves the whole hierarchy together.
SIZE_RATIOS = {
    'font.size': 1.00,
    'axes.labelsize': 1.15,
    'axes.titlesize': 1.25,
    'xtick.labelsize': 0.95,
    'ytick.labelsize': 0.95,
    'legend.fontsize': 0.92,
    'legend.title_fontsize': 0.95,
    'figure.titlesize': 1.30,
}

# Named presets: (base font size in points, base line width in points).
PRESETS = {
    'notebook': (12.0, 1.8),
    'paper': (11.0, 1.6),   # single-column journal figure
    'talk': (16.0, 2.4),
    'poster': (22.0, 3.2),
}


def available_font(candidates):
    """First font in `candidates` that matplotlib can actually find."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return name
    return None


def style_params(preset='paper', font_scale=1.0, font_family='serif',
                 font=None, math_fontset=None, usetex=False,
                 line_width=None, color_cycle=None, linestyle_cycle=None,
                 tick_direction='in',
                 minor_ticks=True, grid=False, grid_alpha=0.25,
                 legend_frame=False, savefig_dpi=300, figure_dpi=120,
                 transparent=False, **overrides):
    """Build (but do not apply) the rcParam dict for this project's figures.

    Parameters
    ----------
    preset : str
        One of PRESETS; sets the base font size and line width.
    font_scale : float
        Multiplies every text size on top of the preset.
    font_family : {'serif', 'sans-serif'}
        Which stack to draw the text face from.
    font : str or list of str, optional
        Explicit font name(s), overriding the stack for `font_family`.
    math_fontset : str, optional
        mathtext fontset. Defaults to the STIX face matching `font_family`
        ('stix' / 'stixsans') when those fonts are installed, else the
        DejaVu equivalent.
    usetex : bool
        Render text with a real LaTeX installation. Off by default because the
        cluster nodes have no `latex` binary; mathtext handles $...$ fine.
    line_width : float, optional
        Data line width; defaults to the preset's value.
    color_cycle : sequence of colors, optional
        Replaces the default color property cycle. Pass OKABE_ITO for a
        colorblind-safe categorical set, or sequential_colors(n) for an
        ordered one.
    linestyle_cycle : sequence of linestyles or True, optional
        Cycle dash patterns alongside the colors, so series identity does not
        rest on color alone. ``True`` uses LINESTYLES.
    tick_direction : {'in', 'out', 'inout'}
    minor_ticks : bool
        Show minor ticks, and draw ticks on all four sides.
    grid, grid_alpha : bool, float
    legend_frame : bool
        Draw the legend box. Off by default so it does not compete with data.
    savefig_dpi, figure_dpi : float
    transparent : bool
        Save figures with a transparent background.
    **overrides
        Any rcParam by name, e.g. ``**{'axes.labelweight': 'bold'}``. Applied
        last, so it wins over everything above.

    Returns
    -------
    dict
        Ready to hand to ``plt.rcParams.update`` or ``plt.rc_context``.
    """
    if preset not in PRESETS:
        raise ValueError(f'preset must be one of {sorted(PRESETS)}, '
                         f'got {preset!r}')
    base_size, base_line_width = PRESETS[preset]
    if line_width is None:
        line_width = base_line_width

    if font is not None:
        family_fonts = [font] if isinstance(font, str) else list(font)
    else:
        family_fonts = (SERIF_STACK if font_family == 'serif' else SANS_STACK)

    if math_fontset is None:
        # Match the math face to the text face so $...$ does not stand out.
        have_stix = available_font(['STIXGeneral']) is not None
        if font_family == 'serif':
            math_fontset = 'stix' if have_stix else 'dejavuserif'
        else:
            math_fontset = 'stixsans' if have_stix else 'dejavusans'

    params = {
        # --- typography -------------------------------------------------
        'font.family': font_family,
        'font.serif': SERIF_STACK if font is None else family_fonts,
        'font.sans-serif': SANS_STACK if font is None else family_fonts,
        'mathtext.fontset': math_fontset,
        'mathtext.default': 'it',   # variables italic, \rm for upright labels
        'text.usetex': usetex,
        'axes.titleweight': 'normal',
        'axes.titlepad': 10.0,
        'axes.labelpad': 6.0,

        # --- axes and frame ---------------------------------------------
        'axes.linewidth': 0.9,
        'axes.edgecolor': '0.25',
        'axes.labelcolor': '0.15',
        'axes.axisbelow': True,
        'axes.grid': grid,
        'grid.color': '0.7',
        'grid.linewidth': 0.6,
        'grid.alpha': grid_alpha,

        # --- ticks ------------------------------------------------------
        'xtick.direction': tick_direction,
        'ytick.direction': tick_direction,
        'xtick.top': minor_ticks,
        'ytick.right': minor_ticks,
        'xtick.minor.visible': minor_ticks,
        'ytick.minor.visible': minor_ticks,
        'xtick.major.size': 6.0,
        'ytick.major.size': 6.0,
        'xtick.minor.size': 3.0,
        'ytick.minor.size': 3.0,
        'xtick.major.width': 0.9,
        'ytick.major.width': 0.9,
        'xtick.minor.width': 0.7,
        'ytick.minor.width': 0.7,
        'xtick.color': '0.25',
        'ytick.color': '0.25',
        'xtick.labelcolor': '0.15',
        'ytick.labelcolor': '0.15',

        # --- data marks -------------------------------------------------
        'lines.linewidth': line_width,
        'lines.markersize': 5.0,
        'lines.solid_capstyle': 'round',
        'patch.linewidth': 0.0,

        # --- legend -----------------------------------------------------
        'legend.frameon': legend_frame,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        'legend.borderpad': 0.4,
        'legend.handlelength': 1.8,
        'legend.labelspacing': 0.35,
        'legend.columnspacing': 1.2,

        # --- output -----------------------------------------------------
        'figure.dpi': figure_dpi,
        'figure.facecolor': 'white',
        'savefig.dpi': savefig_dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.transparent': transparent,
        'pdf.fonttype': 42,   # embed real TrueType, not Type-3 outlines
        'ps.fonttype': 42,
    }

    for key, ratio in SIZE_RATIOS.items():
        params[key] = base_size * font_scale * ratio

    if color_cycle is not None or linestyle_cycle is not None:
        if color_cycle is not None:
            colors = list(color_cycle)
        else:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        cycle = plt.cycler(color=colors)
        if linestyle_cycle is not None:
            styles = (LINESTYLES if linestyle_cycle is True
                      else list(linestyle_cycle))
            # cycler addition needs equal lengths, so repeat/trim to match.
            repeats = -(-len(colors) // len(styles))
            cycle = cycle + plt.cycler(
                linestyle=(styles * repeats)[:len(colors)])
        params['axes.prop_cycle'] = cycle

    params.update(overrides)
    return params


def set_style(preset='paper', **kwargs):
    """Apply the project style globally. Returns the rcParams that were set."""
    params = style_params(preset, **kwargs)
    plt.rcParams.update(params)
    return params


@contextlib.contextmanager
def style_context(preset='paper', **kwargs):
    """Apply the style only inside a ``with`` block."""
    with plt.rc_context(style_params(preset, **kwargs)):
        yield


def reset_style():
    """Back to stock matplotlib."""
    mpl.rcdefaults()


def sequential_colors(n, cmap='viridis', low=0.05, high=0.70):
    """`n` colors sampled along a colormap, for an *ordered* series set.

    Redshift is a magnitude, not a set of unrelated categories, so the lines
    read better on a dark-to-light ramp than on a categorical cycle. The
    default map is perceptually uniform and safe for red-green color
    deficiency; `cmap='cividis'` is the stricter choice (built so that
    deuteranopes see very nearly the same figure, and monotonic in greyscale),
    at the cost of a duller mid-range. The default range trims the pale end so
    every line keeps contrast against a white background.
    """
    return plt.get_cmap(cmap)(np.linspace(low, high, n))


def line_styles(n):
    """`n` distinct dash patterns from LINESTYLES, repeating if n is large."""
    repeats = -(-n // len(LINESTYLES))
    return (LINESTYLES * repeats)[:n]


def series_cycle(n, cmap='viridis', low=0.05, high=0.70, linestyles=True):
    """A color(+dash) cycler for `n` ordered series.

    Hand it to ``ax.set_prop_cycle(...)``, or to ``set_style`` as
    ``**{'axes.prop_cycle': ps.series_cycle(6)}``.
    """
    cycle = plt.cycler(color=list(sequential_colors(n, cmap, low, high)))
    if linestyles:
        cycle = cycle + plt.cycler(linestyle=line_styles(n))
    return cycle
