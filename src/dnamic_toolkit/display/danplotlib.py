"""
Selection of functions to make dan-style plots.
"""

from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import pandas as pd
import numpy as np
import colorsys
from math import floor, log10
import dnamic_toolkit.display.tol_colors as tol_colors
# import math
# from .to_precision import to_precision
# from .scienceplots import *
from .styles import *

base_ebar_params = {'marker' : 'o',
                    'linestyle' : '',
                    'linewidth' : 1, 
                    'capsize':1}

def adjust_lightness(color, amount=0.5):
    """Adjust the lightness of a given color."""
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    rgb = mc.to_rgb(c)
    hls = colorsys.rgb_to_hls(*rgb)
    adjusted_rgb = colorsys.hls_to_rgb(hls[0], max(0, min(1, amount * hls[1])), hls[2])
    return adjusted_rgb

def dscatter(x, y, *params, xerr=None, yerr=None, ax=None, color=None, show=False,
             outline_brightness=0.8, empty=False, create_csv_name=None, **ebar_params):
    """Create a scatter plot with optional error bars and customizable appearance."""
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = 'red' # next(ax._get_lines.prop_cycler)['color']
    
    markeredgecolor = adjust_lightness(color, outline_brightness)
    markercolor = adjust_lightness(color, 2 - outline_brightness)
    
    ebar_params.setdefault('marker', 'o')
    
    ebar_params.setdefault('markeredgecolor', markeredgecolor)
    ebar_params.setdefault('ecolor', markeredgecolor)
    ebar_params['color'] = markercolor

    if empty:
        ebar_params['markeredgecolor'] = markeredgecolor
        ebar_params['ecolor'] = markeredgecolor
        ebar_params['color'] = 'w'

    ax.errorbar(x, y, *params, xerr=xerr, yerr=yerr, **ebar_params)

    if create_csv_name is not None:
        data = {
            'x': x,
            'y': y
        }
        if xerr is not None:
            shape_x = np.asarray(xerr).shape
            if len(shape_x) == 2:
                data['xerr_upper'] = xerr[0]
                data['xerr_lower'] = xerr[1]
            else:
                data['xerr'] = xerr
        if yerr is not None:
            shape_y = np.asarray(yerr).shape
            if len(shape_y) == 2:
                data['xerr_upper'] = yerr[0]
                data['xerr_lower'] = yerr[1]
            else:
                data['xerr'] = yerr
        df = pd.DataFrame(data)
        if not create_csv_name.lower().endswith('.csv'):
            create_csv_name += '.csv'
        df.to_csv(create_csv_name, index=False)
    
    if show:
        plt.show()
        
def uncert_to_str(val,err):
    prec = floor(log10(err))
    err = round(err/10**prec)*10**prec
    val = round(val/10**prec)*10**prec
    if prec > 0:
        valerr = '{:.0f}({:.0f})'.format(val,err)
    else:
        valerr = '{:.{prec}f}({:.0f})'.format(val,err*10**-prec,prec=-prec)
    return valerr

def set_plot_style(style=None):
    global base_ebar_params
    plt.style.use('default')
    if style == None:    
        default_cycler = (cycler(color=list(tol_colors.tol_cset('bright'))))
        default_cycler = (cycler(color=[colors['durham_purple'],
                                        colors['durham_cyan'],
                                        colors['durham_red'],
                                        colors['durham_yellow'],
                                        colors['durham_gold']]))
        default_cycler = (cycler(color=list(['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33'])))
        #red blue green
    
        plt.rc('axes', prop_cycle=default_cycler)
    
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['mathtext.fontset'] = 'cm'
        # plt.rcParams['mathtext.fontset']: 'stixsans'
        plt.rcParams['mathtext.rm'] = 'serif'
        plt.rcParams['font.serif'] = 'cmr10'
        plt.rcParams['font.sans-serif'] = "cmss10"
        # plt.rc('axes', unicode_minus=False)
        plt.rcParams['font.size'] = 10.95
        
        plt.rcParams['axes.unicode_minus'] = False 
    
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['ytick.right'] = 'True'
        plt.rcParams['xtick.top'] = 'True'
    
        plt.rcParams['figure.figsize'] = [5.895, 3]
        # plt.rcParams["figure.labelsize"] = 'medium'
    
        plt.rcParams["legend.edgecolor"] = 'k'
        plt.rcParams["legend.fancybox"] = False
        
    elif style == 'paper':
        plt.style.use(['dan.styles.tweezer_paper'])
        
        # plt.rcParams['text.usetex'] = False
        # plt.rcParams['svg.fonttype'] = 'none' # 'none': Assume fonts are installed on the machine where the SVG will be viewed.
        
        # plt.rcParams['font.family'] = 'Nimbus Sans L'
        # plt.rcParams['mathtext.fontset'] = 'Nimbus Sans L'
        
        # plt.rcParams['axes.formatter.useoffset'] = False
        
        # plt.rcParams['savefig.bbox'] = None
        # plt.rcParams['savefig.pad_inches'] = 0
        
        base_ebar_params = {'marker' : 'o',
                            'linestyle' : '',
                            'linewidth' : 1, 
                            'capsize':0}
    
#%% define durham colours

colors = {}
colors['durham_purple'] = '#68246D'
colors['durham_yellow'] = '#FFD53A'
colors['durham_cyan'] = '#00AEEF'
colors['durham_red'] = '#BE1E2D'
colors['durham_gold'] = '#AFA961'
colors['durham_heather'] = '#CBA8B1'
colors['durham_stone'] = '#DACDA2'
colors['durham_sky'] = '#A5C8D0'
colors['durham_cedar'] = '#B6AAA7'
colors['durham_concrete'] = '#B3BDB1'

set_plot_style()

if __name__ == '__main__':
    import numpy as np
    
    x = np.linspace(0,1,11)
    y = np.linspace(0,20,11)
    yerr = np.random.random(11)
    
    fig, ax = plt.subplots()
    dscatter(x,y,yerr=yerr,ax=ax, marker='s')
    dscatter(x,2*y,yerr=yerr,ax=ax)
    plt.show()