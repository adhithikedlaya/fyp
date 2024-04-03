"""
Plotting parameters and functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_para():
    """Parameters for plots
    """
    plt.rcParams.update(plt.rcParamsDefault)
    params = {
        'figure.figsize': [12, 8],
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'axes.titlesize': 25,
        'figure.dpi': 300,
        'figure.subplot.wspace': 0.3,
        'figure.subplot.hspace': 0.3
        }
    plt.rcParams.update(params)


def plot_all(t, var_all, indicator, method):
    """Plot stimulus, all variables, and BOLD responses on one plot to
       reproduce Fig. 3 (C) or 4 (C) in Havlicek et. al.'s paper

    Args:
    - t -- list of time values
    - var_all -- contains lists of stimulus, all variables, and BOLD responses
    - indicator -- controls the plot title to be "Coupled CBF-CBV" or
                   "Uncoupled CBF-CBV"
    - method -- string of the name of the numerical method
    """
    plot_para()

    u, xE, xI, a, f, fout, v, q, E, y = var_all

    fig, host = plt.subplots()
    ax2 = host.twinx()

    host.plot(t, u, ".", color="grey")
    host.plot(t, xE, ".", color="C0")
    host.plot(t, xI, "-", color="C5")
    host.plot(t, a, "-", color="C4")
    host.plot(t, f, ".", color="C2")
    host.plot(t, fout, "-", color="C8")
    host.plot(t, v, ".", color="C1")
    host.plot(t, q, "-", color="C6")
    host.plot(t, E, "-", color="C9")

    ax2.plot(t, y, "o", color="C3")

    # Create manual symbols for legend
    u_leg = Line2D([0], [0], label='Stimulus', marker='o', markersize=12,
                   markerfacecolor='grey', linestyle='', markeredgecolor='w')
    xE_leg = Line2D([0], [0], label='$x_E$ (Neuronal)', marker='o',
                    markersize=12, markerfacecolor='C0', linestyle='',
                    markeredgecolor='w')
    xI_leg = Line2D([0], [0], label='$x_I$', marker='o', markersize=8,
                    markerfacecolor='C5', linestyle='', markeredgecolor='w')
    a_leg = Line2D([0], [0], label='$a$', marker='o', markersize=8,
                   markerfacecolor='C4', linestyle='', markeredgecolor='w')
    f_leg = Line2D([0], [0], label='$f$ (CBF)', marker='o', markersize=12,
                   markerfacecolor='C2', linestyle='', markeredgecolor='w')
    fout_leg = Line2D([0], [0], label='$f_{out}$', marker='o', markersize=8,
                      markerfacecolor='C8', linestyle='', markeredgecolor='w')
    v_leg = Line2D([0], [0], label='$v$ (CBV)', marker='o', markersize=12,
                   markerfacecolor='C1', linestyle='', markeredgecolor='w')
    q_leg = Line2D([0], [0], label='$q$', marker='o', markersize=8,
                   markerfacecolor='C6', linestyle='', markeredgecolor='w')
    E_leg = Line2D([0], [0], label='$E$', marker='o', markersize=8,
                   markerfacecolor='C9', linestyle='', markeredgecolor='w')
    y_leg = Line2D([0], [0], label='$y$ (BOLD)', marker='o', markersize=12,
                   markerfacecolor='C3', linestyle='', markeredgecolor='w')

    handle_list = [u_leg, xE_leg, xI_leg, a_leg, f_leg, fout_leg, v_leg, q_leg,
                   E_leg, y_leg]

    host.set_xlabel("$t$ (sec)")
    host.set_ylabel("Variables expect $y$")
    host.legend(handles=handle_list, loc='upper right')
    host.grid()

    ax2.set_ylabel("$y$ - BOLD signal")

    if indicator is True:
        plt.title('Reproduce Fig. 3/4 (C) (Coupled CBF-CBV)\n'+str(method)+'')
    else:
        plt.title('Reproduce Fig. 3/4 (C) (Uncoupled CBF-CBV)\n'+str(
            method)+'')

    plt.show()


def sub_plot(t, var_select, name_list, color_list, stimulus, method,
             indicator, linestyle, linewidth, amp_lign=True):
    """Produce subplots for xE, f, v, y to reproduce Fig. 3 (C) or 4 (C) in
       Havlicek et. al.'s paper

    Args:
    - t -- list of time values
    - var_select -- lists of xE, f, v, y
    - name_list -- string of the names of the variables
    - color_list -- string of the color code for plotting each variable
    - stimulus -- time duration of the input stimulus
    - method -- string of the name of the numerical method
    - indicator -- controls the plot title to be "Coupled CBF-CBV" or
                   "Uncoupled CBF-CBV"
    - linestyle -- list of linestyles for the plot
    - linewidth -- list of linewidths for the plot
    - amp_lign -- align the peaks of the curves
    """

    plt.rcParams.update(plt.rcParamsDefault)
    params = {
        'figure.figsize': [15, 12],
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'axes.titlesize': 25,
        'figure.dpi': 300,
        'figure.subplot.wspace': 0.3,
        'figure.subplot.hspace': 0.3
        }
    plt.rcParams.update(params)

    for i in range(len(name_list)):
        plt.subplot(2, 2, i+1)

        for j in range(len(var_select)):
            if amp_lign is True:
                shift = var_select[i][j] - var_select[i][j][0]
                normalise = shift / shift.max() + var_select[i][j][0]
            else:
                normalise = var_select[i][j]
            plt.plot(t, normalise,
                     ls=''+str(linestyle[j])+'', lw=''+str(linewidth[j])+'',
                     color=''+str(color_list[i])+'')

        plt.xlabel('$t$ (sec)')
        plt.ylabel(''+str(name_list[i])+' (AU)')
        plt.grid()

    if indicator is True:
        plt.suptitle('Reproduce Fig. 3/4 (C) with stimulus '+str(
            stimulus)+' s (Coupled CBF-CBV)\n'+str(method)+'', fontsize=30)
    else:
        plt.suptitle('Reproduce Fig. 3/4 (C) with stimulus '+str(
            stimulus)+' s (Uncoupled CBF-CBV)\n'+str(method)+'', fontsize=30)

    plt.show()


def L2norm_bar(cons_list, y_list_all, cons_name, method):
    """Calculate the L2 norm when the value of the input costant changes, and
       compare the resulted L2 norm on a bar chart

    Arg:
    - cons_list -- a list of values in the plausible range of a constant
    - y_list_all -- a list of the resulted BOLD responses calculated from the
                    numerical solver
    - cons_name -- a string for the name of the constant
    - method -- string of the name of the numerical method

    Return:
    A bar chart of the calculated L2 norms
    """
    plot_para()

    # Bar plot for L2 norm
    fig, ax = plt.subplots()

    names = []
    values = []

    # Calculate L2 norm for y
    for j in range(len(y_list_all) - 1):
        root_len = np.sqrt(len(y_list_all[0]))
        l2norm = np.linalg.norm(y_list_all[0] - y_list_all[j+1]) / root_len

        names.append(""+str(round(cons_list[0], 3))+"\nto\n"+str(round(
            cons_list[j+1], 3))+"")
        values.append(l2norm)

    ax.bar(names, values, color='#A7A7EE', width=0.4)

    # Add annotation to bars
    for i in ax.patches:
        plt.annotate(format(i.get_height(), '.3f'),
                     (i.get_x() + i.get_width()/2,
                     i.get_height()), ha='center', va='center', size=18,
                     xytext=(0, 11), textcoords='offset points')

    ax.set_title('$L^2$ norm of BOLD response changes due to different '+str(
        cons_name)+'\n'+str(method)+'')
    ax.set_xlabel(""+str(cons_name)+" change")
    ax.set_ylabel("$L^2$ norm")
    ax.grid()

    plt.show()
