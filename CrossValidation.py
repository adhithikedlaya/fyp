"""
Produce plots to cross validate with Fig. 3 (C) or 4 (C) in Havlicek et. al.'s
paper
"""

import numpy as np

import Stimulus_PDCM_BOLD as spb
import PlotFunctions as pf
from Euler_1stOrderForward import euler_1st
# from RK_4thOrderForward import runge_kutta_4th


# Select the numerical method
method = 'Euler 1st order forward'
#method = 'Runge-Kutta 4th order forward'


# Reproduce Fig. 3 and 4
amp_lign = True  # rescaling to align the peaks

stimulus_list = [1, 30]  # two stimulus durations

sigma = 1
psi = 0.6
chi = 1.5
phi = 0.6
alpha = 0.32
tMTT = 2
E0 = 0.4

for i in range(len(stimulus_list)):
    stimulus = stimulus_list[i]

    # Iteration values
    t_ref = spb.time_ref(stimulus)  # controls initial and final t

    h = 0.01  # step size, unit: sec
    t = np.arange(t_ref[0], t_ref[-1] + t_ref[1], h)  # all t values

    if stimulus == 1:
        # CBF-CBV coupled
        mu_list = [0.8, 0.2, 0.5, 2]
        lamb_list = [0.2, 0.2, 0.15, 0.2]
        c_couple = [0.25, 0.25, 0.25, 0.25]

        # CBF-CBV uncoupled
        tau_list = [3, 8, 15, 27]
        c_uncouple = [0.25, 0.25, 0.25, 0.25]

    elif stimulus == 30:
        # CBF-CBV coupled
        mu_list = [1, 0.3, 0.35, 0.65]
        lamb_list = [0.2, 0.2, 0.1, 0.1]
        c_couple = [0.25, 0.25, 0.25, 0.25]

        # CBF-CBV uncoupled
        tau_list = [3, 8, 15, 27]
        c_uncouple = [0.25, 0.25, 0.25, 0.25]

    # CBF-CBV coupled
    indicator = True

    linestyle_list = ['-', '-', '--', '-.']
    linewidth_list = [5, 3, 3, 3]

    xE_all = []
    f_all = []
    v_all = []
    y_all = []
    for j in range(len(mu_list)):
        mu = mu_list[j]
        lamb = lamb_list[j]
        c = c_couple[j]

        func = spb.StimulusPDCMBOLD(w=stimulus,
                                    sigma=sigma, mu=mu, lamb=lamb, c=c,
                                    phi=phi, chi=chi, psi=psi,
                                    alpha=alpha, tMTT=tMTT, E0=E0,
                                    cross_valid=True)

        if method.find('Euler') == 0:
            var_init = [spb.xE_init, spb.xI_init, spb.a_init, spb.f_init,
                        spb.v_init, spb.q_init, spb.fout_init, spb.E_init,
                        spb.y_init]

            (u_list, xE_list, xI_list, a_list, f_list, v_list, q_list,
             fout_list, E_list, y_list) = euler_1st(func, t, h, var_init,
                                                    indicator)

            xE_all.append(xE_list) 
            f_all.append(f_list)
            v_all.append(v_list)
            y_all.append(y_list)

        elif method.find('Runge') == 0:
            var_ode_init = [spb.xE_init, spb.xI_init, spb.a_init, spb.f_init,
                            spb.v_init, spb.q_init]
            var_equ_init = [spb.E_init, spb.fout_init, spb.y_init]

            results_ode, results_equ, u_list = runge_kutta_4th(
                func, t, h, var_ode_init, var_equ_init, indicator)

            xE_all.append(results_ode[:, 0])
            f_all.append(results_ode[:, 3])
            v_all.append(results_ode[:, 4])
            y_all.append(results_equ[:, 2])

    var_select = [xE_all, f_all, v_all, y_all]
    name_list = ['$x_E$ (Neuronal)', '$f$ (CBF)', '$v$ (CBV)', '$y$ (BOLD)']
    color_list = ['C0', 'C2', 'C1', 'C3']

    pf.sub_plot(t, var_select, name_list, color_list, stimulus, method,
                indicator, linestyle_list, linewidth_list, amp_lign)

    # CBF-CBV uncoupled
    indicator = False

    linestyle_list = ['-', ':', '-.', '--']
    linewidth_list = [4, 3, 3, 3]

    xE_all = []
    f_all = []
    v_all = []
    y_all = []
    for j in range(len(tau_list)):
        mu = mu_list[0]
        lamb = lamb_list[0]
        c = c_uncouple[j]

        tau = tau_list[j]

        func = spb.StimulusPDCMBOLD(w=stimulus,
                                    sigma=sigma, mu=mu, lamb=lamb, c=c,
                                    phi=phi, chi=chi, psi=psi,
                                    alpha=alpha, tMTT=tMTT, E0=E0,
                                    tau=tau)

        if method.find('Euler') == 0:
            var_init = [spb.xE_init, spb.xI_init, spb.a_init, spb.f_init,
                        spb.v_init, spb.q_init, spb.fout_init, spb.E_init,
                        spb.y_init]

            (u_list, xE_list, xI_list, a_list, f_list, v_list, q_list,
             fout_list, E_list, y_list) = euler_1st(func, t, h, var_init,
                                                    indicator)

            xE_all.append(xE_list)
            f_all.append(f_list)
            v_all.append(v_list)
            y_all.append(y_list)

        elif method.find('Runge') == 0:
            var_ode_init = [spb.xE_init, spb.xI_init, spb.a_init, spb.f_init,
                            spb.v_init, spb.q_init]
            var_equ_init = [spb.E_init, spb.fout_init, spb.y_init]

            results_ode, results_equ, u_list = runge_kutta_4th(
                func, t, h, var_ode_init, var_equ_init, indicator)

            xE_all.append(results_ode[:, 0])
            f_all.append(results_ode[:, 3])
            v_all.append(results_ode[:, 4])
            y_all.append(results_equ[:, 2])

    var_select = [xE_all, f_all, v_all, y_all]
    name_list = ['$x_E$ (Neuronal)', '$f$ (CBF)', '$v$ (CBV)', '$y$ (BOLD)']
    color_list = ['C0', 'C2', 'C1', 'C3']

    # pf.sub_plot(t, var_select, name_list, color_list, stimulus, method,
    #             indicator, linestyle_list, linewidth_list, amp_lign)
