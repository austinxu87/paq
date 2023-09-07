import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from utils.utils import *
import os
from datetime import datetime


def run_sweep(sweep_type, sweep_range, params, filename):

    np.random.seed(0)

    MC = params['MC']
    max_iter = params['max_iter']
    if sweep_type == "d":
        r, y, eta_up = params['r'], params['y'], params['eta_up']
    elif sweep_type == "r":
        d, y, eta_up = params['d'], params['y'], params['eta_up']

    n_v = params['n_v']

    norm_err_sweep_all = {}
    for sweep_val in sweep_range:
        if sweep_type == 'd':
            d = sweep_val
        elif sweep_type == 'r':
            r = sweep_val
        
        print('%s = %0d -----------------------------------' % (sweep_type, sweep_val) )

        norm_err_sweep = []
        for n in n_v:
            norm_err_mc = []

            noise_var = (eta_up ** 2) / 3
            m = int(np.ceil(np.sqrt(n / d)) * noise_var)
            N = int(m*n)
            print(f"n = {n}, N = {N} --------------------------" )
            for mc in range(MC):
                if (mc + 1) % 5 == 0:
                    print("mc = %0d / %0d" % (mc+1, MC))

                Sig, tau = get_gt_orth(d, r, y, eta_up, n)

                As, gammas = get_meas(d, m, n, tau, Sig, y = y, eta_up = eta_up)
                A_flat = As.reshape(d*d, -1)
                lda = get_lda(d, n, m, r, C_max = 1)
                Sig_est = get_est(A_flat, y, lda, d, n, max_iter=max_iter)

                norm_err_mc.append(np.linalg.norm(Sig - Sig_est, 'fro') / np.linalg.norm(Sig, 'fro'))

            norm_err_sweep.append(norm_err_mc)

        norm_err_sweep_all[sweep_val] = norm_err_sweep

    save_exp(norm_err_sweep_all, n_v, sweep_type, params, filename)
    plot_exp(norm_err_sweep_all, n_v, sweep_type, params, filename)


def run_sweep_m(m_v, params, filename):

    np.random.seed(0)

    MC = params['MC']
    max_iter = params['max_iter']
    d, r, y, eta_up = params['d'], params['r'], params['y'], params['eta_up']
    N = params['N']

    filename += '_N%0d_d%0d' % (N, d)

    norm_err_sweep = []
    for m in m_v:
        print(f'm = {m} -----------------------------------')
        n = int(N / m)
        norm_err_mc = []

        for mc in range(MC):
            if (mc + 1) % 5 == 0:
                print("mc = %0d / %0d" % (mc+1, MC))


            Sig, tau = get_gt_orth(d, r, y, eta_up, n)
            As, gammas = get_meas(d, m, n, tau, Sig, y = y, eta_up = eta_up) #As are gamma_i a_i a_i^T, so gammas unused
            A_flat = As.reshape(d*d, -1)

            lda = get_lda(d, n, m, r, C_max = 1)
            Sig_est = get_est(A_flat, y, lda, d, n, max_iter=max_iter)

            norm_err_mc.append(np.linalg.norm(Sig - Sig_est, 'fro') / np.linalg.norm(Sig, 'fro'))

        norm_err_sweep.append(norm_err_mc)

    save_exp_m(norm_err_sweep, m_v, params, filename)
    plot_exp_m(norm_err_sweep, m_v, params, filename)



if __name__ == '__main__':

    # Run sweeps of dimension d and rank r
    sweep_types = ['d', 'r']

    sweep_vals = {
        'd' : [40, 50, 60],
        'r' : [5, 8, 9, 10, 15, 20],
    }

    params = {
        'r' : 9,
        'y' : 200,
        'eta_up' : 200,
        'd' : 50,
        'MC' : 20,
        'n_v' : np.arange(500, 5001, 100),
        'N' : 50000,
        'max_iter' : 50000,
        'averaging' : True,
        'truncation' : True,
        'trunc_type' : 'fixed'
    }


    exp_dir = os.path.join("/Users/austinxu/Documents/paq-learning/figures/", datetime.today().strftime('%Y-%m-%d'))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    sweep_types = ['d']

    for sweep_type in sweep_types:
        print('Running sweep for %s ====================================' % sweep_type)
        sweep_range = sweep_vals[sweep_type]

        filename = os.path.join(exp_dir, "error_vary_" + sweep_type)
        run_sweep(sweep_type, sweep_range, params, filename)
        load_plot_exp(filename)


    # Run sweep of averaging parameter m
    
    params['N'] = 50000
    params['d'] = 50
    m_v = [1, 2, 5, 10, 20, 25, 40, 50]
    filename = os.path.join(exp_dir, 'error_vary_m')
    run_sweep_m(m_v, params, filename)


    N = 20000
    params['N'] = 20000
    m_v = [1, 2, 5, 8, 10, 20, 25, 50]
    filename = os.path.join(exp_dir, 'error_vary_m')
    run_sweep_m(m_v, params, filename)

    #N = 40000
    params['N'] = 40000
    params['d'] = 40
    m_v = [2, 5, 10, 16, 20, 25, 40, 50]
    filename = os.path.join(exp_dir, 'error_vary_m')
    run_sweep_m(m_v, params, filename)

    #N = 16000
    params['N'] = 16000
    params['d'] = 40
    m_v = [1, 2, 5, 8, 10, 20, 25, 50]
    filename = os.path.join(exp_dir, 'error_vary_m')
    run_sweep_m(m_v, params, filename)

