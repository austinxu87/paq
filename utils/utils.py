import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

#Inverted Measurements

def get_meas(D, m, n, tau, Sig, y = 5, eta_up = 5):

    As = np.zeros((D, D, n))
    gammas = []
    gammas_all = []
    gammas_bar_all = []
    for i in range(n):
        #Sample a from multivariate normal, take quadratic measurement
        a_i = np.random.normal(size=(D))
        quad_i = a_i.T @ Sig @ a_i

        #Get noisy distance
        if eta_up > 0:
            y_tilde = y + np.random.uniform(low=-eta_up, high=eta_up, size=m)
        else:
            y_tilde = np.array([y for _ in range(m)])

        #Get m samples of noisy distance, average, then truncate
        gamma_sq = y_tilde / quad_i
        gammas_all += list(gamma_sq)
        gamma_bar = np.average(gamma_sq)
        gammas_bar_all.append(gamma_bar)
        #print(np.sum(gamma_sq > tau) / m)
        gamma_tilde = np.minimum(tau, gamma_bar)


        #Store sensing matrix and gammas
        As[:,:,i] = gamma_tilde*np.outer(a_i, a_i)
        gammas.append(gamma_tilde)

    gammas_all = np.array(gammas_all)
    #print("all gammas: mean = %0f, std = %0f" % (np.mean(gammas_all), np.std(gammas_all)))
    #print("gamma_bar: mean = %0f, std = %0f" % (np.mean(gammas_bar_all), np.std(gammas_bar_all)))
    return As, gammas

#Noiseless measurements with no averaging or truncation
def get_meas_noiseless(D, N, Sig, y = 1):
    As = np.zeros((D, D, N))
    gammas = []

    for i in range(N):
        a_i = np.random.normal(size=(D))
        quad_i = a_i.T @ Sig @ a_i
        gamma = y / quad_i

        gammas.append(gamma)
        As[:,:,i] = gamma*np.outer(a_i, a_i)


    return As, gammas


#Generate ground truth Sigma and tau for outer product of Gaussian matrices
#  Same data generation strategy as Chen et al (https://arxiv.org/pdf/1310.0807.pdf)
def get_gt(D, r, y, eta_up, n):
    L = np.random.normal(size=(D, r))
    Sig = L@L.T
    singular_vals = np.sort(np.linalg.svd(Sig)[1])

    M_14 = ((y + eta_up) / (4*singular_vals[-r]*(r-8)))
    tau = M_14*np.sqrt(n / D)

    return Sig, tau

#Generate ground truth Sigma and tau for outer product of orthonormal matrices
#  Same approach as https://arxiv.org/pdf/1709.06171.pdf
def get_gt_orth(D, r, y, eta_up, n):
    L = np.random.randn(D, r)
    Q, _ = np.linalg.qr(L)
    Sig = (D / np.sqrt(r)) * Q @ Q.T

    singular_vals = np.sort(np.linalg.svd(Sig)[1])

    if r > 8:
        M_14 = ((y + eta_up) / (4*singular_vals[-r]*(r-8)))
    else:
        M_14 = ((y + eta_up) / (4*singular_vals[-r]*r))

    tau = M_14*np.sqrt(n / D)
    return Sig, tau

#Get reg. parameter lambda based on theory
def get_lda(D, n, m, r, singular_vals=None, C_max = None):
    #Input C_max controls the scaling constant.
    #  If C_max = None, compute C_max based on theory (Prop 2)
    #  If C_max = (some const), just use C_max, ignore theory
    if not C_max:
        kappa_0 = 4
        v_1 = 40*kappa_0**2
        c_1 = 2 * np.exp(1)*kappa_0

        C_1  = 9*(2*y + eta_up)**2*(np.sqrt(v_1) + c_1 + 2*kappa_0) / (2*singular_vals[-r]) #this constant is big
        C_2 = (9/7)*((2*eta_up)**2/12)/(7*singular_vals[-r])

        C_max = np.maximum(C_1, C_2)

    return 2*C_max*(np.sqrt(D/n) + 1/m)/r


#Solve estimation problem
def get_est(A_flat, y, lda, D, n, max_iter=10000):
    Sig_hat = cp.Variable((D, D), PSD=True)

    loss = cp.sum_squares(y - cp.vec(Sig_hat) @ A_flat) / n + lda*cp.norm(Sig_hat, 'nuc')
    obj = cp.Minimize(loss)
    prob = cp.Problem(obj)

    prob.solve(solver=cp.SCS, max_iters=max_iter)

    return Sig_hat.value


def plot_exp(norm_err_sweep, n_v, type_plot, params, filename, err_bar = 'std_err'):
    plt.rcParams.update({'font.size': 72})

    filename += "_"
    filename += err_bar

    N_v = []
    for n in n_v:
        d = params['d']
        eta_up = params['eta_up']
        noise_var = (eta_up ** 2) / 3

        m = int(np.ceil(np.sqrt(n / d)) * noise_var)
        N = n * m
        N_v.append(N)

    N_v = np.array(N_v)


    #plot parameters
    #fig_num = 0 1 2 ----->     x-axis is n, n, logn
    #                           y-axis is err, log err, log err
    #fig_num = 3 4 5 ----->     x-axis is N, N, logN
    #                           y-axis is err, log err, log err
    for fig_num in [5]:
        print(fig_num)
        plt.figure(fig_num)

        if fig_num in [0,1]:
            x_plot = n_v
        elif fig_num == 2:
            x_plot = np.log10(n_v)
        elif fig_num in [3,4]:
            x_plot = N_v
        else:
            x_plot = np.log10(N_v)

        for k, v in sorted(norm_err_sweep.items()):

            mean_err = np.array([np.mean(e) for e in v])
            std_err = np.array([np.std(e) for e in v])
            if err_bar == 'std_err':
                std_err /= np.sqrt(params['MC'])

            legend_str = type_plot + " = " + str(k)

            if fig_num in [0,3]:
                plt.plot(x_plot, mean_err, label=legend_str, linewidth=10.0)
                plt.fill_between(x_plot, mean_err-std_err, mean_err+std_err, alpha=0.7)
            elif fig_num == 5:
                x_plot = np.log10(N_v / (k))
                plt.plot(x_plot, np.log10(mean_err), label=legend_str, linewidth=10.0)
                plt.fill_between(x_plot, np.log10(mean_err-std_err), np.log10(mean_err+std_err), alpha=0.7)
            else:
                plt.plot(x_plot, np.log10(mean_err), label=legend_str, linewidth=10.0)
                plt.fill_between(x_plot, np.log10(mean_err-std_err), np.log10(mean_err+std_err), alpha=0.7)

            plt.legend(prop={'size': 52})

            plot_scale_str = ""
            if fig_num in [0,1]:
                plt.xlabel("Number of effective measurements")
                plot_scale_str += "_n"
            elif fig_num == 2:
                plt.xlabel("Number of effective measurements (log10)")
                plot_scale_str += "_logn"
            elif fig_num in [3, 4]:
                plt.xlabel("Number of total measurements")
                plot_scale_str += "_N"
            elif fig_num == 5:
                plt.xlabel("Normalized total measurements N/d (log10)")
                plot_scale_str += "_logN_rd"


            if fig_num in [0, 3]:
                plt.ylabel("Normalized est. error")
                plot_scale_str += "_err"
            else:
                plt.ylabel("Normalized est. error (log10)")
                plot_scale_str += "_logerr"

        plt.gcf().set_size_inches(28, 21)
        plt.savefig(filename + plot_scale_str + '.jpg', bbox_inches='tight', dpi=300)
        plt.close()
        #plt.show()


def save_exp(norm_err_sweep, n_v, type_plot, params, filename):
    exp_out = {
        'norm_err' : norm_err_sweep,
        'n_v' : n_v,
        'sweep_type' : type_plot,
        'params' : params
    }

    pickle.dump(exp_out, open(filename + '_m_up.pkl', 'wb'))

def save_exp_m(norm_err_sweep, m_v, params, filename):
    exp_out = {
        'norm_err' : norm_err_sweep,
        'm_v' : m_v,
        'params' : params
    }

    pickle.dump(exp_out, open(filename + '_sweep_m.pkl', 'wb'))


def load_plot_exp(filename):
    exp_saved = pickle.load(open(filename + '_m_up.pkl', 'rb'))

    params = exp_saved['params']
    norm_err = exp_saved['norm_err']
    n_v = exp_saved['n_v']
    sweep_type = exp_saved['sweep_type']

    plot_exp(norm_err, n_v, sweep_type, params, filename)



def plot_exp_m(norm_err_sweep, m_v, params, filename, err_bar = 'std_err'):
    plt.rcParams.update({'font.size': 72})

    filename += "_"
    filename += err_bar

    for fig_num in range(2):
        plt.figure(fig_num)

        if fig_num == 0:
            x_plot = m_v
        elif fig_num == 1:
            x_plot = np.log10(m_v)

        mean_err = np.array([np.mean(e) for e in norm_err_sweep])
        std_err = np.array([np.std(e) for e in norm_err_sweep])
        if err_bar == 'std_err':
            std_err /= np.sqrt(params['MC'])


        plt.plot(x_plot, mean_err, linewidth=10.0)
        plt.fill_between(x_plot, mean_err-std_err, mean_err+std_err, alpha=0.7)

        plot_scale_str = ""
        if fig_num == 0:
            plt.xlabel("Averaging parameter m")
            plot_scale_str += "_m"
        elif fig_num == 1:
            plt.xlabel("Averaging parameter m (log10)")
            plot_scale_str += "_logm"

        plt.ylabel("Normalized estimation error")
        plot_scale_str += "_err"

        plt.gcf().set_size_inches(28, 21)
        plt.savefig(filename + plot_scale_str + '.jpg', bbox_inches='tight', dpi=300)
        plt.close()
        #plt.show()
