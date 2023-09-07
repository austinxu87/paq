import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from utils.baseline_utils import *
import os
import json
from datetime import datetime
from collections import defaultdict

def get_est_noiseless(A_flat, y, lda, D, n, max_iter=10000):
    Sig_hat = cp.Variable((D, D), PSD=True)

    loss = cp.sum_squares( y - (cp.vec(Sig_hat) @ A_flat) ) / n + lda*cp.norm(Sig_hat, 'nuc')
    obj = cp.Minimize(loss)
    prob = cp.Problem(obj)

    prob.solve(solver=cp.SCS, max_iters=max_iter)

    return Sig_hat.value

def get_triplet_est(A_flat, labels, lda, D, n, max_iter=10000):
    Sig_hat = cp.Variable((D,D), PSD=True)

    loss = cp.sum(cp.pos(1 - cp.multiply(labels, (cp.vec(Sig_hat) @ A_flat)) ) ) / n + lda*cp.norm(Sig_hat, 'nuc')
    obj = cp.Minimize(loss)
    prob = cp.Problem(obj)

    prob.solve(solver=cp.SCS, max_iters=max_iter)

    return Sig_hat.value

def get_binary_est(A_flat, labels, lda, y, D, n, max_iter=10000):
    Sig_hat = cp.Variable((D,D), PSD=True)

    loss = cp.sum(cp.pos( cp.multiply(labels, y - (cp.vec(Sig_hat) @ A_flat)) ) ) / n + lda*cp.norm(Sig_hat, 'nuc')
    obj = cp.Minimize(loss)
    prob = cp.Problem(obj)

    prob.solve(solver=cp.SCS, max_iters=max_iter)

    return Sig_hat.value


np.random.seed(0)

d = 50
r = 10
y = 10
MC = 10
lda = 0.05
tuple_lens = [4, 5]

fname = 'noiseless_baseline_sim.json'
err_all = defaultdict(dict)
if os.path.isfile(fname):
    with open(fname, 'r') as f:
        err_loaded = json.load(f)

        for k, v in err_loaded.items():
            err_all[k] = v

    
N_v = list(range(100, 1001, 100)) + [1500, 2000, 2500]
for N in N_v:
    err_N_paq = []
    err_N_trip = []
    err_N_bin = []
    err_N_tuple = []
    for mc in range(MC):
        print(f'N = {N}')
        print(f'mc = {mc+1} / {MC}')
        L = np.random.normal(size=(d, r))
        Sig = L@L.T
        Sig /= np.linalg.norm(Sig)
        
        N_key = str(N)

        if 'paq' not in err_all[N_key]:
            print('Processing PAQ')
            As, gammas = get_meas(d, N, Sig, y = y)
            A_flat = As.reshape(d*d, -1)
            S_hat = get_est_noiseless(A_flat, y, 0.05, d, N)

            err = np.linalg.norm(S_hat - Sig, 'fro') / np.linalg.norm(Sig)
            err_N_paq.append(err)

        if 'triplets' not in err_all[N_key]:
            print('Processing triplets')
            As, labels, _ = get_triplets(d, N, Sig, noise=False)
            A_flat = As.reshape(d*d, -1)
            labels = np.array(labels)

            Sig_hat = get_triplet_est(A_flat, labels, lda, d, N)
            Sig_hat /= np.linalg.norm(Sig_hat)
            
            err = np.linalg.norm(Sig_hat - Sig, 'fro') / np.linalg.norm(Sig)
            err_N_trip.append(err)

        if 'binary' not in err_all[N_key]:
            print('Processing binary')
            As, labels, _ = get_binary(d, N, Sig, y, noise=False)
            A_flat = As.reshape(d*d, -1)
            labels = np.array(labels)

            Sig_hat = get_binary_est(A_flat, labels, lda, y, d, N)
            Sig_hat /= np.linalg.norm(Sig_hat)

            err = np.linalg.norm(Sig_hat - Sig, 'fro') / np.linalg.norm(Sig)
            err_N_bin.append(err)

        for tuple_len in tuple_lens:
            err_N_tuple_len = []
            if f'tuple-{tuple_len}' not in err_all[N_key]:
                print(f'Processing tuple-{tuple_len}')
                As, labels, _ = get_tuplewise(d, N, Sig, y, tuple_len, noise=False)
                A_flat = As.reshape(d*d, -1)
                labels = np.array(labels)

                Sig_hat = get_triplet_est(A_flat, labels, lda, d, As.shape[-1])
                Sig_hat /= np.linalg.norm(Sig_hat)

                err = np.linalg.norm(Sig_hat - Sig, 'fro') / np.linalg.norm(Sig)
                err_N_tuple_len.append(err)
            err_N_tuple.append(err_N_tuple_len)

    if err_N_paq:
        err_all[N_key]['paq'] = err_N_paq
        print(f'PAQ mean error = {np.mean(err_N_paq)} | std = {np.std(err_N_paq)}')
        
    if err_N_trip:
        err_all[N_key]['triplets'] = err_N_trip
        print(f'Triplet mean error = {np.mean(err_N_trip)} | std = {np.std(err_N_trip)}')
    
    if err_N_bin:
        err_all[N_key]['binary'] = err_N_bin
        print(f'Binary mean error = {np.mean(err_N_bin)} | std = {np.std(err_N_bin)}')

    for ct, tuple_len in enumerate(tuple_lens):
        if err_N_tuple[ct]:
            err_all[N_key][f'tuple-{tuple_len}'] = err_N_tuple[ct]
            print(f'Tuple-{tuple_len} mean error = {np.mean(err_N_tuple[ct])} | std = {np.std(err_N_tuple[ct])}')
    
    with open(fname, 'w') as f:
        json.dump(err_all, f, indent=4)

    print('--------------')
