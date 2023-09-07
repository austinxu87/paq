import numpy as np

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

def get_meas(D, N, Sig, y = 1):
    As = np.zeros((D, D, N))
    gammas = []

    for i in range(N):
        a_i = np.random.normal(size=(D))
        quad_i = a_i.T @ Sig @ a_i
        gamma = (y + np.random.uniform(-1,1)) / quad_i

        gammas.append(gamma)
        As[:,:,i] = gamma*np.outer(a_i, a_i)

    return As, gammas

# Get triplets
def get_triplets(d, N, Sig, noise = False):
    As = np.zeros((d, d, N))
    labels = []
    a_is = []

    for i in range(N):
        a_i = np.random.normal(size=(3,d))
        A = 2*np.outer(a_i[0], a_i[2] - a_i[1]) + np.outer(a_i[1], a_i[1]) - np.outer(a_i[2], a_i[2])
        As[:,:,i] = A

        dist = np.trace(A @ Sig)
        if noise:
            dist += np.random.uniform(-1,1)

        if dist > 0:
            outcome = 1
        else:
            outcome = -1

        labels.append(outcome)
        a_is.append(a_i)

    return As, labels, a_is

def get_binary(d, N, Sig, y, noise = False):
    As = np.zeros((d, d, N))
    labels = []
    a_is = []

    for i in range(N):
        a_i = np.random.normal(size=(2,d))
        A = np.outer(a_i[0] - a_i[1], a_i[0] - a_i[1])
        As[:,:,i] = A

        dist = np.trace(A @ Sig)
        if noise:
            dist += np.random.uniform(-1,1)

        if dist > y:
            outcome = 1
        else:
            outcome = -1
        
        labels.append(outcome)
        a_is.append(a_i)

    return As, labels, a_is


def get_tuplewise(d, N, Sig, y, tuple_len, noise = False):
    num_As = int(N*(tuple_len-2)*(tuple_len-1) / 2)
    As = np.zeros((d, d, num_As))
    labels = []
    a_is = []
    store_ind = 0
    for i in range(N):
        a_i = np.random.normal(size=(tuple_len, d)) #a_i[0] is reference
        
        # get all distances
        dists = []
        for j in range(1,tuple_len):
            A = np.outer(a_i[0] - a_i[j], a_i[0] - a_i[j])
            dist = np.trace(A @ Sig)
            if noise:
                dist += np.random.uniform(-1,1)

            dists.append(dist)

        # sort distances, decompose into triplets
        inds_sorted = np.argsort(dists)
        for ind1 in range(len(inds_sorted)):
            j = inds_sorted[ind1] + 1
            for ind2 in range(ind1+1, len(inds_sorted)):
                k = inds_sorted[ind2] + 1
                A = 2*np.outer(a_i[0], a_i[k] - a_i[j]) + np.outer(a_i[j], a_i[j]) - np.outer(a_i[k], a_i[k])
                As[:,:,store_ind] = A
                labels.append(-1)
                a_is.append(a_i[inds_sorted ,:])

                store_ind += 1

    return As, labels, a_is
        

                
