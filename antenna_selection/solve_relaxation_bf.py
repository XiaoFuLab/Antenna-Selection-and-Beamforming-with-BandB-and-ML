import cvxpy as cp
import numpy as np
import time

def solve_relaxed(H=None, 
                z_mask=None, 
                z_sol=None, 
                min_sinr=1.0, 
                sigma_sq=1.0):
    """
    Lower bound method that solves a smaller sized subproblem with selected and undecided antennas 
    """
    N_original, M = H.shape

    # sigma_sq= sigma_sq*np.ones(M)
    # min_sinr= min_sinr*np.ones(M) #SINR levels, from -10dB to 20dB

    H_short = H.copy()
    for n in range(N_original-1, -1, -1):
        if z_mask[n] and not z_sol[n]:
            H_short = np.concatenate((H_short[:n, :], H_short[n+1:, :]), axis=0)
    # print(H_short)
    num_off_ants = np.sum(z_mask*(1-z_sol))
    assert num_off_ants<N_original, 'number of allowed antennas {} < 1'.format(N_original - num_off_ants)
    N = int(N_original - num_off_ants)

    c_1 = (1/np.sqrt(min_sinr*sigma_sq))
    c_2 = (1/sigma_sq)

    W = cp.Variable((N, M), complex=True, name='W')
    obj = cp.Minimize(cp.square(cp.norm(W, 'fro')))

    constraints = []
    for m in range(M):
        Imask = np.eye(M)
        Imask[m,m] = 0
        constraints += [c_1*cp.real(np.expand_dims(H_short[:,m], axis=0).conj() @ W[:,m]) >= cp.norm(cp.hstack((c_2*((W @ Imask).H) @ H_short[:,m], np.ones(1))), 2)]

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
    except Exception as e:
        print(e)
        return np.zeros(N_original), np.zeros((N_original,M)), np.inf, False
        
    if prob.status in ['infeasible', 'unbounded']:
        # print('infeasible antenna solution')
        return np.zeros(N_original), np.zeros((N_original,M)), np.inf, False

    W_sol = W.value
    for n in range(N_original):
        if z_mask[n] and not z_sol[n]:
            W_sol = np.concatenate((W_sol[:n, :], np.zeros((1,M)), W_sol[n:,:]), axis=0)
    assert W_sol.shape[0] == N_original, 'W not of correct shape'
    
    return z_sol.copy(), W_sol, prob.objective.value, True


def compute_sinr(W=None, H=None, sigma_sq=1):
    assert W is not None and H is not None, "Input not provided"
    W_H = np.matmul(H.conj().T, W)
    W_H = np.abs(W_H)**2
    mask = np.eye(W_H.shape[0])
    mask_comp = 1-mask
    direct = np.sum(W_H*mask, axis=1)
    interference = W_H*mask_comp
    aggregate_interference = np.sum(interference, axis=1)
    print('sinr computation', direct.shape, aggregate_interference.shape)

    sinr = direct/(aggregate_interference + sigma_sq)
    return sinr

if __name__=='__main__':
    # N, K = 8,3
    # max_ant = 5
    # np.random.seed(150)
    # H = np.random.randn(N, K) + 1j*np.random.randn(N, K)

    # # z_sol = np.random.binomial(size=N, n=1, p= 0.5)
    # # z_mask = np.random.binomial(size=N, n=1, p=0.2)

    # z_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    # z_sol = np.array([1, 0, 1, 1, 1, 0, 0, 1])
    # # print(z_mask)
    
    # print(z_sol)
    # t1 = time.time()
    # z, W, obj = solve_beamforming_relaxed(H, max_ant=5, z_sol=z_sol, z_mask=z_mask)
    # print("TIME for completion: ", time.time()- t1)
    # print(H)
    # # t1 = time.time()
    # # z, W, obj = solve_beamforming_relaxed(H, max_ant=5, z_sol=z_sol, z_mask=z_mask, z_init=z, W_init=W)
    # # print("TIME for completion 2: ", time.time()- t1)

    # # obj2 = solve_beamforming_with_selected_antennas(H, z_sol)

    # # print(z)
    # # print(W)
    # print(obj)
    # # print(obj2)
    # # if obj2== np.inf:
    # # print('problem infeasible')
    N,M,L = 8,3,5
    np.random.seed(100)

    H = (np.random.randn(N,M) + 1j*np.random.randn(N,M))/np.sqrt(2)

    # H =np.array([[ 0.1414 + 0.3571j,   0.3104 + 0.8294j,  -0.2649 + 0.7797j],
    #     [-0.7342 + 0.0286j,  -0.0290 - 0.1337j,  -1.1732 - 0.2304j],
    #     [ 0.3743 - 0.1801j,   0.2576 + 0.0612j,  -0.2194 - 0.4121j],
    #     [-1.5677 - 0.3873j,  -0.5400 + 1.0695j,  -0.4384 + 0.5315j],
    #     [ 0.0760 + 1.1855j,   0.3886 + 1.2680j,  -0.7245 - 0.6108j]])

    z_mask = np.random.binomial(size=N, n=1, p= 0.1)
    z_sol = np.random.binomial(size=N, n=1, p= 0.1)
    # z_mask = np.zeros(N)
    # z_sol = np.zeros(N)
    num_offs = np.sum(z_mask*(1-z_sol))

    import time

    t1 = time.time()
    _, w, obj, optimal = solve_relaxed(H=H, z_mask=z_mask, z_sol=z_sol)
    time_taken = time.time()-t1

    print(w)
    print(obj)
    print(optimal)
    print('time taken', time_taken)
    print('this is bf')
    print(H)
    pass