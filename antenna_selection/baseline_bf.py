import cvxpy as cp
import numpy as np
# from beamforming import solve_beamforming_with_selected_antennas
from antenna_selection.solve_relaxation_bf import solve_relaxed



def as_omar(H, max_ant=5):
    lmbda_lb = 0
    lmbda_ub = 1e6
    # global lmbda_lb, lmbda_ub
    N,K = H.shape

    # step 1
    u = np.zeros((N,1))
    u_new = np.ones((N,1))
    r = 0
    max_iter = 30
    num_problems = 0
    while np.linalg.norm(u-u_new)>0.0001 and r < max_iter:
    # while r < max_iter:
        print('sparse iteration  {}'.format(r))
        r += 1
        u = u_new.copy()
        W, _ = sparse_iteration(H, u)
        a = np.linalg.norm(W, axis=1)
        mask = (a>0.01)*1
        if mask.sum()<= max_ant:
            print('exiting here')
            break
        u_new = 1/(np.linalg.norm(W, axis=1) + 1e-5)
    prelim_mask = mask.copy()

    num_problems = r
    # if mask.sum() > max_ant:
    #     return
    before_iter_ant_count = mask.sum()
    if mask.sum() > max_ant:
        return None, mask.copy(), num_problems
    # step 2
    r = 0
    max_iter = 50
    while mask.sum() != max_ant and r < max_iter:
        r += 1
        # if mask.sum() < max_ant:
        lmbda = lmbda_lb + (lmbda_ub - lmbda_lb)/2
        try:
            W, _ = sdp_omar(H, lmbda, u_new)
        except:
            print('here inside exception')
            break
        a = np.linalg.norm(W, axis=1)
        mask = (a>0.01)*1
        print('iteration {}'.format(r), lmbda, mask.sum(), lmbda_lb, lmbda_ub)
        if mask.sum() == max_ant:
            break
        elif mask.sum() > max_ant:
            lmbda_lb = lmbda
        elif mask.sum() < max_ant:
            lmbda_ub = lmbda
    if mask.sum()>max_ant:
        mask = prelim_mask.copy()    
    print('num selected antennas', mask.sum())
    num_problems += r
    after_iter_ant_count = mask.sum()

    # step 3
    _, W, obj, opt = solve_relaxed(H, z_mask=np.ones(N), z_sol = mask)
    print(obj)
    print('Before lambda iteration: {}'.format(before_iter_ant_count))
    print('After lambda iteration: {}'.format(after_iter_ant_count))
    if mask.sum() > max_ant:
        return None, mask.copy(), num_problems
    return obj, mask.copy(), num_problems

def sparse_iteration(H, u, M=1000, noise_var=1, min_snr=1):
    """
    Solves the relaxed formulation of Omar et al 2013
    """
    # print('z mask: {},\n z value: {}'.format(z_mask, z_sol))
    N, K = H.shape
    W = cp.Variable((N,K), complex=True)

    obj = cp.Minimize((u.T @ cp.norm_inf(W, axis=1)))

    zero = np.zeros(N)
    one = np.ones(N)
    c_1 = (1/np.sqrt(min_snr*noise_var))
    c_2 = (1/noise_var)

    constraints = []
    for k in range(K):
        Imask = np.eye(K)
        Imask[k,k] = 0
        constraints += [c_1*cp.real(np.expand_dims(H[:,k], axis=0).conj() @ W[:,k]) >= cp.norm(cp.hstack((c_2*(W @ Imask).H @ H[:,k], np.ones(1))), 2)]

    # for k in range(K):
    #     constraints += [c_1*cp.real(np.expand_dims(H[:,k], axis=0) @ W[:,k]) >= cp.norm(cp.hstack((c_2*W.H @ H[:,k], np.ones(1))), 2)]
        
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)
    
    if prob.status in ['infeasible', 'unbounded']:
        print('infeasible solution')
        return None, None, np.inf

    return W.value, np.linalg.norm(W.value, 'fro')**2

def sdp_omar(H, lmbda, u, M=1000, noise_var=1, min_snr=1):
    """
    Solves the relaxed formulation of Omar et al 2013
    """
    # print('z mask: {},\n z value: {}'.format(z_mask, z_sol))
    N, K = H.shape
    W = cp.Variable((N,K), complex=True)

    obj = cp.Minimize(cp.square(cp.norm(W, 'fro')) + lmbda*(u.T @ cp.norm_inf(W, axis=1)))

    zero = np.zeros(N)
    one = np.ones(N)
    c_1 = (1/np.sqrt(min_snr*noise_var))
    c_2 = (1/noise_var)

    constraints = []
    for k in range(K):
        Imask = np.eye(K)
        Imask[k,k] = 0
        constraints += [c_1*cp.real(np.expand_dims(H[:,k], axis=0).conj() @ W[:,k]) >= cp.norm(cp.hstack((c_2*(W @ Imask).H @ H[:,k], np.ones(1))), 2)]

    # for k in range(K):
    #     constraints += [c_1*cp.real(np.expand_dims(H[:,k], axis=0) @ W[:,k]) >= cp.norm(cp.hstack((c_2*W.H @ H[:,k], np.ones(1))), 2)]
        
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)
    
    if prob.status in ['infeasible', 'unbounded']:
        print('infeasible solution')
        return None, np.inf

    return W.value, np.linalg.norm(W.value, 'fro')**2

if __name__=='__main__':
    N, K = 8,8
    max_ant = 4
    H = (np.random.randn(N, K) + 1j*np.random.randn(N, K))/np.sqrt(2)
    as_omar(H, max_ant=max_ant)
    # W, obj = sdp_omar(H, 50, np.ones((N,1)))
    # print(np.linalg.norm(W, axis=1), obj)
