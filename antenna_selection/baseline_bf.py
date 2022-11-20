import cvxpy as cp
import numpy as np
import time
from antenna_selection.solve_relaxation_bf import solve_relaxed
from antenna_selection.utils import post_process


def cvx_baseline(H, 
            max_ant=5,
            min_sinr=1.0,
            sigma_sq=1.0,
            max_iter=30):
    '''
    Implementation of Mehanna et al., 2013 as a basline
    '''
    start_time = time.time()
    lmbda_lb = 0
    lmbda_ub = 1e6
    # global lmbda_lb, lmbda_ub
    N, _ = H.shape

    # step 1
    u = np.zeros((N,1))
    u_new = np.ones((N,1))
    r = 0
    num_problems = 0
    # while np.linalg.norm(u-u_new)>0.00001 and r < max_iter:
    while r < max_iter:
        print('Sparse iteration  {}'.format(r))
        r += 1
        u = u_new.copy()
        W, _ = sparse_iteration(H, u, sigma_sq=sigma_sq, min_sinr=min_sinr)
        a = np.linalg.norm(W, axis=1)
        mask = (a>0.01)*1
        if mask.sum()<= max_ant:
            print('Sparse enough solution found. Exiting sparse iteration...')
            break
        u_new = 1/(np.linalg.norm(W, axis=1) + 1e-5)
    prelim_mask = mask.copy()

    num_problems = r
    
    if mask.sum() > max_ant:
        post_result = post_process(H,
                                    mask,
                                    max_ant=max_ant,
                                    sigma_sq=sigma_sq,
                                    min_sinr=min_sinr,
                                    robust_beamforming=False)
        return {'objective':post_result['objective'], 'solution': post_result['solution'], 'num_problems':num_problems, 'time': time.time()-start_time}
    # step 2
    r = 0
    max_iter = 50
    while mask.sum() < max_ant and r < max_iter:
        r += 1
        # if mask.sum() < max_ant:
        lmbda = lmbda_lb + (lmbda_ub - lmbda_lb)/2
        try:
            W, _ = sdp_omar(H, lmbda, u_new, sigma_sq=sigma_sq, min_sinr=min_sinr)
        except:
            print('Exception occured')
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

    # if mask.sum() > max_ant:
    #     return {'objective': None, 'solution': mask.copy(), 'num_problems': num_problems, 'time': time.time()-start_time}
    # step 3
    _, W, obj, opt = solve_relaxed(H, z_mask=np.ones(N), z_sol = mask, min_sinr=min_sinr, sigma_sq=sigma_sq)
    print(obj)
    return {'objective': obj, 'solution': mask.copy(), 'num_problems': num_problems, 'time': time.time()-start_time}

def sparse_iteration(H, u, M=1000, sigma_sq=1.0, min_sinr=1.0):
    """
    Solves the relaxed formulation of Omar et al 2013
    """
    # print('z mask: {},\n z value: {}'.format(z_mask, z_sol))
    N, K = H.shape
    W = cp.Variable((N,K), complex=True)

    obj = cp.Minimize((u.T @ cp.norm_inf(W, axis=1)))

    zero = np.zeros(N)
    one = np.ones(N)
    c_1 = (1/np.sqrt(min_sinr*sigma_sq))
    c_2 = (1/sigma_sq)

    constraints = []
    for k in range(K):
        Imask = np.eye(K)
        Imask[k,k] = 0
        constraints += [c_1*cp.real(np.expand_dims(H[:,k], axis=0).conj() @ W[:,k]) >= cp.norm(cp.hstack((c_2*(W @ Imask).H @ H[:,k], np.ones(1))), 2)]

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)
    
    if prob.status in ['infeasible', 'unbounded']:
        print('infeasible solution')
        return None, None, np.inf

    return W.value, np.linalg.norm(W.value, 'fro')**2

def sdp_omar(H, lmbda, u, M=1000, sigma_sq=1.0, min_sinr=1.0):
    """
    Solves the relaxed formulation of Omar et al 2013
    """
    # print('z mask: {},\n z value: {}'.format(z_mask, z_sol))
    N, K = H.shape
    W = cp.Variable((N,K), complex=True)

    obj = cp.Minimize(cp.square(cp.norm(W, 'fro')) + lmbda*(u.T @ cp.norm_inf(W, axis=1)))

    zero = np.zeros(N)
    one = np.ones(N)
    c_1 = (1/np.sqrt(min_sinr*sigma_sq))
    c_2 = (1/sigma_sq)

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
    from antenna_selection.bb_unified import solve_bb
    num_trials = 5
    N, K = 8,4
    max_ant = 4
    sigma_sq=0.1
    min_sinr=10.0

    ogaps = []
    for i in range(num_trials):
        H = (np.random.randn(N, K) + 1j*np.random.randn(N, K))/np.sqrt(2)
        out_cvx = cvx_baseline(H, max_ant=max_ant, sigma_sq=sigma_sq, min_sinr=min_sinr)
        out_bb = solve_bb(H, 
                            max_ant=max_ant,
                            robust_beamforming=False,
                            robust_margin=0.02,
                            min_sinr=min_sinr,
                            sigma_sq=sigma_sq)
        ogaps.append((out_cvx['objective'] - out_bb['objective'])/out_bb['objective']*100)
    print('avg ogap ', np.mean(ogaps))
