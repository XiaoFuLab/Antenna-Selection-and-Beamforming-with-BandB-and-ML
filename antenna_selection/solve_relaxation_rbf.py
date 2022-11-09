import cvxpy as cp
import numpy as np

import models.helper as helper


def solve_rsdr(H=None, 
                z_mask=None, 
                z_sol=None, 
                min_sinr=1.0, 
                sigma_sq=1.0, 
                robust_margin=0.1):
    """
    z_mask and z_sol determin the antennas that need to be turned off
    if z_mask[n] = 1 and z_sol[n] = 0, then the antenna needs to be turned off

    Returns: z_sol(just for backward compatibility with old code, ignore this), optimal W, objective, bool (whether solution was obtained) 
    """
    N_original, M = H.shape

    sigma_sq= sigma_sq*np.ones(M)
    min_sinr= min_sinr*np.ones(M) #SINR levels, from -10dB to 20dB
    robust_margin= robust_margin*np.ones(M)

    H_short = H.copy()
    for n in range(N_original-1, -1, -1):
        if z_mask[n] and not z_sol[n]:
            H_short = np.concatenate((H_short[:n, :], H_short[n+1:, :]), axis=0)

    num_off_ants = np.sum(z_mask*(1-z_sol))
    assert num_off_ants<N_original, 'number of allowed antennas < 1'
    N = int(N_original - num_off_ants)
    
    X = []
    for i in range(M):
        X.append(cp.Variable((N,N), hermitian=True))

    s = cp.Variable(M)
    t = cp.Variable(M)
    obj = cp.Minimize(cp.real(cp.sum([cp.trace(Xi) for Xi in X])))
    constraints = []
    for m in range(M):
        Q = (1+1/min_sinr[m])*X[m] - cp.sum(X)
        r = Q @ H_short[:,m:m+1]
        s = H_short[:,m:m+1].conj().T @ Q @ H_short[:,m:m+1] - sigma_sq[m:m+1]
        Z = cp.hstack((Q+t[m]*np.eye(N), r))
        # Z = cp.vstack((Z, cp.hstack((cp.conj(cp.transpose(r)), s-t[m:m+1]*robust_margin[m:m+1]**2 ))))
        Z = cp.vstack((Z, cp.hstack((cp.conj(cp.transpose(r)), s-cp.multiply(t[m:m+1],robust_margin[m:m+1]**2) ))))

        constraints += [X[m] >> 0]
        constraints += [Z >> 0]
    constraints += [t >= 0]
    
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(verbose=False)
    except:
        return np.zeros(N_original), np.zeros((N_original,M)), np.inf, False

    if prob.status in ['infeasible', 'unbounded']:
        # print('infeasible antenna solution')
        return np.zeros(N_original), np.zeros((N_original,M)), np.inf, False

    w_sols = []
    for Xi in X:
        evec, rank1 = helper.get_rank1(Xi.value)
        # print('\n the solver solution', w.value, evec)
        if not rank1:
            raise helper.SolverException('solution not rank1')
        # assert rank1, 'solution not rank 1. \n H matrix {} \nz_mask {} \n z_sol {}'.format(H, z_mask, z_sol)
        w_sols.append(np.reshape(evec, (N,1))) 

    W_sol = np.concatenate(w_sols, axis=1)

    # return order: w_solution, objective_value, optimality


    # reshape the solution X into the correct dimension
    for n in range(N_original):
        if z_mask[n] and not z_sol[n]:
            W_sol = np.concatenate((W_sol[:n, :], np.zeros((1,M)), W_sol[n:,:]), axis=0)
    assert W_sol.shape[0] == N_original, 'W not of correct shape'
    # For backward compatibility, also return z_sol
    return z_sol.copy(), W_sol, prob.objective.value, True


def get_rank1(X):
    """
    X: PSD matrix
    """
    assert X.shape[0] == X.shape[1], 'matrix not square'
    eval, evec = np.linalg.eig(X)
    
    rank1 =  np.real(max(eval)/sum(eval)) > 0.99
    principal_id = np.argmax(eval)

    return evec[:,principal_id], rank1


if __name__=='__main__':
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

    num_offs = np.sum(z_mask*(1-z_sol))

    import time

    t1 = time.time()
    _, w, obj, optimal = solve_rsdr(H=H, z_mask=z_mask, z_sol=z_sol)
    time_taken = time.time()-t1

    print(w)
    print(obj)
    print(optimal)
    print('time taken', time_taken)
    print(H)
