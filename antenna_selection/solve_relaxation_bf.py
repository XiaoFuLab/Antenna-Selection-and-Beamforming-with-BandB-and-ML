import cvxpy as cp
import numpy as np
import time

def beamforming(H, z, noise_var=1, min_snr=1):
    N, K = H.shape
    W = cp.Variable((N,K), complex=True)
    obj = cp.Minimize(cp.square(cp.norm(W, 'fro')))

    c_1 = (1/np.sqrt(min_snr*noise_var))
    c_2 = (1/noise_var)

    mask = np.diag(z.squeeze().copy())
    constraints = []
    for k in range(K):
        constraints += [c_1*cp.real(np.expand_dims(H[:,k], axis=0) @ W[:,k]) >= cp.norm(cp.hstack((c_2*(W.H @ mask) @ H[:,k], np.ones(1))), 2)]
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=True)
    return W.value
    
class Beamforming():
    def __init__(self, H, max_ant=None, T=1000, noise_var=1, min_snr=1):
        self.N, self.K = H.shape
        
        self.H = H.copy()
        self.max_ant = max_ant
        
        self.zero = np.zeros(self.N)
        self.one = np.ones(self.N)
        
        self.c_1 = (1/np.sqrt(min_snr*noise_var))
        self.c_2 = (1/noise_var)

        self.z_constr = cp.Parameter(self.N)
        self.z_mask = cp.Parameter(self.N)
        self.T = cp.Parameter()
        self.T.value = T

        self.W = cp.Variable((self.N, self.K), complex=True, name='W')
        self.z = cp.Variable((self.N), complex=False, name='z')
        self.obj = cp.Minimize(cp.square(cp.norm(self.W, 'fro')))

        self.constraints = []
        for k in range(self.K):
            Imask = np.eye(self.K)
            Imask[k,k] = 0
            self.constraints += [self.c_1*cp.real(np.expand_dims(self.H[:,k], axis=0).conj() @ self.W[:,k]) >= cp.norm(cp.hstack((self.c_2*(self.W @ Imask).H @ self.H[:,k], np.ones(1))), 2)]
        self.constraints += [self.z >= self.zero, self.z <= self.one]
        self.constraints += [cp.sum(self.z) == self.max_ant] 
        # self.constraints += [self.z == cp.multiply(self.z_mask,self.z_sol)]
        self.constraints += [cp.multiply(self.z, self.z_mask) == self.z_constr]

        # for n in range(N):
        #     if self.z_mask[n]:
        #         self.constraints += [self.z[n] == self.z_constr[n]]

        for k in range(self.K):
            self.constraints += [cp.real(self.W[:,k]) <=  self.T*self.z]
            self.constraints += [cp.real(self.W[:,k]) >= -self.T*self.z]
            self.constraints += [cp.imag(self.W[:,k]) <=  self.T*self.z]
            self.constraints += [cp.imag(self.W[:,k]) >= -self.T*self.z]


        # self.constraints += [cp.norm(self.W, 2, axis=1) <= self.T*self.z]
        self.prob = cp.Problem(self.obj, self.constraints)


    def solve_beamforming(self, z_mask=None, z_sol=None, W_init=None, z_init=None, T=None):
        if W_init is not None:
            self.W.value = W_init.copy()
        if z_init is not None:
            self.z.value = z_init.copy()
        if T is not None:
            self.T.value = T
        
        self.z_mask.value = z_mask.copy()
        self.z_constr.value = (z_mask*z_sol).copy()
        try:
            self.prob.solve(solver=cp.MOSEK, verbose=False)
        except:
            return None, None, np.inf, False
            
        if self.prob.status in ['infeasible', 'unbounded']:
            print('infeasible solution')
            return None, None, np.inf, False

        return self.z.value, self.W.value, np.linalg.norm(self.W.value, 'fro')**2, True


class BeamformingWithSelectedAntennas():
    def __init__(self, H, max_ant=None, noise_var=1, min_snr=1):
        self.N, self.K = H.shape
        
        self.H = H.copy()
        self.max_ant = max_ant

        self.zero = np.zeros(self.N)
        self.one = np.ones(self.N)
        
        self.c_1 = cp.Parameter()
        self.c_1.value = (1/np.sqrt(min_snr*noise_var))
        self.c_2 = (1/noise_var)

        self.z_constr = cp.Parameter(self.N)
        
        self.W = cp.Variable((self.N, self.K), complex=True, name='W')
        self.obj = cp.Minimize(cp.square(cp.norm(self.W, 'fro')))

        self.constraints = []
        for k in range(self.K):
            Imask = np.eye(self.K)
            Imask[k,k] = 0
            self.constraints += [self.c_1*cp.real(np.expand_dims(self.H[:,k], axis=0).conj() @ cp.multiply(self.W[:,k], self.z_constr)) >= cp.norm(cp.hstack((self.c_2*((self.W @ Imask).H @ cp.diag(self.z_constr)) @ self.H[:,k], np.ones(1))), 2)]
        
        self.prob = cp.Problem(self.obj, self.constraints)


    def solve_beamforming(self, z=None, W_init=None):
        if W_init is not None:
            self.W.value = W_init.copy()
        
        self.z_constr.value = np.round(z.copy())
        
        try:
            self.prob.solve(solver=cp.MOSEK, verbose=False)
        except:
            return None, np.inf, False
        if self.prob.status in ['infeasible', 'unbounded']:
            # print('infeasible antenna solution')
            return None, np.inf, False
        
        # for k in range(self.K):
        #     Imask = np.eye(self.K)
        #     Imask[k,k] = 0
        #     print('constraint', k, (self.c_1*cp.real(np.expand_dims(self.H[:,k], axis=0) @ cp.multiply(self.W.value[:,k], self.z_constr)) ).value, (cp.norm(cp.hstack((self.c_2*((self.W.value @ Imask).conj().T @ cp.diag(self.z_constr)) @ self.H[:,k], np.ones(1))), 2)).value)
        # print('c2', self.c_2)

        return self.W.value.copy(), np.linalg.norm(self.W.value.copy(), 'fro')**2, True


def solve_relaxed(H=None, 
                z_mask=None, 
                z_sol=None, 
                gamma=1, 
                sigma_sq=1):
    """
    Lower bound method that solves a smaller sized subproblem with selected and undecided antennas 
    """
    N_original, M = H.shape

    # sigma_sq= sigma_sq*np.ones(M)
    # gamma= gamma*np.ones(M) #SINR levels, from -10dB to 20dB

    H_short = H.copy()
    for n in range(N_original-1, -1, -1):
        if z_mask[n] and not z_sol[n]:
            H_short = np.concatenate((H_short[:n, :], H_short[n+1:, :]), axis=0)
    # print(H_short)
    num_off_ants = np.sum(z_mask*(1-z_sol))
    assert num_off_ants<N_original, 'number of allowed antennas {} < 1'.format(N_original - num_off_ants)
    N = int(N_original - num_off_ants)

    c_1 = (1/np.sqrt(gamma*sigma_sq))
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
    except:
        return np.zeros(N_original), np.zeros((N_original,M)), np.inf, False
        
    if prob.status in ['infeasible', 'unbounded']:
        # print('infeasible antenna solution')
        return np.zeros(N_original), np.zeros((N_original,M)), np.inf, False


    # for m in range(M):
    #     Imask = np.eye(M)
    #     Imask[m,m] = 0
    #     print('constr', m, (c_1*cp.real(np.expand_dims(H_short[:,m], axis=0).conj() @ W[:,m])).value, (cp.norm(cp.hstack((c_2*((W @ Imask).H) @ H_short[:,m], np.ones(1))), 2)).value)
    

    # print('W', W.value)
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