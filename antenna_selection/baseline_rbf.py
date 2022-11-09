import cvxpy as cp
import numpy as np
from antenna_selection.solve_relaxation_rbf import *

class ReweightedPenalty:
    def __init__(self, 
                H=None, 
                min_sinr=1, 
                sigma_sq=2, 
                robust_margin=0.3, 
                max_ant= None):
        assert H is not None and max_ant is not None, "required arguments channel matrix H or max_ant L not provided"
        self.H = H.copy()
        # self.H = H[0,::] + 1j*H[1,::]
        self.N, self.M = H.shape
        self.sigma_sq= sigma_sq*np.ones(self.M)
        self.min_sinr= min_sinr*np.ones(self.M) 
        self.robust_margin= robust_margin*np.ones(self.M)

        self.max_ant = max_ant

        self.epsilon = 1e-5
        self.lmbda_lb = 0
        self.lmbda_ub = 1e6
        self.num_sdps = 0

    def solve(self):
        # step 1
        U = np.zeros((self.N, self.N))
        U_new = np.ones((self.N, self.N))
        
        r = 0
        max_iter = 30
        while np.linalg.norm(U-U_new, 'fro')>0.00001 and r < max_iter:
        # while r < max_iter:
            print('sparse iteration  {}'.format(r))
            r += 1
            U = U_new.copy()
            X_tilde = self.sparse_iteration(U)
            self.num_sdps += 1
            if X_tilde is None:
                return {'objective': None, 'solution': mask.copy(), 'num_problems': self.num_sdps}

            a = np.diag(X_tilde)
            mask = (a>0.01)*1
            if mask.sum()<= self.max_ant:
                print('Sparse enough solution found {}'.format(mask.sum()))
                break
            U_new = 1/(X_tilde + self.epsilon)

        prelim_mask = mask.copy()
        before_iter_ant_count = mask.sum()
        if mask.sum() > self.max_ant:
            # sparse enough solution not found!
            return {'objective': None, 'solution': mask.copy(), 'num_problems': self.num_sdps}

        # step 2
        r = 0
        max_iter = 30
        while mask.sum() != self.max_ant and r < max_iter:
            r += 1
            lmbda = self.lmbda_lb + (self.lmbda_ub - self.lmbda_lb)/2
            X_tilde = self.solve_sdps_with_soft_as(lmbda, U_new)
            
            self.num_sdps += 1
            # if X_tilde is None:
            #     return None, np.zeros(self.N), self.num_sdps

            if X_tilde is not None:
                a = np.diag(X_tilde)
                mask = (a>0.01)*1
            print('iteration {}'.format(r), lmbda, mask.sum(), self.lmbda_lb, self.lmbda_ub)
            if mask.sum() == self.max_ant:
                break
            elif mask.sum() > self.max_ant:
                self.lmbda_lb = lmbda
            elif mask.sum() < self.max_ant:
                self.lmbda_ub = lmbda
        if mask.sum()>self.max_ant:
            mask = prelim_mask.copy()    
        print('num selected antennas', mask.sum())

        after_iter_ant_count = mask.sum()
        # step 3
        _, W, obj, optimal = solve_rsdr(H=self.H, 
                                        z_mask=np.ones(self.N), 
                                        z_sol=mask,
                                        sigma_sq=self.sigma_sq,
                                        min_sinr=self.min_sinr,
                                        robust_margin=self.robust_margin
                                        )
        
        if mask.sum() > self.max_ant:
            return {'objective': None, 'solution': mask.copy(), 'num_problems': self.num_sdps}
        return {'objective': obj, 'solution': mask.copy(), 'num_problems': self.num_sdps}

    def sparse_iteration(self, U):
        
        X = []
        for i in range(self.M):
            X.append(cp.Variable((self.N, self.N), hermitian=True))
        X_tilde = cp.Variable((self.N, self.N), complex=False)
        t = cp.Variable(self.M)

        obj = cp.Minimize(cp.trace(U @ X_tilde))
        
        constraints = []
        for m in range(self.M):
            Q = (1+1/self.min_sinr[m])*X[m] - cp.sum(X)
            r = Q @ self.H[:,m:m+1]
            s = self.H[:,m:m+1].conj().T @ Q @ self.H[:,m:m+1] - self.sigma_sq[m:m+1]
            Z = cp.hstack((Q+t[m]*np.eye(self.N), r))
            # Z = cp.vstack((Z, cp.hstack((cp.conj(cp.transpose(r)), s-t[m:m+1]*robust_margin[m:m+1]**2 ))))
            Z = cp.vstack((Z, cp.hstack((cp.conj(cp.transpose(r)), s-cp.multiply(t[m:m+1], self.robust_margin[m:m+1]**2) ))))

            constraints += [X[m] >> 0]
            constraints += [Z >> 0]
            constraints += [X_tilde >= cp.abs(X[m])]
            constraints += [t >= 0]
        
        prob = cp.Problem(obj, constraints)
        prob.solve(verbose=False)

        if prob.status in ['infeasible', 'unbounded']:
            # print('infeasible antenna solution')
            # return np.ones((self.N, self.N))
            return None

        return X_tilde.value


    def solve_sdps_with_soft_as(self, lmbda, U):
        X = []
        for i in range(self.M):
            X.append(cp.Variable((self.N, self.N), hermitian=True))
        X_tilde = cp.Variable((self.N, self.N), complex=False)
        t = cp.Variable(self.M)

        obj = cp.Minimize(cp.real(cp.sum([cp.trace(Xi) for Xi in X])) + lmbda*cp.trace(U @ X_tilde))
        
        constraints = []
        for m in range(self.M):
            Q = (1+1/self.min_sinr[m])*X[m] - cp.sum(X)
            r = Q @ self.H[:,m:m+1]
            s = self.H[:,m:m+1].conj().T @ Q @ self.H[:,m:m+1] - self.sigma_sq[m:m+1]
            Z = cp.hstack((Q+t[m]*np.eye(self.N), r))
            # Z = cp.vstack((Z, cp.hstack((cp.conj(cp.transpose(r)), s-t[m:m+1]*robust_margin[m:m+1]**2 ))))
            Z = cp.vstack((Z, cp.hstack((cp.conj(cp.transpose(r)), s-cp.multiply(t[m:m+1], self.robust_margin[m:m+1]**2) ))))

            constraints += [X[m] >> 0]
            constraints += [Z >> 0]
            constraints += [X_tilde >= cp.abs(X[m])]
            constraints += [t >= 0]
        
        prob = cp.Problem(obj, constraints)

        try:
            prob.solve(verbose=False)
        except:
            return None
        if prob.status in ['infeasible', 'unbounded']:
            # print('infeasible antenna solution')
            # return np.ones((self.N, self.N))
            return None

        return X_tilde.value

if __name__=='__main__':
    N, M, L = 8,4,3

    H = (np.random.randn(N,M) + 1j*np.random.randn(N,M))/np.sqrt(2)

    # H =np.array([[ 0.1414 + 0.3571j,   0.3104 + 0.8294j,  -0.2649 + 0.7797j],
    #     [-0.7342 + 0.0286j,  -0.0290 - 0.1337j,  -1.1732 - 0.2304j],
    #     [ 0.3743 - 0.1801j,   0.2576 + 0.0612j,  -0.2194 - 0.4121j],
    #     [-1.5677 - 0.3873j,  -0.5400 + 1.0695j,  -0.4384 + 0.5315j],
    #     [ 0.0760 + 1.1855j,   0.3886 + 1.2680j,  -0.7245 - 0.6108j]])
    import time

    t1 = time.time()
    iterIns = ReweightedPenalty(H=H, max_ant=L)
    obj, mask = iterIns.solve()
    time_taken = time.time()-t1

    print(obj)
    print('time taken', time_taken)