""" 
Wrapper for solve_relaxation module. This module implements saving results for all the problems ever solved so that redundant computations can be avoided
"""
from antenna_selection.solve_relaxation_bf import solve_relaxed as perfect_channel_solve
from antenna_selection.solve_relaxation_rbf import solve_rsdr as robust_channel_solve
import numpy as np

class EfficientRelaxation:
    def __init__(self, H=None, 
                min_sinr=1,
                sigma_sq=1, 
                robust_margin=0.1,
                robust_beamforming=False):
        self.robust_beamforming = robust_beamforming
        self.H = H.copy()
        self.min_sinr = min_sinr
        self.sigma_sq = sigma_sq
        self.robust_margin = robust_margin
        self.data = {}
        self.data['node'] = []
        self.data['solution'] = []
        self.num_problems = 0
        self.num_unique_problems = 0

    def _save_solutions(self, z_mask=None, 
                        z_sol=None,
                        z_result=None, 
                        W_sol=None,
                        obj=None,
                        optimal=None):
        """
        Stores the solutions in RAM as a dictionary

        Does not save duplicate solutions. For example if the node is already present in the data, it does not store.
        """
        assert z_mask is not None and z_sol is not None, "Save solutions: one of the input is None"
        assert len(self.data['node']) == len(self.data['solution'])
        self.data['node'].append((z_mask.copy(), z_sol.copy()))
        self.data['solution'].append((z_result.copy(), W_sol.copy(), obj, optimal))

    def print_nodes(self):
        for item in self.data['node']:
            print(item[0]*(1-item[1]))

    @staticmethod
    def _compare_nodes(z_mask_query, z_mask, z_sol_query, z_sol):
        """
        returns True if the two nodes are equivalent
        """
        # ensure that they are integral
        if np.sum(z_mask*(1-z_mask_query)) == 0:   # check if z_mask is a subset of z_mask_query (A \subseteq A_query)
            if np.sum(np.abs(z_sol_query*z_mask - z_sol*z_mask)) == 0: # on set A values of y should be the same
                remaining_antennas = z_mask_query - z_mask
                if np.sum(z_sol_query*remaining_antennas) == np.sum(remaining_antennas): # values of y_query should be 1 in the set A_query\A
                    return True
        return False

    def solve_efficient(self, z_mask=None,
                        z_sol=None):
        '''
        Wrapper for solving the relaxed problems for BF and RBF
        First checks whether an equivalent node problem has already been solved.
        If so, it returns the stored solution, otherwise, it computes the new solution.
        '''
        assert z_mask is not None and z_sol is not None, "Solve efficient: one of the input is None"

        self.num_problems += 1
        for i in range(len(self.data['node'])):
            if self._compare_nodes(z_mask.copy(), self.data['node'][i][0], z_sol.copy(), self.data['node'][i][1]):
                return z_sol.copy(), self.data['solution'][i][1], self.data['solution'][i][2], self.data['solution'][i][3]

        self.num_unique_problems += 1
        if self.robust_beamforming:
            z, W, obj, optimality = robust_channel_solve(H=self.H, 
                                        z_mask=z_mask, 
                                        z_sol=z_sol, 
                                        min_sinr=self.min_sinr, 
                                        sigma_sq=self.sigma_sq, 
                                        robust_margin=self.robust_margin)
            self._save_solutions(z_mask=z_mask.copy(), 
                                z_sol=z_sol.copy(),
                                z_result = z.copy(),
                                W_sol=W.copy(),
                                obj=obj,
                                optimal=optimality)
            return z, W, obj, optimality
        else:
            z, W, obj, optimality = perfect_channel_solve(H=self.H, 
                                        z_mask=z_mask, 
                                        z_sol=z_sol, 
                                        min_sinr=self.min_sinr, 
                                        sigma_sq=self.sigma_sq)
            self._save_solutions(z_mask=z_mask.copy(), 
                                z_sol=z_sol.copy(),
                                z_result = z.copy(),
                                W_sol=W.copy(),
                                obj=obj,
                                optimal=optimality)
            return z, W, obj, optimality
            
    def get_total_problems(self):
        return self.num_unique_problems
