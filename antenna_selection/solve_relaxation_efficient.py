""" 
Wrapper for solve_relaxation module. This module implements saving results for all the problems ever solved so that redundant computations can be avoided
"""
from antenna_selection.solve_relaxation_bf import solve_relaxed as perfect_channel_solve
from antenna_selection.solve_relaxation_rbf import solve_rsdr as robust_channel_solve
import numpy as np

class EfficientRelaxation:
    def __init__(self, H=None, 
                gamma=1,
                sigma_sq=1, 
                epsi=0.3,
                robust=False):
        self.robust = robust
        self.H = H.copy()
        self.gamma = gamma
        self.sigma_sq = sigma_sq
        self.epsi = epsi
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
        
    # def _fetch_solution(z_mask=None,
    #                 z_sol=None)
    #     """
    #     Fetch the solution from the data dictionary if already solved. 
    #     Returns:
    #         W_solution, objective_value, optimality, solution_found
    #         None, None, False, False if could not find the solution in the solution dictionary
    #     """
    #     assert z_mask is not None and z_sol is not None, "Fetch solutions: one of the input is None"


    def solve_efficient(self, z_mask=None,
                        z_sol=None):
        assert z_mask is not None and z_sol is not None, "Solve efficient: one of the input is None"
        # print('total problems is {}, total unique problems {}'.format(self.num_problems, self.num_unique_problems))
        self.num_problems += 1
        for i in range(len(self.data['node'])):
            if self._compare_nodes(z_mask.copy(), self.data['node'][i][0], z_sol.copy(), self.data['node'][i][1]):
                # print('matched z_mask_query {}, z_mask_stored {}, z_sol_query {}, z_sol_stored {}'.format(z_mask, self.data['node'][i][0], z_sol, self.data['node'][i][1]))
                # print('solution', self.data['solution'][i])
                # for j in range(len(self.data['node'])):
                #     print(self.data['node'][j], self.data['solution'][j][2])

                # z, W, obj, optimality = robust_channel_solve(H=self.H, 
                #                                         z_mask=z_mask, 
                #                                         z_sol=z_sol)
                # z, W, obj, optimality = robust_channel_solve(H=self.H, 
                #                                         z_mask=self.data['node'][i][0], 
                #                                         z_sol=self.data['node'][i][1])
                
                
                return z_sol.copy(), self.data['solution'][i][1], self.data['solution'][i][2], self.data['solution'][i][3]
                # return z_sol.copy(), self.data['solution'][i][1:]

        self.num_unique_problems += 1
        if self.robust:
            z, W, obj, optimality = robust_channel_solve(H=self.H, 
                                        z_mask=z_mask, 
                                        z_sol=z_sol, 
                                        gamma=self.gamma, 
                                        sigma_sq=self.sigma_sq, 
                                        epsi=self.epsi)
            self._save_solutions(z_mask=z_mask.copy(), 
                                z_sol=z_sol.copy(),
                                z_result = z.copy(),
                                W_sol=W.copy(),
                                obj=obj,
                                optimal=optimality)
            return z, W, obj, optimality
        else:
            print('calling solve relaxed')
            print(z_mask, z_sol)
            z, W, obj, optimality = perfect_channel_solve(H=self.H, 
                                        z_mask=z_mask, 
                                        z_sol=z_sol, 
                                        gamma=self.gamma, 
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

# def solve(H=None, 
#             z_mask=None, 
#             z_sol=None, 
#             gamma=1, )
