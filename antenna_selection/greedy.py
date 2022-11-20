'''
Implementation of the Greedy approach for joint beamforming and antenna selection.
This code works for both beamforming and robust beamforming problem. 
General Procedure:
    R = list of all antennas
    Loop until (N-L) antennas are removed:
        1. Compute the objective for all possible combinations of |R|-1 antennas in R.
        2. Remove the antenna that results in maximum objective (power).
'''

import numpy as np
import pickle
import os
import time
from collections import namedtuple
from dataclasses import dataclass
from antenna_selection.observation import Observation
from antenna_selection.bb_unified import BBenv as Environment, DefaultBranchingPolicy, solve_bb
from models.helper import SolverException
from antenna_selection.baseline_bf import cvx_baseline as bf_cvx_baseline
from antenna_selection.baseline_rbf import ReweightedPenalty
from antenna_selection.solve_relaxation_bf import solve_relaxed
from antenna_selection.solve_relaxation_rbf import solve_rsdr
from antenna_selection.utils import post_process

from collections import namedtuple

def greedy(robust_beamforming=False,
            H=None,
            max_ant=None,
            sigma_sq=1.0,
            min_sinr=1.0,
            robust_margin=0.1, 
            timeout=np.inf,
            max_problems=np.inf,
            random_post_process=False):

    #TODO: Unify the input type (take complex tensor of [N,M])
    start_time = time.time()
    Candidate = namedtuple('Candidate', 'selection, objective')

    N,M = H.shape
    selected_antennas = np.ones((N,), dtype=bool)
    num_problems = 0
    for i in range(N-max_ant):
        print('selecting {} antennas'.format(N-i))
        # Construct the antenna combintions and evaluate the cost
        candidates = []
        for j in range(N):
            if selected_antennas[j] == 0:
                continue

            # specify the mask 
            z = selected_antennas.copy()
            z[j] = 0
            if robust_beamforming:
                _, W_sol, objective, solved = solve_rsdr(H=H,
                                                    z_mask=np.ones(N),
                                                    z_sol=z,
                                                    min_sinr=min_sinr,
                                                    sigma_sq=sigma_sq,
                                                    robust_margin=robust_margin,
                                                    )
            else:
                _, W_sol, objective, solved = solve_relaxed(H=H,
                                                    z_mask=np.ones(N),
                                                    z_sol=z,
                                                    min_sinr=min_sinr,
                                                    sigma_sq=sigma_sq,
                                                    )
            num_problems += 1
            if not solved:
                objective = np.inf
            candidates.append(Candidate(z, objective))
            if num_problems > max_problems or time.time()-start_time > timeout:
                break
        best_candidate = min(candidates, key=lambda x: x.objective)
        selected_antennas = best_candidate.selection
        best_objective = best_candidate.objective
        time_taken = time.time() - start_time
        if num_problems > max_problems or time.time()-start_time > timeout:
            break 
    if np.sum(selected_antennas)>max_ant:
        if not random_post_process:
            post_result = post_process(H,
                                        selected_antennas,
                                        max_ant=max_ant,
                                        sigma_sq=sigma_sq,
                                        min_sinr=min_sinr,
                                        robust_beamforming=robust_beamforming)
            selected_antennas = post_result['solution']
        else:
            indices = np.where(selected_antennas == 1)[0]
            np.random.shuffle(indices)
            selected_antennas[indices[:np.sum(selected_antennas)-max_ant]] = 0

    assert np.sum(selected_antennas) == max_ant, "Number of selected antennas {} != {}".format(np.sum(selected_antennas), max_ant)
    if robust_beamforming:
        _, _, best_objective, solved = solve_rsdr(H=H,
                                                    z_mask=np.ones(N),
                                                    z_sol=selected_antennas,
                                                    min_sinr=min_sinr,
                                                    sigma_sq=sigma_sq,
                                                    robust_margin=robust_margin,
                                                    )
    else:
        _, _, best_objective, solved = solve_relaxed(H=H,
                                                    z_mask=np.ones(N),
                                                    z_sol=selected_antennas,
                                                    min_sinr=min_sinr,
                                                    sigma_sq=sigma_sq
                                                    )

    return {'objective': best_objective, 'solution': selected_antennas, 'num_problems': num_problems, 'time': time_taken}

if __name__=='__main__':
    problem_sizes = [
                    (4,3,2),
                    (8,5,3),
                    (8,6,4),
                    (10,8,6),
                    ]
    
    N,M,L = 10, 6, 6
    robust_beamforming = False
    min_sinr = 10.0
    sigma_sq = 0.1
    robust_margin = 0.1
    # np.random.seed(100)
    H = (np.random.randn(N,M) + 1j*np.random.randn(N,M))/np.sqrt(2)

    greedy_output_random = greedy(robust_beamforming=robust_beamforming,
                            H=H,
                            max_ant=L,
                            sigma_sq=sigma_sq,
                            min_sinr=min_sinr,
                            robust_margin=robust_margin,
                            timeout=2,
                            random_post_process=True
                            )
    
    greedy_output_largest = greedy(robust_beamforming=robust_beamforming,
                            H=H,
                            max_ant=L,
                            sigma_sq=sigma_sq,
                            min_sinr=min_sinr,
                            robust_margin=robust_margin,
                            timeout=2,
                            random_post_process=False
                            )
    print('greedy random', greedy_output_random)
    print('greedy largest', greedy_output_largest)
