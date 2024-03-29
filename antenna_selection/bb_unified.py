import torch
import torch.nn as nn
import numpy as np
import time 

from antenna_selection.solve_relaxation_rbf import *
from antenna_selection.solve_relaxation_efficient import EfficientRelaxation
from antenna_selection.observation import Observation

from models.gnn_policy import GNNNodeSelectionPolicy
from models.gnn_dataset import get_graph_from_obs
from models.setting import TASK, DEBUG


class Node(object):
    def __init__(self, z_mask=None, z_sol=None, z_feas=None, W_sol=None, U=False, L=False, depth=0, parent_node=None, node_index = 0):
        '''
        @params: 
            z_mask: vector of boolean, 1 means that the corresponding variable(antenna) is decided (= A U B in the paper)
            z_sol: value of z at the solution of the cvx relaxation (1 if in A and 0 otherwise)
            z_feas: value of z after making z_sol feasible (i.e. boolean with constraint satisfaction)
            U: current global upper bound
            L: current global lower bound
            depth: depth of the node from the root of the BB tree
            node_index: unique index assigned to the node in the BB tree
            parent_node: reference to the parent Node objet
            node_index: unique index to identify the node (and count them)
        TODO: This could have been a named tuple.
        '''
        self.z_mask = z_mask.copy()
        self.z_sol = z_sol.copy()
        self.z_feas = z_feas.copy()
        self.W_sol = W_sol.copy()
        self.U = U
        self.L = L
        self.depth = depth
        self.parent_node = parent_node
        self.node_index = node_index

    def copy(self):
        N,M = self.W_sol.shape
        new_node = Node(z_mask=self.z_mask,
                        z_sol=self.z_sol,
                        z_feas=self.z_feas,
                        W_sol=self.W_sol,
                        U=self.U, 
                        L=self.L, 
                        depth=self.depth, 
                        parent_node=None, 
                        node_index = self.node_index)
        return new_node

class DefaultBranchingPolicy(object):
    '''
    Default Branching Policy: This policy returns the antenna index from the unselected antennas with the maximum power assigned. 
    This is currently using Observation object in order to extract the current solution and the decided antenna set. 
    (change this to Node, so the code is readable and and insensitive to change in Obervation class)
    TODO: Convert it into a function as it no longer requires storing data for future computation.
    '''
    def __init__(self):
        pass

    def select_variable(self, observation, candidates):
        # Fetch W_sol, z_mask (= A U B in the paper) 
        N,M = observation.antenna_features.shape[0], observation.variable_features.shape[0]
        W_sol = observation.edge_features[:,6] + 1j*observation.edge_features[:,7]
        W_sol = W_sol.reshape((N,M))

        z_mask = observation.antenna_features[:, 2]
        z_sol = observation.antenna_features[:,0]

        power_w = np.linalg.norm(W_sol, axis=1)
        
        power_w = (1-z_mask)*power_w 
        return np.argmax(power_w)


class BBenv(object):
    def __init__(self, observation_function=Observation, node_select_policy_path='default', epsilon=0.001):
        '''
        Initializes a B&B environment.
        For solving several B&B problem instances, one needs to call the reset function with the problem instance parameters
        @params: 
            observation_function: What kind of features to use (Linear for SVM and fully connecte, or Graph for GNN)
                                    Graph based features is represented by Observation class.
            node_select_policy_path: one of {'default', 'oracle', policy_params}
                                     if the value is 'oracle', optimal solution should be provided in the reset function
                                     policy_params refers to the actual state_dict of the policy network
                                     appropriate policy_type should be given according the policy parameters provided in this argument
            epsilon: The maximum gap between the global upper bound and global lower bound for the termination of the B&B algorithm.
        '''
        self._is_reset = None
        self.epsilon = epsilon # stopping criterion 
        self.H = None
        
        self.nodes = []     # list of problems (nodes)
        self.num_nodes = 0
        self.num_active_nodes = 0
        self.all_nodes = []      # list of all nodes to serve as training data for node selection policy
        self.optimal_nodes = []
        self.node_index_count = 0

        self.L_list = []    # list of lower bounds on the problem
        self.U_list = []    # list of upper bounds on the problem
        
        self.global_L = np.nan # global lower bound
        self.global_U = np.nan  # global upper bound        

        self.action_set_indices = None 
        self.active_node = None # current active node

        self.global_U_ind = None
        self.failed_reward = -2000

        self.node_select_model = None

        self.init_U = 999999
        self.node_select_policy = self.default_node_select        
        
        self.z_incumbent = None
        self.W_incumbent = None
        self.current_opt_node = None
        self.min_bound_gap = None

        if node_select_policy_path == 'default':
            self.node_select_policy = self.default_node_select
        elif node_select_policy_path == 'oracle':
            self.node_select_policy = self.oracle_node_select
        else:
            self.node_select_model = GNNNodeSelectionPolicy()
            self.node_select_model.load_state_dict(torch.load(node_select_policy_path))
            self.node_select_policy = self.learnt_node_select

                
        self.observation_function = observation_function
        self.include_heuristic_solutions = False
        self.heuristic_solutions = []
        
        self.bm_solver = None
        
    def set_heuristic_solutions(self, solution):
        '''
        This method is used to help BB not prune the nodes that contain the solution provided by heuristic methods.
        Provide antenna selections computed by heuristic methods in order to incorporate them into the BB
        '''
        self.include_heuristic_solutions = True
        self.heuristic_solutions.append(solution)


    def reset(self, 
                instance, 
                max_ant,  
                oracle_opt=None, 
                robust_beamforming=False, 
                min_sinr=1.0, 
                sigma_sq=1.0, 
                robust_margin=0.1):
        '''
        Solve new problem instance with given max_ant, min_sinr, sigma_sq, and robust_margin
        '''
        # clear all variables
        self.H = None
        self.nodes = []  # list of problems (nodes)
        self.all_nodes = []
        self.optimal_nodes = []
        self.node_index_count = 0

        self.L_list = []    # list of lower bounds on the problem
        self.U_list = []    # list of upper bounds on the problem
        self.global_L = np.nan # global lower bound
        self.global_U = np.nan  # global upper bound        
        self.action_set_indices = None 
        self.active_node = None
        self.global_U_ind = None
        self.num_nodes = 1

        self.H = instance

        # EfficientRelaxation saves the solutions so that the same lower or upper bound problem is not solved twice
        self.bm_solver = EfficientRelaxation(H=self.H,
                            robust_margin=robust_margin,
                            min_sinr=min_sinr,
                            sigma_sq=sigma_sq,
                            robust_beamforming=robust_beamforming)

        self.min_bound_gap = np.ones(self.H.shape[-1])*0.01
        
        self.max_ant = max_ant

        # number of transmitters and users
        self.N, self.M = self.H.shape 
        self._is_reset = True
        self.action_set_indices = np.arange(1,self.N)

        z_mask = np.zeros(self.N)
        # values of z (selection var) at the z_mask locations
        # for the root node it does not matter
        z_sol = np.zeros(self.N)

        [z, W, lower_bound, optimal] = self.bm_solver.solve_efficient(z_mask=z_mask, z_sol=z_sol)

        self.global_L = lower_bound
        self.z_incumbent = self.get_feasible_z(W_sol=W, 
                                            z_sol=z,
                                            z_mask=z_mask,
                                            max_ant=self.max_ant)

        [_, W_feas, self.global_U, optimal] = self.bm_solver.solve_efficient(z_mask=np.ones(self.N), z_sol=self.z_incumbent)

        if not self.global_U == np.inf:
            self.W_incumbent = W_feas.copy()
        else:
            self.W_incumbent = np.zeros(self.H.shape)

        self.active_node = Node(z_mask=z_mask, z_sol=z, z_feas=self.z_incumbent, W_sol = W, U=self.global_U, L=lower_bound, depth=1, node_index=self.node_index_count) 
        self.current_opt_node = self.active_node
        
        self.active_node_index = 0
        self.nodes.append(self.active_node)
        self.L_list.append(lower_bound)
        self.U_list.append(self.global_U)
        self.all_nodes.append(self.active_node)

        if oracle_opt is not None:
            self.oracle_opt = oracle_opt
        else:
            self.oracle_opt = np.zeros(self.N)

    def push_children(self, var_id, node_id, parallel=False):
        '''
        Creates two children and appends it to the node list. Also executes fathom condition.
        @params:
            var_id: selected variable (in our case, antenna) to branch on
            node_id: selected node to branch on
            parallel: whether to run the node computations in parallel
        '''
        self.delete_node(node_id)
        if var_id == None:
            return
        
        if sum(self.active_node.z_mask*self.active_node.z_sol) == self.max_ant:
            print('\n #####################')
            print('current node is already determined')
            print()
            return 

        max_possible_ant = sum(self.active_node.z_mask*self.active_node.z_sol) + sum(1-self.active_node.z_mask)
        if max_possible_ant < self.max_ant:
            # this condition should never occur (node would be infeasible, sum(z) != L)
            print('\n*******************')
            print('exception: max antenna possible < L')
            print(self.active_node.z_mask)
            print(self.active_node.z_sol)
            return 

        elif max_possible_ant == self.max_ant:
            # this condition should also never occur
            print('\n*******************')
            print('exception: max antenna possible = L')
            self.active_node.z_sol = self.active_node.z_mask*self.active_node.z_sol + (1-self.active_node.z_mask)*np.ones(self.N)
            self.active_node.z_mask = np.ones(self.N)
            return
            
        else:
            z_mask_left = self.active_node.z_mask.copy()
            z_mask_left[var_id] = 1

            z_mask_right = self.active_node.z_mask.copy()
            z_mask_right[var_id] = 1

            z_sol_left = self.active_node.z_sol.copy()
            z_sol_left[var_id] = 0

            z_sol_right = self.active_node.z_sol.copy()
            z_sol_right[var_id] = 1

            if sum(z_sol_right*z_mask_right) == self.max_ant:
                z_sol_right = z_sol_right*z_mask_right
                z_mask_right = np.ones(self.N)

        children_sets = []
        children_sets.append([z_mask_left.copy() , z_sol_left.copy()])
        children_sets.append([z_mask_right.copy() , z_sol_right.copy()])
        # children_sets[0].append(1)
        # children_sets[1].append(2)

        if DEBUG:
            print('expanding node id {}, children {}, lb {}, z_inc {}'.format(self.active_node.node_index, (self.active_node.z_mask, self.active_node.z_sol), self.active_node.L, self.z_incumbent))
        
        children_stats = []
        t1 = time.time()
        for subset in children_sets:
            if DEBUG:
                print('\n creating children {}'.format(subset))
            children_stats.append(self.create_children(subset))
        if DEBUG:
            print('time taken by loop {}'.format(time.time()-t1))

        for stat in children_stats:
            U, L, _, _, new_node = stat
            if new_node is not None:
                self.L_list.append(L)
                self.U_list.append(U)
                self.nodes.append(new_node)
                self.all_nodes.append(new_node)
        
        if len(self.nodes) == 0:
            if DEBUG:
                print('all nodes exhausted')
            return

        # Update the global upper and lower bound 
        # update the incumbent solutions
        min_L_child = min([children_stats[i][1] for i in range(len(children_stats))])
        self.global_L = min(min(self.L_list), min_L_child)
        min_U_index = np.argmin([children_stats[i][0] for i in range(len(children_stats))])
        if self.global_U > children_stats[min_U_index][0]:
            # print('node depth at global U update {}'.format(self.active_node.depth + 1))
            self.global_U = children_stats[min_U_index][0] 
            self.z_incumbent = children_stats[min_U_index][2].copy()
            self.W_incumbent = children_stats[min_U_index][3].copy()

    def create_children(self, constraint_set):
        '''
        Create the Node with the constraint set
        Compute the local lower and upper bounds 
        return the computed bounds to the calling function to update
        '''
        z_mask, z_sol = constraint_set 
        
        # check if the maximum number of antennas are already selected or all antennas are already assigned (z is fully assigned)
        if np.sum(z_mask*np.round(z_sol))==self.max_ant or np.sum(z_mask*(1 - np.round(z_sol))) == len(z_mask) - self.max_ant:
            if np.sum(z_mask*np.round(z_sol))==self.max_ant:
                z_sol = np.round(z_sol)*z_mask
            elif np.sum(z_mask*(1 - np.round(z_sol))) == len(z_mask) - self.max_ant:
                z_sol = np.round(z_sol)*z_mask + (1-np.round(z_sol))*(1-z_mask)

            [_, W, L, optimal] = self.bm_solver.solve_efficient(z_mask=np.ones(self.N), z_sol=z_sol)

            # check this constraint
            if not optimal:
                print('antennas: {} not optimal, may be infeasible'.format(None))
                return np.inf, np.inf, np.zeros(self.N), np.zeros(self.N), None

            assert L >= self.active_node.L - self.epsilon, 'selected antennas: lower bound of child node less than that of parent'

            z_feas = z_sol.copy()

            U = self.get_objective(W, z_feas)

            # create and append node
            self.node_index_count += 1
            new_node = Node(z_mask=z_mask,
                            z_sol=z_sol,
                            z_feas=z_feas,
                            W_sol=W,
                            U=U,
                            L=L,
                            depth=self.active_node.depth+1,
                            node_index=self.node_index_count
                            )
            return U, L, z_feas, W, new_node

        elif np.sum(z_mask*np.round(z_sol))>self.max_ant:
            return np.inf, np.inf, np.zeros(self.N), np.zeros(self.N), None

        else:
            [z, W, L, optimal] = self.bm_solver.solve_efficient(z_sol=z_sol, z_mask=z_mask)
                                                         
            if not optimal:
                if DEBUG:
                    print('relaxed: not optimal', z,L,optimal)
                else:
                    print('relaxed: not optimal, may be infeasible')
                return np.inf, np.inf, np.zeros(self.N), np.zeros(self.N), None

            if L < self.active_node.L - self.epsilon:
                print('child node', constraint_set, L)   
                print('parent node', self.active_node.z_mask, self.active_node.z_sol, self.active_node.l_angle, self.active_node.u_angle, self.active_node.L)                                             
                print(self.H)
            
            assert L >= self.active_node.L - self.epsilon, 'relaxed: lower bound of child node less than that of parent'

            if not L == np.inf:
                z_feas = self.get_feasible_z(W_sol=W,
                                            z_sol=z,
                                            z_mask=z_mask,
                                            max_ant=self.max_ant)

                [_, W_feas, L_feas_relaxed, optimal] =  self.bm_solver.solve_efficient(z_mask=np.ones(self.N), z_sol=z_feas)
            
                if optimal:
                    U = self.get_objective(W_feas, z_feas)
                else:
                    U = np.inf

                # create and append node
                self.node_index_count += 1
                new_node = Node(z_mask=z_mask,
                                z_sol=z,
                                z_feas=z_feas,
                                W_sol=W,
                                U=U,
                                L=L,
                                depth=self.active_node.depth+1,
                                node_index=self.node_index_count
                                )
                                                                                    
                return U, L, z_feas, W_feas, new_node
            
            else:
                return np.inf, np.inf, np.zeros(self.N), np.zeros(self.N), None

    def get_objective(self, W, z_feas):
        return np.linalg.norm(W*np.expand_dims(z_feas, 1), 'fro')**2

    def set_node_select_policy(self, node_select_policy_path='default'):
        '''
        what policy to use for node selection
        @params: 
            node_select_policy_path: one of ('default', 'oracle', gnn_node_policy_parameters)
                                        'default' -> use the lowest lower bound first policy
                                        'oracle' -> select the optimal node (optimal solution should be provided in the reset function)
                                        gnn_node_policy_parameters -> If neither of the above two arguments, this method assumes 
                                            that gnn classifier parameters have been provided
        '''
        if node_select_policy_path=='default':
            self.node_select_policy = 'default'
        elif node_select_policy_path == 'oracle':
            self.node_select_policy = 'oracle'
        else:
            self.node_select_model = GNNNodeSelectionPolicy()
            if DEBUG:
                print('setting policy path', node_select_policy_path)
            model_state_dict = torch.load(node_select_policy_path)
            self.node_select_model.load_state_dict(model_state_dict)
            self.node_select_policy = 'ml_model'

    def select_variable_default(self):
        '''
        Currently this method is not being used for default variable selection. 
        '''
        z_sol_rel = (1-self.active_node.z_mask)*(np.abs(self.active_node.z_sol - 0.5))
        return np.argmax(z_sol_rel)

    def select_node(self):
        '''
        Default node selection method
        TODO: the fathom method has been moved from here. So the loop is not needed
        '''
        node_id = 0
        node_id = self.rank_nodes()
        self.active_node = self.nodes[node_id]
        return node_id, self.observation_function().extract(self), self.is_optimal(self.active_node)


    def prune(self, observation):
        if isinstance(observation, Observation):
            observation = get_graph_from_obs(observation, self.action_set_indices)
        if self.node_select_policy == 'oracle':
            return not self.is_optimal(self.active_node)
        elif self.node_select_policy == 'default':
            return False
        else:
            # out = self.node_select_model(observation.antenna_features, observation.edge_index, observation.edge_attr, observation.variable_features) 
            # out = out.sum()
            # out = self.sigmoid(out) 
            if self.include_heuristic_solutions:
                heuristic_match = self.contains_heuristic(self.active_node)
                if heuristic_match:
                    return False

            with torch.no_grad():
                out = self.node_select_model(observation, 1)

            if out < 0.5:
                # print('prune')
                return True
            else:
                # print('select')
                return False

    def rank_nodes(self):
        return np.argmin(self.L_list)

    def fathom_nodes(self):
        del_ind = np.argwhere(np.array(self.L_list) > self.global_U + self.epsilon)
        if len(del_ind)>0:
            del_ind = sorted(list(del_ind.squeeze(axis=1)))
            for i in reversed(del_ind):
                # print('fathomed nodes')
                self.delete_node(i)
        
    def fathom(self, node_id):
        if self.nodes[node_id].L > self.global_U:
            self.delete_node(node_id)
            return True
        return False

    def delete_node(self, node_id):
        del self.nodes[node_id]
        del self.L_list[node_id]
        del self.U_list[node_id]

    def is_optimal(self, node, oracle_opt=None):
        if oracle_opt is None:
            oracle = self.oracle_opt
        else:
            oracle = oracle_opt
        if np.linalg.norm(node.z_mask*(node.z_sol - oracle)) < 0.0001:
            return True
        else:
            return False

    def contains_heuristic(self, node):
        contains = False
        for heuristic_sol in self.heuristic_solutions:
            if np.linalg.norm(node.z_mask*(node.z_sol - heuristic_sol)) < 0.0001:
                contains = True
                break
        return contains

    def is_terminal(self):
        if (self.global_U - self.global_L)/abs(self.global_U) < self.epsilon:
            return True
        else:
            return False

    def default_node_select(self):
        '''
        Use the node with the lowest lower bound
        '''
        return np.argmin(self.L_list)

    @staticmethod
    def get_feasible_z(W_sol=None, z_mask=None, z_sol=None, max_ant=None):
        '''
        Selects the antennas that have been assigned the maximum power in the solution of W
        '''

        power_w = np.linalg.norm(W_sol, axis=1)
        power_w = (1-z_mask)*power_w
        used_ant = int(np.sum(z_mask*z_sol))
        assert used_ant <= max_ant, 'used antennas already larger than max allowed antennas'
        if used_ant == max_ant:
            return z_mask*z_sol

        z_feas = z_mask*z_sol

        # test the effect of power_w
        # power_w = np.random.permutation(power_w)

        z_feas[np.flip(np.argsort(power_w))[:max_ant-used_ant]] = 1 
        return z_feas

def solve_bb(instance, 
                max_ant=5, 
                max_iter=10000, 
                policy_type='gnn',  
                robust_beamforming=False, 
                robust_margin=None,
                min_sinr=1.0,
                sigma_sq=1.0):
    t1 = time.time()
    if policy_type == 'default':
        env = BBenv(observation_function=Observation, epsilon=0.01)
    elif policy_type == 'gnn':
        env = BBenv(observation_function=Observation, epsilon=0.01)
    elif policy_type == 'oracle':
        env = BBenv(observation_function=Observation, epsilon=0.01)
        pass

    branching_policy = DefaultBranchingPolicy()

    t1 = time.time()

    env.reset(instance, 
                max_ant=max_ant, 
                robust_beamforming=robust_beamforming, 
                robust_margin=robust_margin,
                min_sinr=min_sinr,
                sigma_sq=sigma_sq)
    timestep = 0
    done = False
    lb_list = []
    ub_list = []
    print('\ntimestep', timestep, env.global_U, env.global_L)

    while timestep < max_iter and len(env.nodes)>0 and not done:
        
        env.fathom_nodes()
        if len(env.nodes) == 0:
            break
        node_id, node_feats, label = env.select_node()
        
        if len(env.nodes) == 0:
            break

        branching_var = branching_policy.select_variable(node_feats, env.action_set_indices)
        done = env.push_children(branching_var, node_id, parallel=False)
        timestep = timestep+1
        lb_list.append(env.global_L)
        ub_list.append(env.global_U)
        print('\ntimestep: {}, global U: {}, global L: {}'.format(timestep, env.global_U, env.global_L))
        if env.is_terminal():
            break
    return {'solution': env.z_incumbent.copy(), 'objective': env.global_U, 'time': time.time()-t1, 'num_problems': env.bm_solver.get_total_problems()}

if __name__ == '__main__':
    np.random.seed(seed = 100)
    robust_beamforming = True
    # if TASK == 'beamforming':
    #     robust_beamforming = False
    N = 8
    M = 4
    max_ant = 4
    min_sinr = 10.0
    sigma_sq = 0.1
    robust_margin = 0.01

    u_avg = 0
    t_avg = 0
    tstep_avg = 0
    for i in range(1):
        instance = (np.random.randn(N, M) + 1j*np.random.randn(N,M))/np.sqrt(2)
        _, global_U, t, num_problems = solve_bb(instance, 
                                                max_ant=max_ant, 
                                                max_iter = 10000, 
                                                robust_beamforming=robust_beamforming,
                                                min_sinr=min_sinr,
                                                sigma_sq=sigma_sq,
                                                robust_margin=robust_margin)
        u_avg += global_U
        t_avg += t

    print('\nAverage global U: {} avg time: {}, avg num problems: {}'.format(u_avg, t_avg, num_problems))

