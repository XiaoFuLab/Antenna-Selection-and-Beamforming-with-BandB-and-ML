import numpy as np
import time
from collections import namedtuple
from dataclasses import dataclass
from antenna_selection.observation import Observation
from antenna_selection.bb_unified import BBenv as Environment, DefaultBranchingPolicy, solve_bb
from models.helper import SolverException
from antenna_selection.solve_relaxation_bf import solve_relaxed
from antenna_selection.solve_relaxation_rbf import solve_rsdr
# from antenna_selection.paper_experiments.greedy import greedy

parameter_fields = ('robust_beamforming', 'train_size', 'test_size', 'sigma_sq', 'min_sinr', 'robust_margin', 'num_trials', 'timeout', 'max_problems')
Parameters = namedtuple('Parameters', parameter_fields)
Parameters.__new__.__defaults__ = (None,) * len(Parameters._fields)

TrainParameters = namedtuple('TrainParameters', ['robust_beamforming', 'train_size', 'sigma_sq', 'min_sinr', 'robust_margin'])

INVALID_TOKENS = (np.inf, None, np.nan)

def get_feasibility(power, oracle_power=None):
    '''
    @params:
        power: list of power values for the mehod under consideration
        oracle_power: list of power values obtained from the oracle method
    '''
    if oracle_power is None:
        oracle_power = [1]*len(power)
    total = np.sum([1 for item in oracle_power if item not in INVALID_TOKENS])
    feasibility = np.sum([1 for  (p, oracle_p) in zip(power, oracle_power) if (p not in INVALID_TOKENS and oracle_p not in INVALID_TOKENS)])/total
    return feasibility

def get_ogap(power, oracle_power=None, ):
    '''
    @params:
        power: list of power values for the mehod under consideration
        oracle_power: list of power values obtained from the oracle method
    '''
    assert oracle_power is not None, "Please provide oracle power values"
    ogap = np.mean([(p-oracle_p)/oracle_p*100 for (p, oracle_p) in zip(power, oracle_power) if oracle_p not in INVALID_TOKENS and p not in INVALID_TOKENS ])
    return ogap
 
def get_speedup(time, oracle_time):
    return np.mean([oracle_t/t for (t, oracle_t) in zip(time, oracle_time)])


@dataclass
class OracleArg:
    instance: np.array = None
    max_ant: int = None
    robust_beamforming: bool = None         
    sigma_sq: float = 1.0
    min_sinr: float = 1.0
    robust_margin: float = 0.1

@dataclass
class GreedyArg:
    instance: np.array = None
    max_ant: int = None
    robust_beamforming: bool = None         
    sigma_sq: float = 1.0
    min_sinr: float = 1.0
    robust_margin: float = 0.1
    max_problems: int = np.inf
    timeout: float = None

@dataclass
class CvxArg:
    instance: np.array = None
    max_ant: int = None
    robust_beamforming: bool = None         
    sigma_sq: float = 1.0
    min_sinr: float = 1.0
    robust_margin: float = 0.1
    max_problems: int = np.inf
    timeout: float = None


@dataclass
class MLArgTest:
    instance: np.array = None
    max_ant: int = None
    robust_beamforming: bool = None
    sigma_sq: float = 1.0
    min_sinr: float = 1.0
    robust_margin: float = 0.1
    timeout: float = None
    policy_filepath: str = None
    mask_heuristics: np.array = None
    max_problems: int = np.inf

@dataclass
class MLArgTrain:
    instance: np.array = None
    max_ant: int = None
    w_optimal: np.array = None
    optimal_objective: float = None
    robust_beamforming: bool = None
    file_count: int = None
    sigma_sq: float = 1.0
    min_sinr: float = 1.0
    robust_margin: float = 0.1
    policy_filepath: str = None
    mask_heuristics: np.array = None

    
def solve_bb_pool(arguments):
    try:
        output = solve_bb(instance=arguments.instance, 
                            max_ant=arguments.max_ant, 
                            robust_beamforming=arguments.robust_beamforming,
                            min_sinr=arguments.min_sinr,
                            sigma_sq=arguments.sigma_sq,
                            robust_margin=arguments.robust_margin)
        return {'solution': output['solution'], 'objective': output['objective'], 'time': output['time'], 'num_problems': output['num_problems']}

    except SolverException as e:
        print('Solver Exception: ', e)
        return {'solution':None, 'objective':np.inf, 'time':0, 'num_problems':0}

# def solve_greedy_pool(arguments: GreedyArg):
#     try:
#         output = greedy(H=arguments.instance, 
#                             max_ant=arguments.max_ant, 
#                             robust_beamforming=arguments.robust_beamforming,
#                             min_sinr=arguments.min_sinr,
#                             sigma_sq=arguments.sigma_sq,
#                             robust_margin=arguments.robust_margin,
#                             timeout=arguments.timeout,
#                             max_problems=arguments.max_problems)
#         return {'solution': output['solution'], 'objective': output['objective'], 'time': output['time'], 'num_problems': output['num_problems']}

#     except SolverException as e:
#         print('Solver Exception: ', e)
#         return {'solution':None, 'objective':np.inf, 'time':0, 'num_problems':0}


def solve_ml_pool(arguments: MLArgTest):
    env = Environment(observation_function=Observation, epsilon=0.002)
    env.set_node_select_policy(node_select_policy_path=arguments.policy_filepath)
    
    if arguments.mask_heuristics is not None:
        env.set_heuristic_solutions(arguments.mask_heuristics)

    env.reset(arguments.instance, 
                max_ant=arguments.max_ant, 
                robust_beamforming=arguments.robust_beamforming,
                min_sinr=arguments.min_sinr,
                sigma_sq=arguments.sigma_sq,
                robust_margin=arguments.robust_margin)

    branching_policy = DefaultBranchingPolicy()
    start_time = time.time()
    timestep = 0
    
    while timestep < 1000 and len(env.nodes)>0: 
        if (arguments.timeout is not None and time.time()-start_time > arguments.timeout) or env.bm_solver.get_total_problems()>arguments.max_problems:
            break
        print('model tester timestep {},  U: {}, L: {}'.format(timestep, env.global_U, env.global_L))
        env.fathom_nodes()
        if len(env.nodes) == 0:
            break
        node_id, node_feats, label = env.select_node()
        if len(env.nodes) == 0:
            break
        prune_node = env.prune(node_feats)
        if prune_node:
            env.delete_node(node_id)
            continue
        else:
            branching_var = branching_policy.select_variable(node_feats, env.action_set_indices)
            env.push_children(branching_var, node_id)    
        timestep = timestep+1
    
    result = {'timesteps': timestep,
            'objective':env.global_U,
            'time_taken':time.time()-start_time,
            'global_L':env.global_L,
            'num_problems':env.bm_solver.get_total_problems()}
    return result


def post_process(instance, 
                solution, 
                max_ant: int , 
                min_sinr: float = 1.0, 
                sigma_sq: float = 1.0, 
                robust_margin: float = 0.1,
                robust_beamforming: bool = False) -> float:
    """
    when the solution consists of larger number of antennas than max_ant, 
    select the antennas from the solution that is assigned the highest power when all antennas are turned on
    """
    N, _ = instance.shape 
    if robust_beamforming:
        _, W_sol, obj, solved = solve_rsdr(H=instance, 
                            z_mask=np.ones(N), 
                            z_sol=solution, 
                            min_sinr=min_sinr, 
                            robust_margin=robust_margin,
                            sigma_sq=sigma_sq)
    else:        
        _, W_sol, obj, solved = solve_relaxed(H=instance, 
                            z_mask=np.ones(N), 
                            z_sol=solution, 
                            min_sinr=min_sinr, 
                            sigma_sq=sigma_sq)
    if not solved:
        return {'objective': None, 'solution': solution}
    w_energy = np.linalg.norm(W_sol, 2, axis=1)
    ind = np.argpartition(w_energy, -max_ant)[-max_ant:]
    z_hat = np.zeros(solution.shape)
    z_hat[ind] = 1

    if robust_beamforming:
        _, W_sol, obj, solved = solve_rsdr(H=instance, 
                            z_mask=np.ones(N), 
                            z_sol=z_hat, 
                            min_sinr=min_sinr, 
                            robust_margin=robust_margin,
                            sigma_sq=sigma_sq)
    else:     
        _, _, obj, solved = solve_relaxed(H=instance, 
                        z_mask=np.ones(N), 
                        z_sol=z_hat, 
                        min_sinr=min_sinr, 
                        sigma_sq=sigma_sq)  
    if solved: 
        return {'objective': obj, 'solution': z_hat}
    else:
        return {'objective': None, 'solution': z_hat}

# def get_summary_result(result, param):
#     proposed = {'ogap':None, 'feasibility': None, 'speedup': None, 'num_problems': None, 'time': None, 'relative_ogap': None}
#     ircvxopt = {'ogap':None, 'feasibility': None, 'speedup': None, 'num_problems': None, 'time': None}
#     greedy   = {'ogap':None, 'feasibility': None, 'speedup': None, 'num_problems': None, 'time': None}

#     if 'proposed' in result[param] and len(result[param]['proposed']) > 0:
#         if 'bb' in result[param] and len(result[param]['bb'])> 0:
#             proposed['ogap'] = get_ogap([result[param]['proposed'][j]['objective'] for j in range(len(result[param]['proposed']))], [result[param]['bb'][j]['objective'] for j in range(len(result[param]['bb']))])
#             proposed['feasibility'] = get_feasibility([result[param]['proposed'][j]['objective'] for j in range(len(result[param]['proposed']))], [result[param]['bb'][j]['objective'] for j in range(len(result[param]['bb']))])
#             proposed['num_problems'] = np.mean([result[param]['proposed'][j]['num_problems'] for j in range(len(result[param]['proposed']))])
#             proposed['speedup'] = get_speedup([result[param]['proposed'][j]['time_taken'] for j in range(len(result[param]['proposed']))], [result[param]['bb'][j]['time'] for j in range(len(result[param]['bb']))])
#         else:
#             proposed['ogap'] = np.mean([result[param]['proposed'][j]['objective'] for j in range(len(result[param]['proposed']))])
#             proposed['feasibility'] = get_feasibility([result[param]['proposed'][j]['objective'] for j in range(len(result[param]['proposed']))], list(range(len(result[param]['proposed']))) )
#             proposed['num_problems'] = np.mean([result[param]['proposed'][j]['num_problems'] for j in range(len(result[param]['proposed']))])
#         # proposed['relative_ogap'] = get_ogap([result[param]['proposed'][j]['objective'] for j in range(len(result[param]['proposed'])) if result[param]['cvx_baseline'][j]['objective'] is not None], 
#         #                                   [result[param]['bb'][j]['objective'] for j in range(len(result[param]['cvx_baseline'])) if result[param]['cvx_baseline'][j]['objective'] is not None]))

#     if 'greedy' in result[param] and len(result[param]['greedy']) > 0:
#         if 'bb' in result[param] and len(result[param]['bb'])> 0:
#             greedy['ogap'] = get_ogap([result[param]['greedy'][j]['objective'] for j in range(len(result[param]['greedy']))], [result[param]['bb'][j]['objective'] for j in range(len(result[param]['bb']))])
#             greedy['feasibility'] = get_feasibility([result[param]['greedy'][j]['objective'] for j in range(len(result[param]['greedy']))], [result[param]['bb'][j]['objective'] for j in range(len(result[param]['bb']))])
#             greedy['num_problems'] = np.mean([result[param]['greedy'][j]['num_problems'] for j in range(len(result[param]['greedy']))]))
#             greedy['speedup'] = get_speedup([result[param]['greedy'][j]['time'] for j in range(len(result[param]['greedy']))], [result[param]['bb'][j]['time'] for j in range(len(result[param]['bb']))])
#         else:
#             greedy['num_problems'] = np.mean([result[param]['greedy'][j]['num_problems'] for j in range(len(result[param]['greedy']))]))
#             greedy['ogap'] = np.mean([result[param]['greedy'][j]['objective'] for j in range(len(result[param]['greedy']))]))
#             greedy['feasibility'] = get_feasibility([result[param]['greedy'][j]['objective'] for j in range(len(result[param]['greedy']))], list(range(len(result[param]['proposed'])))))

#     if 'cvx_baseline' in result[param] and len(result[param]['proposed']) > 0:
#         if 'bb' in result[param] and len(result[param]['bb'])> 0:
        
#             ircvxopt['ogap'] = get_ogap([result[param]['cvx_baseline'][j]['objective'] for j in range(len(result[param]['cvx_baseline']))], [result[param]['bb'][j]['objective'] for j in range(len(result[param]['bb']))])
#             ircvxopt['feasibility'] = get_feasibility([result[param]['cvx_baseline'][j]['objective'] for j in range(len(result[param]['cvx_baseline']))], [result[param]['bb'][j]['objective'] for j in range(len(result[param]['bb']))])
#             ircvxopt['num_problems'] = np.mean([result[param]['cvx_baseline'][j]['num_problems'] for j in range(len(result[param]['cvx_baseline']))])
#             ircvxopt['speedup'] = get_speedup([result[param]['cvx_baseline'][j]['num_problems'] for j in range(len(result[param]['cvx_baseline']))], [result[param]['bb'][j]['num_problems'] for j in range(len(result[param]['bb']))])
#         else:
#             ircvxopt['feasibility'] = get_feasibility([result[param]['cvx_baseline'][j]['objective'] for j in range(len(result[param]['cvx_baseline']))], list(range(len(result[param]['proposed']))) )
#             ircvxopt['ogap'] = np.mean([result[param]['cvx_baseline'][j]['objective'] for j in range(len(result[param]['cvx_baseline'])) if result[param]['cvx_baseline'][j]['objective'] is not None])
#             ircvxopt['num_problems'] = np.mean([result[param]['cvx_baseline'][j]['num_problems'] for j in range(len(result[param]['cvx_baseline']))])

#     return {'proposed': proposed, 'greedy': greedy, 'ircvxopt': ircvxopt}