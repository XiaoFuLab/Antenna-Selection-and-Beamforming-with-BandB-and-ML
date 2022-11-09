

from antenna_selection.observation import Observation
from antenna_selection.bb_unified import BBenv as Environment, DefaultBranchingPolicy, solve_bb

import numpy as np

import gzip
import pickle
from pathlib import Path
import time
import os
from torch.multiprocessing import Pool
from models.setting import TASK
from models.helper import SolverException
from torch.utils.data import Dataset
from antenna_selection.utils import MLArgTrain, OracleArg, TrainParameters

MAX_STEPS = 10000

class OracleDataset(Dataset):
    def __init__(self, root=None):
        self.sample_files = [str(path) for path in Path(root).glob('sample_*.pkl')]
        self.save_file_index = len(self.sample_files)
        self.fetch_file_index = 0
        self.root = root

    def re_init(self):
        self.sample_files = [str(path) for path in Path(self.root).glob('sample_*.pkl')]
        self.save_file_index = len(self.sample_files)
        self.fetch_file_index = 0   

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)
            # Sample expected of format H, solution, objective, timesteps, time
            return sample

    def get_batch(self, batch_size):
        sample_list = []
        if not len(self.sample_files) - self.fetch_file_index >= batch_size:
            return None
        for i in range(self.fetch_file_index, self.fetch_file_index + batch_size):
            sample_list.append(self.__getitem__(i))
        self.fetch_file_index += batch_size
        return zip(*sample_list)

    def get_batch_from_indices(self, index_list):
        sample_list = []
        for i in index_list:
            sample_list.append(self.__getitem__(i))
        return zip(*sample_list)


    def save_batch(self, 
                    instances, 
                    optimal_solution, 
                    optimal_obj,
                    oracle_steps,
                    oracle_time):
        
        for i in range(len(instances)):
            self.put((instances[i], optimal_solution[i], optimal_obj[i], oracle_steps[i], oracle_time[i]))

    def put(self, sample):
        with gzip.open(os.path.join(self.root, 'sample_{}.pkl'.format(self.save_file_index)), 'wb') as f:
            pickle.dump(sample, f)
        self.save_file_index += 1


class DataCollect(object):
    def __init__(self, 
                observation_function=Observation,
                parameters: TrainParameters = None,
                train_filepath=None, 
                policy_type='gnn', 
                oracle_solution_filepath=None,
                num_instances=10):

        self.observation_function = observation_function
        self.parameters = parameters

        self.branching_policy = DefaultBranchingPolicy()
        self.policy_type = policy_type
        self.node_select_policy = None

        self.filepath = train_filepath

        if not os.path.isdir(oracle_solution_filepath):
            Path(oracle_solution_filepath).mkdir(exist_ok=True)
        self.oracle_problem_index = 0
        self.oracle_dataset = OracleDataset(root=oracle_solution_filepath)

        self.num_instances = num_instances
        self.file_count_offset = 0


    def collect_data(self, num_instances=10, policy='oracle', train=True):

        N, M, max_ant = self.parameters.train_size
        
        # fetch the following data from saved files or create new if all are used up
        instances = None
        optimal_solution_list = []
        optimal_objective_list = []
        avg_oracle_num_problems = []
        avg_oracle_time = []

        # For training new data is needed in each iteration
        if train:
            samples = self.oracle_dataset.get_batch(self.num_instances)
        else:
            self.oracle_dataset.re_init()
            if self.num_instances > len(self.oracle_dataset):
                samples = None
            else:
                samples = self.oracle_dataset.get_batch_from_indices(list(range(self.num_instances)))
            
        if samples is not None:
            instances, optimal_solution_list, optimal_objective_list, oracle_steps, oracle_time = samples 
            instances = np.stack(instances, axis=0)
            avg_oracle_num_problems = np.mean(oracle_steps)
            avg_oracle_time = np.mean(oracle_time)

        else:
            instances = (np.random.randn(num_instances, N, M) + 1j*np.random.randn(num_instances, N,M))/np.sqrt(2)
            # instances = np.stack((np.real(H), np.imag(H)), axis=1)
            
            arguments_oracle = [OracleArg(instance=instances[i], 
                                max_ant=self.parameters.train_size[2],
                                robust_beamforming=self.parameters.robust_beamforming,
                                sigma_sq=self.parameters.sigma_sq,
                                min_sinr=self.parameters.min_sinr,
                                robust_margin=self.parameters.robust_margin) for i in range(len(instances))]
                                
            # arguments_oracle = list(zip(list(instances), [max_ant]*num_instances))
            print('starting first pool')
            with Pool(min(num_instances, 15)) as p:
                out_oracle = p.map(self.solve_bb_process, arguments_oracle)
                print('first pool ended')

            # Prune away the problem instances that were not feasible (could not be solved)
            for i in range(len(out_oracle)-1, -1, -1):
                if out_oracle[i]['objective'] == np.inf:
                    del out_oracle[i]
                    instances = np.concatenate((instances[:i,::], instances[i+1:,::]), axis=0)

            # the returned order is x_opt:[can be a tuple], global_U, timsteps, time
            optimal_solution_list = [out_oracle[i]['solution'] for i in range(len(out_oracle))]
            optimal_objective_list = [out_oracle[i]['objective'] for i in range(len(out_oracle))]
            avg_oracle_num_problems = np.mean(np.array([out_oracle[i]['num_problems'] for i in range(len(out_oracle))]))
            avg_oracle_time = np.mean(np.array([out_oracle[i]['time'] for i in range(len(out_oracle))]))

            self.oracle_dataset.save_batch(list(instances),
                                            [out_oracle[i]['solution'] for i in range(len(out_oracle))],
                                            [out_oracle[i]['objective'] for i in range(len(out_oracle))],
                                            [out_oracle[i]['num_problems'] for i in range(len(out_oracle))],
                                            [out_oracle[i]['time'] for i in range(len(out_oracle))])

        
        arguments_ml = list(zip(list(instances), optimal_solution_list, optimal_objective_list, range(len(instances)), [policy]*len(instances)))

        arguments_ml = [MLArgTrain(instance = instances[i],
                            max_ant = self.parameters.train_size[2],
                            optimal_objective = optimal_objective_list[i],
                            w_optimal = optimal_solution_list[i],
                            file_count = i,
                            robust_beamforming = self.parameters.robust_beamforming,
                            sigma_sq = self.parameters.sigma_sq,
                            min_sinr = self.parameters.min_sinr,
                            robust_margin = self.parameters.robust_margin,
                            policy_filepath = policy,
                            mask_heuristics = None) for i in range(len(instances))
                            ]

        print('starting second pool')
        with Pool(min(len(instances),15)) as p:
            out_ml = p.map(self.collect_data_instance, arguments_ml)
            print('second pool ended')
        
        # the returned order for collect_data_instance is timesteps, ogap, time (seconds), ratio of optimal nodes to total nodes 
        avg_ml_num_problems = np.mean(np.array([out_ml[i]['num_problems'] for i in range(len(out_ml))]))

        avg_ml_ogap = 0
        num_solved = 0
        for i in range(len(out_ml)):
            if out_ml[i]['ogap'] >-1:
                avg_ml_ogap += out_ml[i]['ogap'] 
                num_solved += 1
        avg_ml_ogap /= num_solved
        # avg_ml_ogap = np.mean(np.array([out_ml[i][1] for i in range(len(out_ml))]))
        avg_ml_time = np.mean(np.array([out_ml[i]['time'] for i in range(len(out_ml))]))

        if train:
            self.file_count_offset += len(instances)

        # return order is time speedup, ogap, steps_speedup
        return {'time_speedup': avg_oracle_time/avg_ml_time, 'ogap': avg_ml_ogap, 'problems_speedup': avg_oracle_num_problems/avg_ml_num_problems}

    def collect_data_instance(self, arguments:MLArgTrain):

        print('function {} started'.format(arguments.file_count))
        #TODO: do the following with parameters not filename
        # print('optimal ', w_optimal)
        env = Environment(observation_function=self.observation_function, epsilon=0.005)

        env.set_node_select_policy(node_select_policy_path=arguments.policy_filepath)
        
        env.reset(arguments.instance, 
                    max_ant = arguments.max_ant,  
                    oracle_opt = arguments.w_optimal, 
                    sigma_sq = arguments.sigma_sq,
                    min_sinr = arguments.min_sinr,
                    robust_margin = arguments.robust_margin,
                    robust_beamforming= arguments.robust_beamforming)

        branching_policy = DefaultBranchingPolicy()
        t1 = time.time()
        timestep = 0
        done = False
        time_taken = 0
        sum_label = 0
        node_counter = 0
        while timestep < MAX_STEPS and len(env.nodes)>0 and not done:
            print('timestep {}'.format(timestep))
            env.fathom_nodes()
            if len(env.nodes) == 0:
                break
            node_id, node_feats, label = env.select_node()
            if len(env.nodes) == 0:
                break
            time_taken += time.time()-t1
            sum_label += label
            self.save_file((node_feats, label), arguments.file_count, node_counter)
            node_counter += 1
            t1 = time.time()
            prune_node = env.prune(node_feats)

            if prune_node:
                env.delete_node(node_id)
                continue
            else:
                last_id = len(env.nodes)

                branching_var = branching_policy.select_variable(node_feats, env.action_set_indices)
                try:
                    done = env.push_children(branching_var, node_id)
                except:
                    break

            timestep = timestep+1
            if env.is_terminal():
                break

        if node_counter < 1:
            print('node counter null H {}, w_opt {}'.format(env.H_complex, arguments.w_optimal))

        ml = env.global_U
        ogap = ((ml - arguments.optimal_objective)/arguments.optimal_objective)*100

        print('instance result: timestep {}, ogap {}, time {}, sum_label {}, optimal objective {}, ml {}'.format(timestep, ogap, time_taken, sum_label, arguments.optimal_objective, ml))
        
        # return order is timesteps, ogap, time (seconds), ratio of optimal nodes to total nodes 
        return {'timestep': timestep, 'ogap': ogap, 'time': time_taken, 'optimal_node_ratio': sum_label/node_counter, 'num_problems':env.bm_solver.num_problems}
    
    def solve_bb_process(self, arguments: OracleArg):
        try:
            output = solve_bb(instance=arguments.instance, 
                            max_ant=arguments.max_ant, 
                            robust_beamforming=arguments.robust_beamforming,
                            min_sinr=arguments.min_sinr,
                            sigma_sq=arguments.sigma_sq,
                            robust_margin=arguments.robust_margin)
            return output
        except SolverException as e:
            print('Solver Exception: ', e)
            return {'solution': None, 'objective': np.inf, 'time': 0, 'num_problems': 0}


    def save_file(self, sample, file_count, node_counter):
        if self.filepath is not None:
            filename = os.path.join(self.filepath,'sample_{}_{}.pkl'.format(self.file_count_offset + file_count, node_counter))
            with gzip.open(filename, 'wb') as f:
                pickle.dump(sample, f)

    def dummy_collect_instance(self, arguments):
        instance, w_optimal, optimal_objective, file_count = arguments
        print('started collect instance {}'.format(file_count))
        import time
        time.sleep(1)
        print('ended collect instance {}'.format(file_count))