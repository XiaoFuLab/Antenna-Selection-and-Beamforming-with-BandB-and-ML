

from antenna_selection.observation import Observation
from antenna_selection.bb_unified import BBenv as Environment, DefaultBranchingPolicy, solve_bb

import numpy as np

import gzip
import pickle
from pathlib import Path
import time
import os
from torch.multiprocessing import Pool

from models.helper import SolverException
from torch.utils.data import Dataset

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
                max_ant=None, 
                policy='oracle',
                train_filepath=None, 
                policy_type='gnn', 
                oracle_solution_filepath=None,
                num_instances=10):
        # env = Environment(observation_function=observation_function, epsilon=0.002)
        self.observation_function = observation_function
        self.max_ant = max_ant
        print('********in Datacollect', self.max_ant)

        self.branching_policy = DefaultBranchingPolicy()
        self.policy_type = policy_type
        self.node_select_policy = None

        self.filepath = train_filepath

        if not os.path.isdir(oracle_solution_filepath):
            print('the folder does not exist')
            Path(oracle_solution_filepath).mkdir(exist_ok=True)
        self.oracle_problem_index = 0
        self.oracle_dataset = OracleDataset(root=oracle_solution_filepath)

        self.num_instances = num_instances
        self.file_count_offset = 0


    def collect_data(self, instance_gen, num_instances=10, policy='oracle', train=True):

        N, M = next(instance_gen).shape[1], next(instance_gen).shape[2]
        
        # fetch the following data from saved files or create new if all are used up
        instances = None
        optimal_solution_list = []
        optimal_objective_list = []
        avg_oracle_steps = []
        avg_oracle_time = []

        oracle_samples_available = False

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
            avg_oracle_steps = np.mean(oracle_steps)
            avg_oracle_time = np.mean(oracle_time)
            oracle_sample_available = True
        else:
            H = (np.random.randn(num_instances, N, M) + 1j*np.random.randn(num_instances, N,M))/np.sqrt(2)
            # H = Channels.copy() 
            instances = np.stack((np.real(H), np.imag(H)), axis=1)
            
            arguments_oracle = list(zip(list(instances), [self.max_ant]*num_instances))
            print('starting first pool')
            with Pool(min(num_instances, 20)) as p:
                out_oracle = p.map(self.solve_bb_process, arguments_oracle)
                print('first pool ended')

            # Prune away the problem instances that were not feasible (could not be solved)
            for i in range(len(out_oracle)-1, -1, -1):
                if out_oracle[i][1] == np.inf:
                    del out_oracle[i]
                    instances = np.concatenate((instances[:i,::], instances[i+1:,::]), axis=0)

            # the returned order is x_opt:[can be a tuple], global_U, timsteps, time
            optimal_solution_list = [out_oracle[i][0] for i in range(len(out_oracle))]
            optimal_objective_list = [out_oracle[i][1] for i in range(len(out_oracle))]
            avg_oracle_steps = np.mean(np.array([out_oracle[i][2] for i in range(len(out_oracle))]))
            avg_oracle_time = np.mean(np.array([out_oracle[i][3] for i in range(len(out_oracle))]))

            self.oracle_dataset.save_batch(list(instances),
                                            [out_oracle[i][0] for i in range(len(out_oracle))],
                                            [out_oracle[i][1] for i in range(len(out_oracle))],
                                            [out_oracle[i][2] for i in range(len(out_oracle))],
                                            [out_oracle[i][3] for i in range(len(out_oracle))])

        arguments_ml = list(zip(list(instances), optimal_solution_list, optimal_objective_list, range(len(instances)), [policy]*len(instances)))
        # arguments_ml = list(zip(list(instances), optimal_solution_list, optimal_objective_list, range(len(instances))))
       
        
        print('starting second pool')
        with Pool(min(len(instances),30)) as p:
            out_ml = p.map(self.collect_data_instance, arguments_ml)
            # out_ml = p.map(self.dummy_collect_instance, arguments_ml)

            print('second pool ended')
        
        # the returned order for collect_data_instance is timesteps, ogap, time (seconds), ratio of optimal nodes to total nodes 

        avg_ml_steps = np.mean(np.array([out_ml[i][0] for i in range(len(out_ml))]))

        avg_ml_ogap = 0
        num_solved = 0
        for i in range(len(out_ml)):
            if out_ml[i][1] > -1:
                avg_ml_ogap += out_ml[i][1] 
                num_solved += 1
        avg_ml_ogap /= num_solved
        # avg_ml_ogap = np.mean(np.array([out_ml[i][1] for i in range(len(out_ml))]))
        avg_ml_time = np.mean(np.array([out_ml[i][2] for i in range(len(out_ml))]))

        if train:
            self.file_count_offset += len(instances)


        # return order is time speedup, ogap, steps_speedup
        return avg_oracle_time/avg_ml_time, avg_ml_ogap, avg_oracle_steps/avg_ml_steps


    def collect_data_instance(self, arguments):
        instance, w_optimal, optimal_objective, file_count, policy_filepath = arguments
        print('function {} started'.format(file_count))
        #TODO: do the following with parameters not filename
        # print('optimal ', w_optimal)
        env = Environment(observation_function=self.observation_function, epsilon=0.005)

        env.set_node_select_policy(node_select_policy_path=policy_filepath, policy_type=self.policy_type)
        
        if TASK == 'robust_beamforming':
            env.reset(instance, max_ant=self.max_ant,  oracle_opt=w_optimal, robust_beamforming=True)
        elif TASK == 'antenna_selection':
            env.reset(instance, max_ant=self.max_ant,  oracle_opt=w_optimal, robust_beamforming=False)
        else:
            env.reset(instance, max_ant=self.max_ant,  oracle_opt=w_optimal)



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
            self.save_file((node_feats, label), file_count, node_counter)
            node_counter += 1
            t1 = time.time()

            # print('Node id for pruning decision {}'.format(env.nodes[node_id].node_index))
            prune_node = env.prune(node_feats)
            # prune_node = False
            if prune_node:
                env.delete_node(node_id)
                continue
            else:
                # print('Node id {}'.format(env.nodes[node_id].node_index))
                last_id = len(env.nodes)

                branching_var = branching_policy.select_variable(node_feats, env.action_set_indices)
                try:
                    done = env.push_children(branching_var, node_id)
                except:
                    break
                
                # for i in range(last_id, len(env.nodes)):
                #     print('Children {} z_mask {}, z_sol {}, l_angle {}, u angle {}'.format(env.nodes[i].node_index, env.nodes[i].z_mask, env.nodes[i].z_sol, env.nodes[i].l_angle, env.nodes[i].u_angle) )

                # print('*********')
                # print()
            timestep = timestep+1
            if env.is_terminal():
                break

        if node_counter < 1:
            print('node counter null H {}, w_opt {}'.format(env.H_complex, w_optimal))
        # ml = np.linalg.norm(env.W_incumbent, 'fro')**2
        ml = env.global_U
        ogap = ((ml - optimal_objective)/optimal_objective)*100
        # if ogap>1:
        #     print('H: {}, w_opt: {}, obj: {}, ml: {}'.format(env.H_complex, w_optimal, optimal_objective, ml))
        #     debug_dict = {'H': env.H_complex,
        #                   'w_opt': w_optimal,
        #                   'obj': optimal_objective,
        #                   'ml': ml}
        #     with open('debug.pkl', 'wb') as f:
        #         pickle.dump(debug_dict, f)

        # time_taken += time.time() - t1
        print('instance result', timestep, ogap, time_taken, sum_label, optimal_objective, ml)
        # if ogap < -0.1:
        #     print('obj: {}, ml: {}'.format(env.H_complex, w_optimal, optimal_objective, ml))
        #     print('w_oracle: {}, w_ml: {}, z_oracl: {}, z_ml {}'.format(w_optimal[1], env.w_incumbent,  w_optimal[0], env.z_incumbent))
        #     debug_dict = {'H': env.H_complex,
        #                   'w_opt': w_optimal,
        #                   'obj': optimal_objective,
        #                   'ml': ml}
        #     with open('debug.pkl', 'wb') as f:
        #         pickle.dump(debug_dict, f)
        # return order is timesteps, ogap, time (seconds), ratio of optimal nodes to total nodes 
        return timestep, ogap, time_taken, sum_label/node_counter
    
    def solve_bb_process(self, tup):
        try:
            instance, max_ant = tup
            if TASK == 'robust_beamforming':
                output = solve_bb(instance, max_ant, robust_beamforming=True)
            else:
                output = solve_bb(instance, max_ant, robust_beamforming=False)

        except SolverException as e:
            print('Solver Exception: ', e)
            return None, np.inf, 0, 0

        return output

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