import numpy as np
import pickle
import torch
import gzip

class Observation(object):
    def __init__(self):
        self.antenna_features  = None # np.zeros(N, 3) # three features for each antenna
        self.variable_features = None # np.zeros(M, 15)
        self.edge_index = None
        self.edge_features     = None # np.zeros(N*M, 3)
        self.candidates        = None # np.arange(M)
        pass

    def extract(self, model):
        # TODO: make the observation out of the model 
        self.candidates = model.action_set_indices
        self.antenna_features = np.zeros((model.N, 4))
        self.antenna_features[:,0] = model.active_node.z_sol
        self.antenna_features[:,1] = model.active_node.z_feas
        self.antenna_features[:,2] = model.active_node.z_mask
        self.antenna_features[:,3] = np.linalg.norm(model.active_node.W_sol,  axis=1)


        # edge features
        self.edge_index = np.stack((np.repeat(np.arange(model.N), model.M), np.tile(np.arange(model.M), model.N)))
        self.edge_features = np.zeros((model.M*model.N, 9))
        self.edge_features[:,0] = np.real(model.H_complex.reshape(-1))
        self.edge_features[:,1] = np.imag(model.H_complex.reshape(-1))
        self.edge_features[:,2] = np.abs(model.H_complex.reshape(-1))

        self.edge_features[:,3] = np.real(model.W_incumbent.reshape(-1))
        self.edge_features[:,4] = np.imag(model.W_incumbent.reshape(-1))
        self.edge_features[:,5] = np.abs(model.W_incumbent.reshape(-1))

        self.edge_features[:,6] = np.real(model.active_node.W_sol.reshape(-1))
        self.edge_features[:,7] = np.imag(model.active_node.W_sol.reshape(-1))
        self.edge_features[:,8] = np.abs(model.active_node.W_sol.reshape(-1))


        # construct variable features
        # global features
        global_upper_bound = 1000 if model.global_U == np.inf else model.global_U
        local_upper_bound =  2000 if model.active_node.U == np.inf else model.active_node.U

        self.variable_features = np.zeros((model.M, 8))
        self.variable_features[:,0] = model.global_L # global lower bound
        self.variable_features[:,1] = global_upper_bound # global upper bound
        self.variable_features[:,2] = (local_upper_bound - global_upper_bound) < model.epsilon 

        # local features
        W_H = np.matmul(model.active_node.W_sol.conj().T, model.H_complex)
        W_H = np.abs(W_H)
        mask = np.eye(*W_H.shape)
        mask_comp = 1-mask
        direct = np.sum(W_H*mask, axis=1)
        interference = W_H*mask_comp
        # aggregate_interference = np.sum(interference, axis=1)
        aggregate_interference = np.sum(interference, axis=0)

        H_w = np.matmul(model.H_complex.conj().T, model.active_node.W_sol)
        self.variable_features[:,3] = np.squeeze(direct)
        self.variable_features[:,4] = np.squeeze(aggregate_interference)
        self.variable_features[:,5] = model.active_node.depth

        self.variable_features[:, 6] = 0 if model.active_node.L == np.inf else model.active_node.L
        self.variable_features[:, 7] = local_upper_bound
        

        #TODO: include the normalized number of times a variable has been selected by the current branching policy    
        return self


class LinearObservation(object):
    """
    Constructs a long obervation vector for linear neural network mapping
    """

    def __init__(self):
        self.observation = None 
        self.candidates  = None # np.arange(M)
        self.variable_features = None
        pass

    def extract(self, model):
        return self




def prob_dep_features_from_obs(observation):
    """
    Arguments: 
        observation: Observation instance (for graph)
        output: Vector of observation (with all the information from the input observation)
    """
    # use the indices of observation to extract the features
    features = np.concatenate((observation.antenna_features.reshape(-1), 
                                    observation.variable_features.reshape(-1), 
                                    observation.edge_features.reshape(-1)))
    return features 

def prob_indep_features_from_obs(observation):
    """
    Arguments: 
        observation: Observation instance (for graph)
        output: Vector of observation (with only those features from the input observation object that is problem size independent)
    List of all problem size independent features in observation object in antenna selection:
        1. [variable features 0] global lower bound
        2. [variable features 1] global upper bound
        3. [variable features 2] local_upper_bound - global_upper_bound < model.epsilon
        4. [variable features 5] active node depth
        5. [variable features 6] local lower bound
        6. [variable features 7] local upper bound
    """
    features = np.zeros(6)
    features[0] = observation.variable_features[0,0]      
    features[1] = observation.variable_features[0,1]
    features[2] = observation.variable_features[0,2]
    features[3] = observation.variable_features[0,5]
    features[4] = observation.variable_features[0,6]
    features[5] = observation.variable_features[0,7]
    
    return features


def get_dataset_svm(sample_files, prob_size_dependent=True):
    assert len(sample_files)>0, "list cannot be of size 0"

    features = []
    labels = []
    # features = torch.zeros(len(sample_files)) 
    # labels = torch.zeros(len(sample_files))

    for i in range(len(sample_files)):
        with gzip.open(sample_files[i], 'rb') as f:
            sample = pickle.load(f)
        sample_observation, target = sample[0], sample[1]
        labels.append(target)
        if prob_size_dependent:
            features.append(torch.tensor(prob_dep_features_from_obs(sample_observation), dtype=torch.float32))
        else:
            features.append(torch.tensor(prob_indep_features_from_obs(sample_observation), dtype=torch.float32))


    return torch.stack(features, axis=0), torch.tensor(labels)


class LinearDataset(torch.utils.data.Dataset):
    def __init__(self, sample_files, prob_size_dependent=True):
        super().__init__()
        self.sample_files = sample_files
        self.prob_size_dependent = prob_size_dependent

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        
        with gzip.open(self.sample_files[idx], 'rb') as f:
            sample = pickle.load(f)
        sample_observation, target = sample[0], sample[1]

        if self.prob_size_dependent:
            features = prob_dep_features_from_obs(sample_observation)
        else:
            features = prob_indep_features_from_obs(sample_observation)

        return torch.tensor(features, dtype=torch.float32),  target



