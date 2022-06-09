import torch
import torch_geometric

LEARNING_RATE = 0.001
NB_EPOCHS = 50
PATIENCE = 10
EARLY_STOPPING = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMB_SIZE = 64

from models.setting import *

class GNNNodeSelectionPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = GNNPolicy()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, obs, num_graphs):
        logits = self.main(obs.antenna_features, obs.edge_index, obs.edge_attr, obs.variable_features)
        logits = logits.reshape([num_graphs, -1]).mean(dim=1)
        return self.sigmoid(logits)

class GNNPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # CONSTRAINT EMBEDDING
        self.antenna_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(ANTENNA_NFEATS),
            torch.nn.Linear(ANTENNA_NFEATS, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(EDGE_NFEATS),
            torch.nn.Linear(EDGE_NFEATS, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(VAR_NFEATS),
            torch.nn.Linear(VAR_NFEATS, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.antenna_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        
        # Two half convolutions
        # print('var', variable_features.shape, 'cons', constraint_features.shape, 'edge', reversed_edge_indices.shape, 'edge_f', edge_features.shape)
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        return output


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need 
    to provide the exact form of the messages being passed.
    """
    def __init__(self):
        super().__init__('add')
        
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(EMB_SIZE, EMB_SIZE)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(EMB_SIZE, EMB_SIZE, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(EMB_SIZE, EMB_SIZE, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, EMB_SIZE)
        )
        
        self.post_conv_module = torch.nn.Sequential(
            torch.nn.LayerNorm(EMB_SIZE)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]), 
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_i) 
                                           + self.feature_module_edge(edge_features) 
                                           + self.feature_module_right(node_features_j))
        return output
    