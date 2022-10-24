import os

ROOT = '/scratch/sagar/Projects/Antenna-Selection-and-Beamforming-with-BandB-and-ML'

# TASK =  # one of 'beamforming', 'robust_beamforming'
TASK = 'beamforming'
# TASK = 'robust_beamforming'

DEBUG = False

DEVICE = 'cuda'

DAGGER_NUM_TRAIN_EXAMPLES_PER_ITER = 30
DAGGER_NUM_VALID_EXAMPLES_PER_ITER = 30
DAGGER_NUM_ITER = 20
BB_MAX_STEPS = 10000

REUSE_DATASET = True

if TASK =='beamforming':
    ANTENNA_NFEATS = 4
    EDGE_NFEATS = 9
    VAR_NFEATS = 8
    NODE_DEPTH_INDEX = 5

    IN_FEATURES = 219

    DATA_PATH = os.path.join(ROOT, 'antenna_selection/data_bf/data/data_multiprocess')
    MODEL_PATH = os.path.join(ROOT, 'antenna_selection/data_bf/trained_models')
    RESULT_PATH = os.path.join(ROOT, 'antenna_selection/data_bf/data')

    LOAD_MODEL = False
    LOAD_MODEL_PATH = os.path.join(ROOT, 'antenna_selection/data_bf/trained_models/gnn2_iter_3')

    CLASS_IMBALANCE_WT = 11 # for 8,3,5 (N,M,L) use 11, for 12,6,5 max_ant use 11

    ETA_EXP = 100000000.0

    # This is the weight given to the regularization term. In practice, setting this to 0 also seems to work just as well. 
    LAMBDA_ETA = 1

elif TASK =='robust_beamforming':
    ANTENNA_NFEATS = 4
    EDGE_NFEATS = 9
    VAR_NFEATS = 8
    NODE_DEPTH_INDEX = 5

    IN_FEATURES = 219

    DATA_PATH = os.path.join(ROOT, 'robust_beamforming/data_rbf/data/dagger')
    MODEL_PATH = os.path.join(ROOT, 'robust_beamforming/data_rbf/trained_models')
    RESULT_PATH = os.path.join(ROOT, 'robust_beamforming/data_rbf/data/')
    VALIDATION_PATH = os.path.join(ROOT, 'robust_beamforming/data_rbf/validation_set')
    
    LOAD_MODEL = False
    LOAD_MODEL_PATH = os.path.join(ROOT, 'robust_beamforming/data_rbf/trained_models/gnn2_iter_3')

    CLASS_IMBALANCE_WT = 11
    ETA_EXP = 100000000.0
    LAMBDA_ETA = 1 # for 8,3,5 use 1e-4, for 12, 6, 8 use 1e-5
 