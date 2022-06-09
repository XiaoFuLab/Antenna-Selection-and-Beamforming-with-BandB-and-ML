import numpy as np


class SolverException(Exception):
    pass

def get_rank1(W):
    """
    W: PSD matrix
    """
    assert W.shape[0] == W.shape[1], 'matrix {} not square'.format(W)
    eval, evec = np.linalg.eig(W)
    rank1 =  np.real(max(eval)/sum(eval)) > 0.99
    principal_id = np.argmax(eval)
    return evec[:,principal_id]*np.sqrt(np.real(eval[principal_id])), rank1
