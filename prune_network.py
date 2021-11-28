"""
    Update the weights for the network after pruning
    A. For stably inactive neurons, remove the corresponding weights before and after them.
    B. For stably active neurons, remove the weights before and update the weights and bias:
    We have the following notations:
        0. h1_a: [c1_a]
                 the stably active neurons
        1. w1_a: [c1_a, c0] 
                 the weights before the active neurons. 
           w1_a = {w11_a, w12_a}, whose rank is r1 and r1 < c1_a 
           w11_a: [r1, c0] 
                  the weights before h11_a;
                  the r1 row vectors which can represent the base in R^r1
           w12_a: [c1_a - r1, c0]
                  the weights before h12_a;
                  the rest row vectors can be presented as a linear combination of the w11_a
        2. w12_a = K1*w11_a, 
           K1: [c1_a - r1, r1]
        3. w2_a:  [c2, c1_a] 
                  the weights after the active neurons.
           w2_a = {w21_a, w22_a}
           w21_a: [c2, r1]
                  the weights after h11_a;
           w22_a: [c2, c1_a - r1]
                  the weights after h12_a;
        4. b1_a = {b11_a, b12_a}
           b11_a: [r1, ]
           b12_a: [c1_a - r1, ]
    We update the weights before and after h1_a as below:
        1. remove w12_a from w1_a
        2. merge w22_a into w21_a:
           w21_a' = w21_a + w22_a * K1
        3. update b2:
           b2'    = b2 + w22_a * (b12_a - K1 * b11_a)
        4. remove w22_a
        
"""
import os
import sys

import numpy as np
from numpy.linalg import matrix_rank, norm
import torch
import sympy

from common.timer import Timer

DEBUG = False
timer = Timer()
#######################################################################################
# remove the corresponding weights before and after the stably inactive neurons
#######################################################################################
def prune_inactive_per_layer(w1, w2, ind_inact):
    """
    w1: [c1, c0] the weights before the current layer
    w2: [c2, c1] the weithgs after the current layer
    ind_inact: a list of the index of the stably inactive neurons
    """
    w1 = np.delete(w1, ind_inact, axis=0)
    w2 = np.delete(w2, ind_inact, axis=1)
    return w1, w2


#######################################################################################
# 1. remove the corresponding weights before the stably active neurons
# 2. update the weights after them
#######################################################################################
def prune_active_per_layer(w1, w2, b1, b2, ind_act):
    """
    w1: [c1, c0] the weights before the current layer
    w2: [c2, c1] the weithgs after the current layer
    ind_inact: a list of the index of the stably inactive neurons
    b1: [c1] 
    b2: [c2]
    """
    ind_act = np.array(ind_act)
    w1_a = w1[ind_act, :]
    b1_a = b1[ind_act]
    r1  = np.linalg.matrix_rank(w1_a.astype(np.float64))
    import pdb;pdb.set_trace()
    if r1 == w1_a.shape[0]:
        return w2,b2,[]
    
    print('Start to solve linearly dependent row vectors')
    K1, is_indep = solve_linearly_depend(w1_a)
    #timer.stop('solve_linearly_depend is done')
    w2_a = w2[:, ind_act]
    b2_old = b2[:]
    b11 = b1_a[is_indep]
    b12 = b1_a[is_indep == 0]
    w21 = w2_a[:, is_indep]
    w22 = w2_a[:, is_indep == 0]
    #   1. remove w12_a from w1_a
    #w1 = np.delete(w1, ind_act[is_indep==0], axis = 0)
    #    2. merge w22_a into w21_a:
    #       w21_a' = w21_a + w22_a * K1
    #    3. update b2:
    #       b2'    = b2 + w22_a * (b12_a - K1 * b11_a)
    w2[:, ind_act[is_indep]] = w21 + w22 @ K1 
    b2                       = b2  + w22 @ (b12 - K1 @ b11)
    #    4. remove w22
    #w2 = np.delete(w2, ind_act[is_indep==0])
    if DEBUG:
        for run in range(10):
            h = np.random.randn(w1.shape[1])
            x = w1 @ h + b1
            x = x[ind_act]
            y_old = w2_a @ x + b2_old
            y_new = w2[:, ind_act[is_indep]] @ x[is_indep] + b2
            if not np.allclose(y_old, y_new):
                import pdb;pdb.set_trace()
    #import pdb;pdb.set_trace()
    return w2, b2, ind_act[is_indep==0]


#######################################################################################
#ref: https://stackoverflow.com/a/44556151/4874916
#######################################################################################
def find_independ_rows(A):
    # find the independent row vectors
    # only work for square matrix
    #lambdas, V = np.linalg.eig(A.T)
    #return lambdas!=0
    
    #import pdb;pdb.set_trace()
    inds = find_independ_rows2(A)
    
    # too slow
    #_, inds = sympy.Matrix(A).T.rref()
    #import pdb;pdb.set_trace()
    return inds

def find_independ_rows2(A):
    r = np.linalg.matrix_rank(A.astype(np.float64))
    base = [A[0,:]]
    base_ind = [0]
    row = 1
    cur_r = 1 
    while cur_r < r:
        tmp = base + [A[row,:]]
        #import pdb;pdb.set_trace()
        if np.linalg.matrix_rank(np.stack(tmp, axis=0).astype(np.float64)) > cur_r:
            cur_r += 1
            base.append(A[row,:])
            base_ind.append(row)
        row += 1
    return base_ind


#######################################################################################
# 1. find the base row vectors for w11
# 2. represent the non-base row vectors as a linear combination of the base
#######################################################################################
def solve_linearly_depend(w11):
    """
        Args:
            w11: [c1_a, c0]
        Returns:
            K  : [c1_a -rank(w11), rank(w11)]
            is_indep: [rank(w11),]
    """
    ##timer.stop('start find_independ_rows')
    # find the independent row vectors
    is_indep = find_independ_rows(w11)
    ##timer.stop('finsh find_independ_rows')
    is_indep = np.array([i in is_indep for i in range(w11.shape[0])])
    #ind_indep    = find_li_vectors(w11)
    dep = w11[is_indep == 0, :] # the linearly dependent row vectors
    base = w11[is_indep, :] # the independent row vectors
    # solve a linear equation: dep = K * base
    K = []
    for i in range(dep.shape[0]):
        y  = dep[i, :]
        A  = np.concatenate([base, y[None,:]], axis=0).T
        is_indep_A = np.array(find_independ_rows(A))
        #print('base[:,is_indep_A]: ', base[:,is_indep_A].shape)
        ##timer.stop(f'start np.linalg.solve: {i} ')
        #import pdb;pdb.set_trace()
        k  = np.linalg.solve(base[:,is_indep_A].T, y[is_indep_A])
        ##timer.stop(f'start np.linalg.solve: {i}')
        assert(np.allclose(np.dot(base.T, k), y))
        K.append(k)
    K = np.stack(K, axis=0)
    return K, is_indep 
 

def sanity_ckp():
    w1 = np.array(
            [[1,0,2,1],
             [1,1,0,0],
             [2,1,2,1],
             [1,2,3,0],
             [11,12,13,0]])
    b1 = np.arange(5)
    w2 = np.arange(10).reshape(2,5)
    b2 = np.array([11,12])
    act_neurons = np.array([[1, 0], [1,1], [1,2]]).T
    inact_neurons = np.array([[1,4]]).T
    w_names = ['w1', 'w2']
    b_names = ['b1', 'b2']
    return [w1, w2], [b1, b2], act_neurons, inact_neurons, w_names, b_names
    
#######################################################################################
# update the weights and bias in the checkpoints
#######################################################################################
def prune_ckp(model_path):
    
    if DEBUG:
        weights, bias, act_neurons, inact_neurons, w_names, b_names = sanity_ckp()
    else:
        timer.start()
        ckp_path = os.path.join(model_path, 'checkpoint_120.tar')
        pruned_ckp_path = os.path.join(model_path, 'pruned_checkpoint_120.tar')
        stb_path = os.path.join(model_path, 'stable_neurons.npy')
        ckp = torch.load(ckp_path)
        weights = []
        bias    = []
        
        w_names = sorted([name for name in ckp['state_dict'].keys() 
                                if 'weight' in name and 'features' in name])
        b_names = sorted([name for name in ckp['state_dict'].keys() 
                                if 'bias' in name and 'features' in name])
        w_names.append('classifier.0.weight')
        b_names.append('classifier.0.bias')

        device = ckp['state_dict'][w_names[0]].device

        for name in w_names:
            weights.append(ckp['state_dict'][name].cpu().numpy())
        for name in b_names:
            bias.append(ckp['state_dict'][name].cpu().numpy())

        stb_neurons = np.load(stb_path, allow_pickle=True).item()
        act_neurons = stb_neurons['stably_active'].squeeze(axis=2).T
        inact_neurons = stb_neurons['stably_inactive'].squeeze(axis=2).T
        ##timer.stop('loaded the checkpoint')
    #import pdb;pdb.set_trace()
    for l in range(1, len(weights)):
        ind_act   = act_neurons[1,   act_neurons[0,:] == l]
        ind_inact = inact_neurons[1, inact_neurons[0,:] == l]
        w1 = weights[l-1]
        w2 = weights[l]
        b1 = bias[l-1]
        b2 = bias[l]
        prune_ind = []
        if len(ind_act) > 0:
            #import pdb;pdb.set_trace()
            w2, b2, prune_ind_act = prune_active_per_layer(w1, w2, b1, b2, ind_act)
            prune_ind.extend(prune_ind_act)
        else:
            prune_ind_act = []
        if len(ind_inact) > 0:
            prune_ind.extend(ind_inact)
        if len(prune_ind) > 0:
            w1 = np.delete(w1, prune_ind, axis=0)
            b1 = np.delete(b1, prune_ind, axis=0)
            w2 = np.delete(w2, prune_ind, axis=1)
        print(f'Layer-{l}: prune {len(prune_ind_act)} stably active neurons')
        print(f'layer-{l}: prune {len(ind_inact)} stably inactive neurons')
        # update the weights and bias
        weights[l-1] = w1
        bias[l-1]    = b1
        weights[l]   = w2
        bias[l]      = b2
        ##timer.stop(f'{l} layer is pruned')
        #import pdb;pdb.set_trace()
    # update the ckeckpoints
    for i, name in enumerate(w_names):
        ckp['state_dict'][name] = torch.from_numpy(weights[i]).cuda(device=device)
    for i, name in enumerate(b_names):
        ckp['state_dict'][name] = torch.from_numpy(bias[i]).cuda(device=device)
    # save the checkpoint 
    torch.save(ckp, pruned_ckp_path)


if __name__ == '__main__':
    #model_path = 'model_dir/CIFAR100-rgb/dnn_CIFAR100-rgb_400-400_7.500000000000001e-05_0001'
    #model_path = 'model_dir/CIFAR10-rgb.0526/dnn_CIFAR10-rgb_400-400_0.000175_0003'
    model_path = sys.argv[1]
    prune_ckp(model_path)
