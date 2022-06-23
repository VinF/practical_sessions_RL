import torch

def dist_dqn(x,theta,aspace=3):
    """
    3 layer neural network
    `x` is input vector of dim 128
    `theta` is a parameter vector that will be unpacked into 3 separate layer matrices
        layer1: L1 x X -> 100x128 x 128xB -> 100xB
        layer2: L2 x L1 -> 25x100 x 100xB -> 25xB
        layer3: L3 x L2 -> 3x25x51 x 25xB -> 3x51xB
        where `B` is the batch size dimension
        
    Returns a Batch Sizex A x 51 tensor where A is the action-space size
    """
    dim0 = 128
    dim1 = 100
    dim2 = 25
    dim3 = 51
    t1 = dim0*dim1
    t2 = dim2*dim1
    theta1 = theta[0:t1].reshape(dim0,dim1)
    theta2 = theta[t1:t1 + t2].reshape(dim1,dim2)
    
    l1 = x @ theta1 #B x 128 x 128 x 100 = B x 100
    l1 = torch.selu(l1)
    l2 = l1 @ theta2 # B x 100 x 100 x 25 = B x 25
    l2 = torch.selu(l2)
    l3 = []
    for i in range(aspace):
        step = dim2*dim3
        theta5_dim = t1 + t2 + i * step
        theta5 = theta[theta5_dim:theta5_dim+step].reshape(dim2,dim3)
        l3_ = l2 @ theta5 #B x 25 x 25 x 51 = B x 51
        l3.append(l3_)
    l3 = torch.stack(l3,dim=1) # B x 3 x 51
    l3 = torch.nn.functional.softmax(l3,dim=2)
    return l3.squeeze()
