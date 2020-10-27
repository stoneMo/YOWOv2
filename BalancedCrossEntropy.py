import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


# Put a weight on positive samples
def WeightedCrossEntropy(logits,label,weight):
    '''
    :param logits:  net's output, which has reshaped [batch size,num_class]
    :param label:   Ground Truth which is ont hot encoing and has typr format of [batch size, num_class]
    :param weight:  a vector that describes every catagory's coefficent whose shape is (num_class,)
    :return: a scalar 
    '''
    loss = np.dot(np.log2(logits)*label,np.expand_dims(weight,axis=1)) + np.log2(1-logits) * (1-label)
    return loss.sum()