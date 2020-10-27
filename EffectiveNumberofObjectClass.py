import numpy as np

# NC used for minimum occurence of a label
def labels_to_class_weights(labels, beta, nc=80):
    # Get class weights (inverse frequency) from training labels
    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    lasses = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class
    weights[weights == 0] = 1  # replace empty bins with 1
    wpo = 1-(beta**weights)
    weights = wpo / (1-beta)
    return torch.Tensor(weights)

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

def ENOCCrossEntropy(logits,label,labels,beta=0.5):
    weights = labels_to_class_weights(labels, beta)
    return WeightedCrossEntropy(logits, label, weights)