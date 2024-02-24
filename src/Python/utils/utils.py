import numpy as np
from idx2numpy import convert_from_file

def get_mnist(b_random=True):
    
    print("\no Getting MNIST datasets\n")

    tot_samples = np.array(convert_from_file('MNIST/train-images.idx3-ubyte'))
    tot_labels = np.array(convert_from_file('MNIST/train-labels.idx1-ubyte'))
    test_samples = np.array(convert_from_file('MNIST/t10k-images.idx3-ubyte'))
    test_labels = np.array(convert_from_file('MNIST/t10k-labels.idx1-ubyte'))

    
    tot_samples = _normalize_set(tot_samples)
    test_samples= _normalize_set(test_samples)

    if b_random : 

        lst = [i for i in range(0,len(tot_samples))]
        np.random.shuffle(lst)
        for index in range(0, len(lst) - 2, 2):
            tmp=tot_samples[index] 
            tot_samples[index] = tot_samples[index + 1] 
            tot_samples[index + 1] = tmp 
            tmp=tot_labels[index] 
            tot_labels[index] = tot_labels[index + 1] 
            tot_labels[index + 1] = tmp 

    #the dataset has no depth, it is necessary to add the depth dimension
    sample_list, test_list = [], [] 
                                                 
    for i in range(len(tot_samples)): sample_list.append(np.expand_dims(tot_samples[i], axis=0))
    for i in range(len(test_samples)): test_list.append(np.expand_dims(test_samples[i], axis=0))

    tot_samples = np.array(sample_list)
    test_samples = np.array(test_list)


    valid_num = len(test_samples)
    train_num = len(tot_samples) - len(test_samples)

    train_samples = np.array(tot_samples[0:train_num , : , :])
    train_labels = np.array(tot_labels[ 0 : train_num ])
    valid_samples = np.array(tot_samples[train_num: , : , :])
    valid_labels = np.array(tot_labels[ train_num : ])
    
    return (train_samples, train_labels, valid_samples, valid_labels, test_samples, test_labels)
