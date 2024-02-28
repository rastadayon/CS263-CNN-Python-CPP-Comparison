import numpy as np
from idx2numpy import convert_from_file
import matplotlib.pyplot as plt
import os

def normalize_images(images):
    normalized_images = []
    for image in images:
        min_val = np.min(image)
        max_val = np.max(image)
        
        normalized_image = (image - min_val) / (max_val - min_val)
        normalized_images.append(normalized_image)
    return normalized_images

def get_mnist(b_random=True):
    
    print("\n>> Getting MNIST datasets <<\n")

    all_images = np.array(convert_from_file('../../MNIST/train-images.idx3-ubyte'))
    all_labels = np.array(convert_from_file('../../MNIST/train-labels.idx1-ubyte'))
    test_images = np.array(convert_from_file('../../MNIST/t10k-images.idx3-ubyte'))
    test_labels = np.array(convert_from_file('../../MNIST/t10k-labels.idx1-ubyte'))

    
    all_images = normalize_images(all_images)
    test_images= normalize_images(test_images)

    if b_random : 
        lst = [i for i in range(0,len(all_images))]
        np.random.shuffle(lst)
        for index in range(0, len(lst) - 2, 2):
            tmp=all_images[index] 
            all_images[index] = all_images[index + 1] 
            all_images[index + 1] = tmp 
            tmp=all_labels[index] 
            all_labels[index] = all_labels[index + 1] 
            all_labels[index + 1] = tmp 

    #the dataset has no depth, it is necessary to add the depth dimension
    sample_list, test_list = [], [] 
                                                 
    for i in range(len(all_images)): sample_list.append(np.expand_dims(all_images[i], axis=0))
    for i in range(len(test_images)): test_list.append(np.expand_dims(test_images[i], axis=0))

    all_images = np.array(sample_list)
    test_images = np.array(test_list)


    valid_num = len(test_images)
    train_num = len(all_images) - len(test_images)

    train_images = np.array(all_images[0: train_num , : , :])
    train_labels = np.array(all_labels[ 0: train_num ])
    valid_images = np.array(all_images[train_num: , : , :])
    valid_labels = np.array(all_labels[train_num : ])
    
    return (train_images, train_labels, valid_images, valid_labels, test_images, test_labels)

def plot_random_images(images, title):
    random_indices = np.random.choice(len(images), 3, replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    fig.suptitle(title)

    for i, ax in enumerate(axes):
        ax.imshow(images[random_indices[i]].reshape(28, 28), cmap='gray')
        ax.axis('off')
    
    save_dir = './..'
    save_path = os.path.join(save_dir, f"{title.lower().replace(' ', '_')}_random_images.png")
    plt.savefig(save_path)    
