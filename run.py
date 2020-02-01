from neuralnet import *
from checker import *
import matplotlib.pyplot as plt
import numpy as np

# Importing train images 
images, labels = load_data('./', mode = 'train')

# shuffling the training dataset
N = np.random.permutation(np.arange(labels.shape[0]))
images = images[N,:]
labels = labels[N]

# pick the first 10000 data for validation
val_images = images[0:10000,:]
val_labels = labels[0:10000]

# normalizing the images and one_hot encoding the labels (validation set)
val_images = normalize_data(val_images)
val_labels = one_hot_encoding(val_labels)

# leave the rest of the data for training
train_images = images[10000:,:]
train_labels = labels[10000:]

# normalizing the images and one_hot encoding the labels (train set)
train_images = normalize_data(train_images)
train_labels = one_hot_encoding(train_labels)

# Importing test images
test_images, test_labels = load_data('./', mode = 't10k')

# shuffling the test dataset
N = np.random.permutation(np.arange(test_labels.shape[0]))


test_images = test_images[N,:]
test_labels = test_labels[N]

# normalizing the images and one_hot encoding the labels (test set)
test_images = normalize_data(test_images)
test_labels = one_hot_encoding(test_labels)

config = load_config('./')
model = Neuralnetwork(config)

B = 101
N = train_images.shape[0]
num_epoch = 5

for epoch in range(num_epoch):
    
    indeces = np.random.permutation(np.arange(N))
    
    for i in range(int(N/B)):
        
        xtrain = train_images[indeces[B*i:B*(i+1)],:]
        ytrain = train_labels[indeces[B*i:B*(i+1)],:]
#         print('xtrain shape: ', xtrain.shape)
        #print(model.layers[0].w)
#         print('b shape: ', model.layers[0].b.shape)
        forwarded, loss = model.forward(xtrain.T, ytrain.T)
        print('loss: ', loss)
#         print('forwarded shape: ', forwarded.shape)
#         print(forwarded)
#         print(np.sum(forwarded[:, 1]))
        
        model.backward()
        break
    break