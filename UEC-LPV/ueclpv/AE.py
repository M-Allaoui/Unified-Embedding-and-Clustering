import h5py
import pandas as pd
import numpy as np
import keras
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.initializers import VarianceScaling
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import normalized_mutual_info_score

"""(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape((x.shape[0], -1))
y = y.reshape((y.shape[0]))
x = np.divide(x, 255.)"""
"""(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=2000, test_split=0.2, seed=113)

tokenizer = Tokenizer(num_words=2000)
x_train = tokenizer.sequences_to_matrix(x_train)
x_test = tokenizer.sequences_to_matrix(x_test)

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape((x.shape[0], -1))
print(x[0])"""

def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(ind)):
            if ind[j][0] == y_pred[i]:
                best_fit.append(ind[j][1])
    return best_fit, ind, w


def cluster_acc(y_true, y_pred):
    _, ind, w = best_cluster_fit(y_true, y_pred)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

"""with h5py.File("datasets/usps.h5", 'r') as hf:
    train = hf.get('train')
    X_tr = train.get('data')[:]
    y_tr = train.get('target')[:]
    test = hf.get('test')
    X_te = test.get('data')[:]
    y_te = test.get('target')[:]
    x = np.concatenate((X_tr, X_te), axis=0)
    y = np.concatenate((y_tr, y_te), axis=0)"""

from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
#Load the digits dataset
#digits = load_digits()
#x = scale(digits.data)

#x = pd.read_csv("C:/Users/GOOD DAY/PycharmProjects/umap_clustering/umap/datasets/nltk_reuters.csv")
#x = pd.read_csv("C:/Users/GOOD DAY/PycharmProjects/umap_clustering/umap/datasets/coil100.csv")
#x = pd.read_csv("C:/Users/GOOD DAY/PycharmProjects/umap_clustering/umap/datasets/CORD19.csv")
#x = pd.read_csv("datasets/stl10_vgg16.csv")
x=pd.read_csv("datasets/cifar10_vgg16.csv")
x=np.array(x)
print(np.shape(x))
dims = [x.shape[-1], 500, 500, 2000, 100]
init = VarianceScaling(scale=1. / 3., mode='fan_in',
                           distribution='uniform')
pretrain_optimizer = SGD(lr=0.01, momentum=0.9)
pretrain_epochs = 50
batch_size = 50

autoencoder, encoder = autoencoder(dims, init=init)

path = F"C:/Users/GOOD DAY/PycharmProjects/umap_clustering/umap"
#autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
autoencoder.compile(optimizer='adam', loss='mse')#mse

#autoencoder.load_weights(path + '/AE_weights/ae_weights_cifar10_resnet50.h5')
autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs) #, callbacks=cb)

autoencoder.save_weights(path + '/AE_weights/ae_weights_cifar10_vgg16.h5')
#autoencoder.load_weights(path + '/ae_weights_reuters2.h5')
data=encoder.predict(x)

np.savetxt('datasets/cifar10_vgg16_DEC.csv', data, delimiter=',', fmt='%f')