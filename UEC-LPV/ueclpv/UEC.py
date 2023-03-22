import h5py
#from keras.datasets import mnist, reuters, cifar10
from sklearn import datasets

import pandas as pd
import matplotlib.pyplot as plt
import time

# Dimension reduction and clustering libraries
import ueclpv
from sklearn.metrics import normalized_mutual_info_score
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np
import sklearn.cluster as cluster
from sklearn import mixture
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
#from keras.preprocessing.text import Tokenizer
from scipy.spatial.distance import euclidean, cdist, pdist, squareform
import collections

"""(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape((x.shape[0], -1))
y=np.array(y)
print(np.shape(y))
y = y.reshape((y.shape[0]))
x = np.divide(x, 255.)"""
#np.savetxt('datasets/mnist_label.csv', y, delimiter=',', fmt='%f')

"""(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape((x.shape[0], -1))
y=np.array(y)
y = y.reshape((y.shape[0]))
print(np.shape(y))"""

"""(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=2000, test_split=0.2, seed=113)

tokenizer = Tokenizer(num_words=2000)
x_train = tokenizer.sequences_to_matrix(x_train)
x_test = tokenizer.sequences_to_matrix(x_test)

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape((x.shape[0], -1))
print(x[0])"""

#y=np.argmax(y, axis=1)

def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
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

def get_tptnfpfn(preds, y):
    #preds = clf.predict(data['x_test'])

    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for real_line, pred_line in zip(y, preds):
        for real, pred in zip(real_line, pred_line):
            if pred == 1:
                if real == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if real == 1:
                    fn += 1
                else:
                    tn += 1
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

def get_accuracy(data):
    sum_ = (data['tp'] + data['tn'] + data['fp'] + data['fn'])
    return float(data['tp'] + data['tn']) / sum_

#Load the digits dataset
"""digits = load_digits()
d = scale(digits.data)
y = digits.target"""

"""iris = datasets.load_iris()
d = iris.data
y = iris.target"""

"""with h5py.File("C:/Users/GOOD DAY/PycharmProjects/Datasets and AE whieghts/datasets/USPS/usps.h5", 'r') as hf:
    train = hf.get('train')
    X_tr = train.get('data')[:]
    y_tr = train.get('target')[:]
    test = hf.get('test')
    X_te = test.get('data')[:]
    y_te = test.get('target')[:]
    d = np.concatenate((X_tr, X_te), axis=0)
    y = np.concatenate((y_tr, y_te), axis=0)"""

# Datasets
#d = pd.read_csv("D:/memoire_Doctorat/Datasets/mnist_784.csv")
#d=pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/pendigits_ae_DEC.csv")

#d = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/coil100.csv")
#y = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/coil100_label.csv")

d = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/USPS/usps_ae_CAE.csv")
#d = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/usps_ae_DEC.csv")
y = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/USPS/usps_y.csv")

#d = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/MNIST/mnist_ae_DEC.csv")
#d = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/MNIST/mnist_CAE.csv")
#y = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/MNIST/mnist_label.csv")

#d = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/nltk_reuters_ae_DEC.csv")
#y = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/nltk_reuters_labels.csv")

#d = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/CIFAR10/cifar10_CAE.csv")
#d = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/CIFAR10/cifar10_resnet50.csv")
#d = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/CIFAR10/cifar10_resnet50_DEC.csv")
#d = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/CIFAR10/cifar10_vgg16.csv")
#d = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/CIFAR10/cifar10_vgg16_CAE.csv")
#d = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/CIFAR10/cifar10_vgg16_DEC.csv")
#y = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/CIFAR10/cifar10_y.csv")

#d = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/stl10_vgg16.csv")
#d = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/stl10_vgg16_ae_DEC.csv")
#d = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/STL/stl10_resnet50_CAE.csv")
#y = pd.read_csv("D:/PycharmProjects/Datasets and AE whieghts/datasets/STL/stl10_y.csv")
#x=np.array(d)

d=np.array(d)
#d=d[:60000]
#d=data[:, :3071]
#y=data[:, 3072]
print(np.shape(d))
y=np.array(y)
#y=y[:60000]
#y1=np.argmax(y, axis=1)
#counter=collections.Counter(y1)
#print(counter)
y = y.reshape((y.shape[0]))
print(np.shape(y))
print(y)
#print(y1)
masked_target = y.copy().astype(np.int8)
#masked_target[np.random.choice(59999, size=9500, replace=False)] = -1

kmeans = cluster.KMeans(n_clusters=10)
GMM = mixture.GaussianMixture(n_components=10)

y_pred_kmeans = kmeans.fit_predict(d)
print("NMI: ",normalized_mutual_info_score(y, y_pred_kmeans))

acc = np.round(cluster_acc(y, y_pred_kmeans),5)
print("Accuracy kmeans labels: ",acc)

"""GMM.fit(d)
GMM_labels=GMM.predict_proba(d)
GMM_labels=GMM_labels.argmax(1)
print("NMI: ",normalized_mutual_info_score(y, GMM_labels))

acc = np.round(cluster_acc(y, GMM_labels),5)
print("Accuracy GMM labels: ",acc)"""

time_start = time.time()
embedding, y_pred, centroids, X, Y, Z = ueclpv.UEC_LPV(n_neighbors=50, n_clusters=10, min_dist=0.0, verbose=True).fit_transform(d)

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
fig = plt.figure(figsize=(20, 15))
ax = fig.gca(projection = '3d')
# Set rotation angle to 30 degrees
ax.view_init(azim=30)
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.set_xlabel('X', fontsize = 20)
ax.set_ylabel('Y', fontsize = 20)
ax.set_zlabel('CE (X, Y)', fontsize = 20)
ax.set_zlim(0, 4.3)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

"""np.savetxt('D:/PycharmProjects/Datasets and AE whieghts/datasets/USPS/USPS_embedding.csv', embedding, delimiter=',', fmt='%f')
np.savetxt('D:/PycharmProjects/Datasets and AE whieghts/datasets/USPS/USPS_centroids.csv', centroids, delimiter=',', fmt='%f')

print('JoLMaProC runtime: {} seconds'.format(time.time()-time_start))
y_pred=np.argmax(y_pred, axis=1)
print(np.shape(y_pred))

#reuters: remember for reuters dataset you should set in the acc1 measure y1[10786, 90] and the y_pred[10786]
#in acc2 measure y1[10786, 90] and the y_pred[10786, 90]
#for nmi measure y[10786] and the y_pred[10786]

# this for reuters######
#y1=np.argmax(y, axis=1)#
########################
acc1 = np.round(cluster_acc(y, y_pred),5)
print("Accuracy1 JoLMaProC: ",acc1)
#print("NMI JoLMaProC: ",normalized_mutual_info_score(y, y_pred))

#res=get_tptnfpfn(y_pred, y1)
#acc2=(get_accuracy(res) * 100)
#print("Accuracy2: ",acc2)

y_pred_kmeans = kmeans.fit_predict(embedding)
#centroids=kmeans.cluster_centers_

acc = np.round(cluster_acc(y, y_pred_kmeans),5)
print("Accuracy kmeans labels: ",acc)
#print("NMI kmeans labels: ",normalized_mutual_info_score(y, y_pred_kmeans))

GMM.fit(embedding)
GMM_labels=GMM.predict_proba(embedding)
GMM_labels=GMM_labels.argmax(1)

acc = np.round(cluster_acc(y, GMM_labels),5)
print("Accuracy GMM labels: ",acc)
#print("NMI GMM labels: ",normalized_mutual_info_score(y, GMM_labels))

plt.scatter(embedding[:, 0], embedding[:, 1], c=y, s=1, cmap='Spectral')
plt.scatter(centroids[:,0], centroids[:, 1], s=3)
plt.show();

def db_index(X, y):
    # get unique labels
    if y.ndim == 2:
        y = np.argmax(y, axis=1)
    uniqlbls = np.unique(y)
    n = len(uniqlbls)
    # pre-calculate centroid and sigma
    centroid_arr = np.empty((n, X.shape[1]))
    sigma_arr = np.empty((n,1))
    dbi_arr = np.empty((n,n))
    mask_arr = np.invert(np.eye(n, dtype='bool'))
    for i,k in enumerate(uniqlbls):
        Xk = X[np.where(y==k)[0],...]
        Ak = np.mean(Xk, axis=0)
        centroid_arr[i,...] = Ak
        sigma_arr[i,...] = np.mean(cdist(Xk, Ak.reshape(1,-1)))
    # compute pairwise centroid distances, make diagonal elements non-zero
    centroid_pdist_arr = squareform(pdist(centroid_arr)) + np.eye(n)
    # compute pairwise sigma sums
    sigma_psum_arr = squareform(pdist(sigma_arr, lambda u,v: u+v))
    # divide
    dbi_arr = np.divide(sigma_psum_arr, centroid_pdist_arr)
    # get mean of max of off-diagonal elements
    dbi_arr = np.where(mask_arr, dbi_arr, 0)
    dbi = np.mean(np.max(dbi_arr, axis=1))
    return dbi
dbi=db_index(d, y_pred)
print("dbi= ", dbi)

SC=metrics.silhouette_score(d, y_pred, metric="euclidean")
print("silhouette score= ", SC)

CH=metrics.calinski_harabasz_score(d, y_pred)
print("calinski harabasz score= ", CH)"""