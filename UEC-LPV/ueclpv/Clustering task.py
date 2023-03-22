import numpy as np
import pandas as pd
import sklearn.cluster as cluster
from sklearn import mixture
import math

def clip(val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)

    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val

def rdist(x, y):
    """Reduced Euclidean distance.

    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2

    return result

from sklearn.utils.linear_assignment_ import linear_assignment
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

#compute the probability assignment for the second tech using smooth sterngthing membership of UMAP
def Q(embedding, centroids):

    q=np.zeros((embedding.shape[0],centroids.shape[0]))
    for i in range(embedding.shape[0]):
        data=embedding[i]
        for k in range(centroids.shape[0]):
            dist_squared = rdist(data, centroids[k])
            q[i,k]=pow((1.0 +dist_squared),-1.0)
            #q[i, k] = pow((1.0 + dist_squared), -1.0)
    q = (q.T / q.sum(axis=1)).T

    return q

#computethe atrget variable P
def P(q):
    weight = q ** 2 / (q.sum(0))
    return (weight.T / (weight.sum(1))).T

def clustering_layout_adam(embedding, centroids):
    initial_alpha=0.1
    alpha=initial_alpha
    beta_1 = 0.9
    beta_2 = 0.999  # initialize the values of the parameters
    epsilon = 1e-8
    dim=embedding.shape[1]
    n_samples=embedding.shape[0]
    n_clusters=centroids.shape[0]
    m_tp=np.zeros((n_samples, dim))
    v_tp=np.zeros((n_samples, dim))
    m_tc = np.zeros((n_clusters, dim))
    v_tc = np.zeros((n_clusters, dim))
    q=Q(embedding, centroids)
    p=P(q)
    for i in range(embedding.shape[0]):
        #embedding[i], m_tp[i], v_tp[i] = update_points(embedding[i], centroids, dim, n_clusters, p[i], q[i], alpha, m_tp[i], v_tp[i], pow(beta_1, n), pow(beta_2, n), epsilon)
        for k in range(n_clusters):

            centroid=centroids[k]
            dist_squared = rdist(embedding[i], centroid)
            if dist_squared > 0.0:
                grad_coeff = pow((dist_squared+ 1.0), -1.0)
                grad_coeff *= 2.0*(p[i, k] - q[i, k])
            else:
                grad_coeff = 0.0

        for d in range(dim):
            grad_d = clip(
                grad_coeff * (embedding[i, d] - centroid[d]))
            m_tp[i, d] = beta_1 * m_tp[i, d] + (1 - beta_1) * grad_d  # updates the moving averages of the gradient
            v_tp[i, d] = beta_2 * v_tp[i, d] + (1 - beta_2) * (grad_d * grad_d)  # updates the moving averages of the squared gradient
            m_cap = m_tp[i, d] / (1 - beta_1)  # calculates the bias-corrected estimates
            v_cap = v_tp[i, d] / (1 - beta_2)  # calculates the bias-corrected estimates
            embedding[i, d]+=(alpha * m_cap) / (math.sqrt(v_cap) + epsilon)
            #embedding[i, d] -= grad_d * alpha
    #centroids, m_tc, v_tc= update_centroids(embedding, centroids, dim, n_clusters, p, q, alpha, m_tc, v_tc, pow(beta_1, n), pow(beta_2, n), epsilon)
    for k in range(n_clusters):

        centroid = centroids[k]
        m_t=m_tc[k]
        v_t=v_tc[k]
        for i in range(embedding.shape[0]):
            x = embedding[i]
            dist_squared = rdist(x, centroid)
            if dist_squared > 0.0:
                grad_coeff = pow((dist_squared + 1.0), -1.0)
                grad_coeff *= -2.0*(p[i, k] - q[i,k])
            else:
                grad_coeff = 0.0

        for d in range(dim):
            grad_d = clip(
                grad_coeff * (x[d] - centroid[d]))
            m_t[d] = beta_1 * m_t[d] + (1 - beta_1) * grad_d  # updates the moving averages of the gradient
            v_t[d] = beta_2 * v_t[d] + (1 - beta_2) * (grad_d * grad_d)  # updates the moving averages of the squared gradient
            m_cap = m_t[d] / (1 - beta_1)  # calculates the bias-corrected estimates
            v_cap = v_t[d] / (1 - beta_2)  # calculates the bias-corrected estimates
            centroid[d]-=(alpha * m_cap) / (math.sqrt(v_cap) + epsilon)
            #centroid[d] -= grad_d * alpha
    return embedding, centroids

def clustering_layout_SGD(embedding, centroids):
    initial_alpha=10.0
    alpha=initial_alpha
    beta_1 = 0.9
    dim=embedding.shape[1]
    n_clusters=centroids.shape[0]
    q=Q(embedding, centroids)
    p=P(q)
    for i in range(embedding.shape[0]):
        for k in range(n_clusters):

            centroid=centroids[k]
            dist_squared = rdist(embedding[i], centroid)
            if dist_squared > 0.0:
                grad_coeff = pow((dist_squared + 1.0), -1.0)
                grad_coeff *= 2.0*(p[i, k] - q[i, k])
            else:
                grad_coeff = 0.0

        for d in range(dim):
            grad_d = clip(grad_coeff * (embedding[i, d] - centroid[d]))
            embedding[i, d] += grad_d * alpha + beta_1
    for k in range(n_clusters):
      centroid = centroids[k]
      for i in range(embedding.shape[0]):
          x = embedding[i]
          dist_squared = rdist(x, centroid)
          if dist_squared > 0.0:
              grad_coeff = pow((dist_squared + 1.0), -1.0)
              grad_coeff *= -2.0*(p[i, k] - q[i,k])
          else:
              grad_coeff = 0.0

      for d in range(dim):
          grad_d = clip(
              grad_coeff * (x[d] - centroid[d])
          )
          centroid[d] += grad_d * alpha +beta_1
    return embedding, centroids

tol=1e-5
ne=40
n_clusters=10

#GT = pd.read_csv("datasets/MNIST/mnist_label.csv")

#embedding = pd.read_csv("datasets/USPS/usps_ae_CAE.csv")
#GT = pd.read_csv("datasets/USPS/usps_y.csv")

embedding = pd.read_csv("datasets/STL/stl10_resnet50_CAE.csv")
GT = pd.read_csv("datasets/STL/stl10_y.csv")

GT=np.array(GT)
embedding=np.array(embedding)
GT=GT[2:,0]
print(np.shape(embedding))
print(np.shape(GT))

kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0)
y_pred = kmeans.fit_predict(embedding)
centroids = kmeans.cluster_centers_
y_pred_last = np.copy(y_pred)
for n in range(ne):
    if n==0:
        """GMM = mixture.GaussianMixture(n_components=n_clusters)
        GMM.fit(embedding)
        GMM_labels = GMM.predict_proba(embedding)
        y_pred = GMM_labels.argmax(1)
        centroids=GMM.means_"""

        kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0)
        y_pred = kmeans.fit_predict(embedding)
        centroids = kmeans.cluster_centers_
        y_pred_last = np.copy(y_pred)

        #centroids = np.random.rand(n_clusters, embedding.shape[1])
        centroids=centroids.astype('float32')
        #y_pred=np.zeros(embedding.shape[0])
        #embedding, centroids = clustering_layout_adam(embedding, centroids)
        #embedding, centroids = clustering_layout_SGD(embedding, centroids)
    else:
        """kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        y_pred = kmeans.fit_predict(embedding)
        centroids = kmeans.cluster_centers_
        y_pred_last = np.copy(y_pred)

        # centroids = np.random.rand(13, embedding.shape[1])
        centroids = centroids.astype('float32')"""
        embedding, centroids = clustering_layout_SGD(embedding, centroids)
        #embedding, centroids = clustering_layout_adam(embedding, centroids)
    print(
        "\tcompleted ", n, " / ", ne, "epochs")
    y_pred=Q(embedding, centroids)
    #y_pred = Q(embedding,centroids, a, b)
    y_pred = y_pred.argmax(1)
    acc1 = np.round(cluster_acc(GT, y_pred), 5)
    print("Accuracy1 of JoLMaProC : ", acc1)

    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
    y_pred_last = np.copy(y_pred)
    if n > 0 and delta_label < tol:
        print('delta_label ', delta_label, '< tol ', tol)
        print('Reached tolerance threshold. Stopping training.')
        break