import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import umap

plt.style.use('ggplot')

"""d = pd.read_csv("datasets/nltk_reuters_ae_DEC.csv")
y = pd.read_csv("datasets/nltk_reuters_labels.csv")
d=np.array(d)
y=np.array(y)
y=y[1:]
print(np.shape(d))
print(np.shape(y))
y=np.argmax(y, axis=1)
print(np.shape(y))"""
d = pd.read_csv("C:/Users/GOOD DAY/PycharmProjects/umap_clustering/umap/datasets/CORD19_ae_DEC.csv")
d=np.array(d)
print(np.shape(d))

# run kmeans with many different k
distortions = []
K = range(2, 50)
for k in K:
    k_means = KMeans(n_clusters=k, random_state=42, n_jobs=-1).fit(d)
    k_means.fit(d)
    distortions.append(sum(np.min(cdist(d, k_means.cluster_centers_, 'euclidean'), axis=1)) / d.shape[0])
    """embedding, y_pred = umap.UMAP(n_neighbors=50, n_clusters=k, min_dist=0).fit_transform(d)
    y_pred = np.argmax(y_pred, axis=1)
    distortions.append(normalized_mutual_info_score(y, y_pred))
    print('Found distortion for {} clusters'.format(k))"""

X_line = [K[0], K[-1]]
Y_line = [distortions[0], distortions[-1]]

# Plot the elbow
plt.plot(K, distortions, 'b-')
plt.plot(X_line, Y_line, 'r')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()