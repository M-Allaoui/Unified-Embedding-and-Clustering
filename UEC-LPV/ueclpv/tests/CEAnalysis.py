@numba.njit(fastmath=True, parallel=True)
def CEAnalysis(
        head_embedding,
        tail_embedding,
        head,
        tail,
        centroids,
        epochs_per_sample,
        P
):
    # print("hi")
    Q = np.zeros((head_embedding.shape[0], head_embedding.shape[0]))
    for i in range(epochs_per_sample.shape[0]):
        j = head[i]
        k = tail[i]

        current = head_embedding[j]
        other = tail_embedding[k]
        #prop=P[j,k]

        dist_squared = rdist(current, other)
        # * prop
        if dist_squared > 0.0:
            Q[j,k]=pow((1.0 +  dist_squared),-1.0)

        else:
            Q[j,k] = 0.0

    X, Y = np.meshgrid(P, Q)
    Z = np.exp(-X ** 2) * np.log(1 + Y ** 2) + (1 - np.exp(-X ** 2)) * np.log((1 + Y ** 2) / (Y ** 2 + 0.01))
    return Z