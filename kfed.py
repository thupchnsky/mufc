#! /usr/bin/env python
# -*- coding: utf-8 -*-

#############################
# This file implements the k-fed, and it is adapted from the publicly available implementations.
#############################

import os
import os.path as osp
import time
import pickle
import argparse
import numpy as np
import scipy
import scipy.sparse as sps
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances as sparse_cdist
from sklearn.utils.extmath import randomized_svd
from model import *
from utils import *


def distance_to_set(A, S, sparse=False):
    """
        S is a list of points. Distance to set is the minimum distance of $x$ to
        points in $S$. In this case, this is computed for each row of $A$.  Note
        that this works with sparse matrices (sparse=True)
        Returns a single array of length len(A) containing corresponding distances.
        Input:
            A: numpy array
            S: numpy array
    """
    # Pair wise distances
    if sparse is False:
        pd = scipy.spatial.distance.cdist(A, S, metric='euclidean')
    else:
        pd = sparse_cdist(A, S)
    dx = np.min(pd, axis=1)
    assert dx.size == A.shape[0]
    return dx


def get_clustering(A, centers, sparse=False):
    """
        Returns a list of integers of length len(A). Each integer is an index which
        tells us the cluster A[i] belongs to. A[i] is assigned to the closest
        center.
    """
    # Pair wise distances
    if sparse is False:
        pd = scipy.spatial.distance.cdist(A, centers, metric='euclidean')
    else:
        pd = sparse_cdist(A, centers)
    assert np.allclose(pd.shape, [A.shape[0], len(centers)])
    indices = np.argmin(pd, axis=1)
    assert len(indices) == A.shape[0]
    return np.array(indices)


def kmeans_cost(A, centers, sparse=False, remean=False):
    """
        Computes the k means cost of rows of $A$ when assigned to the nearest
        centers in `centers`.
        remean: If remean is set to True, then the kmeans cost is computed with
        respect to the actual means of the clusters and not necessarily the centers
        provided in centers argument (which might not be actual mean of the
        clustering assignment).
    """
    clustering = get_clustering(A, centers, sparse=sparse)
    cost = 0
    if remean is True:
        # We recompute mean based on assignment.
        centers2 = []
        for clusterid in np.unique(clustering):
            points = A[clustering == clusterid]
            centers2.append(np.mean(points, axis=0))
        centers = np.array(centers2)
    for clusterid in np.unique(clustering):
        points = A[clustering == clusterid]
        dist = distance_to_set(points, centers, sparse=sparse)
        cost += np.mean(dist**2)
    return cost


def kmeans_pp(A, k, weighted=True, sparse=False, verbose=False):
    """
        Returns $k$ initial centers based on the k-means++ initialization scheme.
        With weighted set to True, we have the standard algorithm. When weighted is
        set to False, instead of picking points based on the D^2 distribution, we
        pick the farthest point from the set (careful deterministic version --
        affected by outlier points). Note that this is not deterministic.
        A: nxd data matrix (sparse or dense). 
        k: is the number of clusters.
        Returns a (k x d) dense matrix.
        K-means ++
        ----------
         1. Choose one center uniformly at random among the data points.
         2. For each data point x, compute D(x), the distance between x and
            the nearest center that has already been chosen.
         3. Choose one new data point at random as a new center, using a
            weighted probability distribution where a point x is chosen with
            probability proportional to D(x)2.
         4. Repeat Steps 2 and 3 until k centers have been chosen.
    """
    n, d = A.shape
    if n <= k:
        if sparse:
            A = A.toarray()
        return np.aray(A)
    index = np.random.choice(n)
    if sparse is True:
        B = np.squeeze(A[index].toarray())
        assert len(B) == d
        inits = [B]
    else:
        inits = [A[index]]
    indices = [index]
    t = [x for x in range(A.shape[0])]
    distance_matrix = distance_to_set(A, np.array(inits), sparse=sparse)
    distance_matrix = np.expand_dims(distance_matrix, axis=1)
    while len(inits) < k:
        if verbose:
            print('\rCenter: %3d/%4d' % (len(inits) + 1, k), end='')
        # Instead of using distance to set we can compute this incrementally.
        dx = np.min(distance_matrix, axis=1)
        assert dx.ndim == 1
        assert len(dx) == n
        dx = dx**2 / np.sum(dx**2)
        if weighted:
            choice = np.random.choice(t, 1, p=dx)[0]
        else:
            choice = np.argmax(dx)
        if choice in indices:
            continue
        if sparse:
            B = np.squeeze(A[choice].toarray())
            assert len(B) == d
        else:
            B = A[choice]
        inits.append(B)
        indices.append(choice)
        last_center = np.expand_dims(B, axis=0)
        assert last_center.ndim == 2
        assert last_center.shape[0] == 1
        assert last_center.shape[1] == d
        dx = distance_to_set(A, last_center, sparse=sparse)
        assert dx.ndim == 1
        assert len(dx) == n
        dx = np.expand_dims(dx, axis=1)
        a = [distance_matrix, dx]
        distance_matrix = np.concatenate(a, axis=1)
    if verbose:
        print()
    return np.array(inits)


def awasthisheffet(A, k, useSKLearn=False, sparse=False, max_iters=100):
    """
        The implementation here uses kmeans++ (i.e. probabilistic) to get initial centers instead of using a 10-approx algorithm.
        1. Project onto $k$ dimensional space.
        2. Use $k$-means++ to initialize.
        3. Use 1:3 distance split to improve initialization.
        4. Run Lloyd steps and return final solution.
        Returns a sklearn.cluster.Kmeans object with the clustering information and
        the list $S_r$.
    """
    assert A.ndim == 2
    n = A.shape[0]
    d = A.shape[1]
    # If we don't have $k$ points then return the matrix as its the best $k$
    # partition trivially.
    if n <= k:
        if sparse:
            A = np.array(A.toarray())
        return A, None
    # This works with sparse and dense matrices. Returns dense always.
    # Randomized though so average.
    U, Sigma, V = randomized_svd(A, n_components=k, random_state=None)
    # Columns of $V$ are eigen vectors
    V = V.T[:, :k]
    # Sparse and dense compatible. A_hat is always dense.
    A_hat = A.dot(V)
    inits = kmeans_pp(A_hat, k, sparse=False)
    # Run STEP 2, modified Lloyd. We have vectorized it for speed up.
    if sparse is False:
        pd = scipy.spatial.distance.cdist(inits, A_hat)
    else:
        pd = sparse_cdist(inits, A_hat)
    Sr_list = []
    for r in range(k):
        th = 3 * pd[r, :]
        remaining_dist = pd[np.arange(k) != r]
        assert np.allclose(remaining_dist.shape, [k - 1, n])
        indicator = (remaining_dist - th) < 0
        indicator = np.sum(indicator.astype(int), axis=0)
        assert len(indicator) == n
        # places where indicator is 0 is our set
        Sr = [i for i in range(len(indicator)) if indicator[i] == 0]
        assert len(Sr) >= 0
        Sr_list.append(Sr)
    # We don't mind lloyd_init being dense. Its only k x d.
    lloyd_init = np.array([np.mean(A_hat[Sr], axis=0) for Sr in Sr_list])
    assert np.allclose(lloyd_init.shape, [k, k])
    # Project back to d dimensional space
    lloyd_init = np.matmul(lloyd_init, V.T)
    assert np.allclose(lloyd_init.shape, [k, d])
    # Run Lloyd's method
    if useSKLearn:
        # Works with sparse matrices as well.
        kmeans = KMeans(n_clusters=k, init=lloyd_init, max_iter=max_iters)
        kmeans.fit(A)
        ret = (kmeans.cluster_centers_, kmeans.labels_)
    else:
        kmeans = MyKmeans(k=k, max_iters=max_iters)
        kmeans.set_centroids(lloyd_init)
        kmeans.run(A, pre_init=True)
        ret = (kmeans.centroids, kmeans.assignments)
    return ret


def kfed(x_dev, dev_k, k, useSKLearn=False, sparse=False, max_iters=100):
    """
        The full decentralized algorithm.
        Warning: Synchronous version, no parallelization across devices. Since the
        sklearn k means routine is itself parallel. 
        x_dev: [Number of devices, data length, data dimension]
        dev_k: Device k (int). The value $k'$ in the paper. Number of clusters
            per device. We use constant for all devices.
        Returns: Local estimators (local centers), central-centers
    """

    def cleaup_max(local_estimators, k, dev_k, useSKLearn=False, sparse=False):
        """
            Central cleanup phase based on the max-from-set rule.
            Switch to either percentile rule or probabilistic (kmeans++) rule in
            case of outlier points.
        """
        assert local_estimators.ndim == 2
        # The first dev_k points definitely in different target clusters.
        init_centers = local_estimators[:dev_k, :]
        remaining_data = local_estimators[dev_k:, :]
        # For the remaining initialization, use max rule.
        while len(init_centers) < k:
            distances = distance_to_set(remaining_data,
                                        np.array(init_centers),
                                        sparse=sparse)
            candidate_index = np.argmax(distances)
            candidate = remaining_data[candidate_index:candidate_index + 1, :]
            # Combine with init_centers
            init_centers = np.append(init_centers, candidate, axis=0)
            # Remove from remaining_data
            remaining_data = np.delete(remaining_data, candidate_index, axis=0)

        assert len(init_centers) == k
        # Perform final clustering.
        if useSKLearn:
            # Works with sparse matrices as well.
            kmeans = KMeans(n_clusters=k,
                            init=init_centers,
                            n_init=1,
                            max_iter=1)  # one round of Lloyd’s iteration
            kmeans.fit(local_estimators)
            ret = (kmeans.cluster_centers_, kmeans.labels_)
        else:
            kmeans = MyKmeans(k=k,
                              max_iters=1)  # one round of Lloyd’s iteration
            kmeans.set_centroids(init_centers)
            kmeans.run(local_estimators, pre_init=True)
            ret = (kmeans.centroids, kmeans.assignments)
        return ret

    num_dev = len(x_dev)
    msg = "Not enough devices "
    msg += "(num_dev=%d, dev_k=%d, k=%d)" % (num_dev, dev_k, k)
    assert dev_k * num_dev >= k, msg
    # Run local $k$-means
    local_clusters = []
    local_assignments = []
    for dev in x_dev:
        cluster_centers, cluster_assignments = awasthisheffet(
            dev,
            dev_k,
            useSKLearn=useSKLearn,
            sparse=sparse,
            max_iters=max_iters)
        local_clusters.append(cluster_centers)
        local_assignments.append(cluster_assignments)
    # This is alwasys dense.
    local_estimates = np.concatenate(local_clusters, axis=0)
    msg = "Not enough estimators. "
    msg += "Estimator matrix size: " + str(local_estimates.shape) + ", while "
    msg += "k = %d" % k
    assert local_estimates.shape[0] > k, msg
    # Local estimators are dense, allow sklearn.KMeans to be used at server side
    centers, server_assignments = cleaup_max(local_estimates,
                                             k,
                                             dev_k,
                                             useSKLearn=useSKLearn,
                                             sparse=False)
    # compute the induced loss
    induced_kfed_loss = 0.0
    for i_client, dev in enumerate(x_dev):
        client_assignments = local_assignments[
            i_client] + dev_k * i_client  # this is the correct index in server_assignments
        induced_kfed_loss += induced_loss(
            dev, centers, server_assignments[client_assignments])
    return local_estimates, centers, induced_kfed_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clusters", type=int, default=10)
    parser.add_argument("--num_clients", type=int)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--k_prime", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--max_iters", type=int, default=300)
    parser.add_argument("--result_path", type=str, default='result')
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='verbosity of loss')
    args = parser.parse_args()

    dataset = load_dataset(args.data_path)  # dataset is a json file
    print("Processing dataset:", args.data_path)

    num_dim = dataset["full_data"].shape[1]
    assert num_dim >= args.k_prime, "current implementation of kfed requires num_dim >= k_prime"

    if not osp.exists(args.result_path):
        os.makedirs(args.result_path)

    kfed_data = []
    for i_client in range(args.num_clients):
        kfed_data.append(dataset["client_" + str(i_client)])

    kfed_local_induced_loss = np.zeros((args.num_trials, ))
    kfed_time = np.zeros((args.num_trials, ))
    for i_trail in range(args.num_trials):
        start = time.time()
        server_data, server_centroids, induced_kfed_loss = kfed(
            kfed_data,
            args.k_prime,
            args.num_clusters,
            max_iters=args.max_iters)
        kfed_local_induced_loss[i_trail] = induced_kfed_loss
        kfed_time[i_trail] = time.time() - start
        if args.verbose:
            print(
                f"Loss: {kfed_local_induced_loss[i_trail]}, Time used: {kfed_time[i_trail]}"
            )
    kfed_res = {"loss": kfed_local_induced_loss, "time": kfed_time}

    dataset_name = args.data_path.split("/")[-1]
    with open(os.path.join(args.result_path, "kfed_" + dataset_name),
              "wb") as f:
        pickle.dump(kfed_res, f)
