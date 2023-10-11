#! /usr/bin/env python
# -*- coding: utf-8 -*-

#############################
# utils files.
#############################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
import pickle
import math


def load_dataset(filepath):
    """
        Return:
            dataset: dict
    """
    with open(filepath, 'rb') as fr:
        dataset = pickle.load(fr)
    return dataset


def sample_points_in_bin(bin_mid, total_points, quant_eps):
    """
        Input:
            bin_mid: numpy.array (d,)
            total_points: points needed to be generated
            quant_eps: quantization region length
    """
    sampled_shifts = np.random.uniform(-quant_eps / 2.0,
                                       quant_eps / 2.0,
                                       size=[total_points, bin_mid.size])
    sampled_points = sampled_shifts + bin_mid
    return sampled_points


def clustering_loss(data, centroids):
    """
        Computes the clustering loss on a dataset given a fixed set of centroids
        Input:
            centroids: numpy.array (k,d)
            data: numpy.array (n,d)
    """
    loss = 0.0
    for i_data in range(data.shape[0]):
        d = np.linalg.norm(data[i_data, :] - centroids, axis=1)
        loss += np.min(d)**2
    return loss


def induced_loss(data, centroids, assignments):
    """
        Compute the loss based on the induced clustering results
        Input:
            centroids: numpy.array (k,d)
            data: numpy.array (n,d)
            assignments: numpy.array (n,). Values are between [0,k-1]
    """
    loss = 0.0
    for i_data in range(data.shape[0]):
        d = np.linalg.norm(data[i_data, :] - centroids[assignments[i_data], :])
        loss += d**2
    return loss


def induced_loss_return_max(data, centroids, assignments):
    """
        Compute the loss based on the induced clustering results
        Input:
            centroids: numpy.array (k,d)
            data: numpy.array (n,d)
            assignments: numpy.array (n,). Values are between [0,k-1]
    """
    loss = 0.0
    argmax_idx = -1
    max_loss = -1
    for i_data in range(data.shape[0]):
        d = np.linalg.norm(data[i_data, :] - centroids[assignments[i_data], :])
        loss += d**2
        if d > max_loss:
            max_loss = d
            argmax_idx = i_data
    return loss, argmax_idx


def split_data(data_combined,
               num_clusters,
               num_clients=None,
               split='iid',
               k_prime=None):
    json_data = {}
    # K-means optimal loss
    clf = KMeans(n_clusters=num_clusters).fit(data_combined)
    kmeans_loss = clf.inertia_
    kmeans_label = clf.labels_
    json_data['kmeans_loss'] = kmeans_loss

    if num_clients is None:
        num_clients = int(
            data_combined.shape[0] /
            100)  # make sure each client does not have too much data

    # initialize for each client
    for i in range(num_clients):
        json_data['client_' + str(i)] = []

    # iid split
    if split == 'iid':
        for k in range(num_clusters):
            data_cluster = data_combined[kmeans_label == k, :]
            size_per_client = math.floor(data_cluster.shape[0] / num_clients)
            for i in range(num_clients - 1):
                json_data['client_' + str(i)].append(
                    data_cluster[i * size_per_client:(i + 1) *
                                 size_per_client, :])
            # fill the rest into the last client
            json_data['client_' + str(num_clients - 1)].append(
                data_cluster[(num_clients - 1) * size_per_client:, :])

        tmp_count = 0
        # concatenate the data for all clients
        for i in range(num_clients):
            json_data['client_' + str(i)] = np.concatenate(
                json_data['client_' + str(i)], axis=0)
            tmp_count += json_data['client_' + str(i)].shape[0]
        # have a final check on the sizes
        assert tmp_count == data_combined.shape[
            0], "Error: data size does not match"
    # non-iid split
    elif split == 'non-iid':
        if k_prime is None:
            k_prime = int(num_clusters / 2)
        assert k_prime <= num_clusters, "Error: not valid k_prime"
        # first get data for each cluster
        data_by_cluster = {}
        data_by_cluster_used = [0] * num_clusters
        size_per_client = int(data_combined.shape[0] / num_clients)
        for k in range(num_clusters):
            data_by_cluster[k] = data_combined[kmeans_label == k, :]

        valid_cluster_idx = [k for k in range(num_clusters)]
        # first fill in the data for first n-1 clients
        for i in range(num_clients - 1):
            tmp_client_data = []
            tmp_client_size = 0
            tmp_client_clusters = np.random.choice(valid_cluster_idx,
                                                   min(k_prime,
                                                       len(valid_cluster_idx)),
                                                   replace=False)
            for tmp_client_cluster_idx in tmp_client_clusters:
                # some intermediate variables
                tmp_1 = data_by_cluster_used[tmp_client_cluster_idx]
                tmp_2 = data_by_cluster[tmp_client_cluster_idx].shape[0]
                if tmp_client_size < size_per_client and tmp_1 < tmp_2:
                    tmp_count = min([
                        np.random.randint(
                            int(size_per_client / k_prime) - 1,
                            size_per_client),
                        size_per_client - tmp_client_size, tmp_2 - tmp_1
                    ])
                    tmp_client_data.append(
                        data_by_cluster[tmp_client_cluster_idx][tmp_1:tmp_1 +
                                                                tmp_count, :])
                    # update each value
                    data_by_cluster_used[tmp_client_cluster_idx] += tmp_count
                    if data_by_cluster_used[tmp_client_cluster_idx] == tmp_2:
                        valid_cluster_idx.remove(
                            tmp_client_cluster_idx
                        )  # will not selected by future clients
                    tmp_client_size += tmp_count
                    if tmp_client_size == size_per_client:
                        break
            json_data['client_' + str(i)] = np.concatenate(tmp_client_data,
                                                           axis=0)
        # leave all other data points to the last client
        cluster_size_last_client = 0
        tmp_client_data = []
        for k in range(num_clusters):
            if data_by_cluster_used[k] < data_by_cluster[k].shape[0]:
                tmp_client_data.append(
                    data_by_cluster[k][data_by_cluster_used[k]:, :])
                cluster_size_last_client += 1
        assert cluster_size_last_client <= k_prime, "Error: k_prime is violated"
        json_data['client_' + str(num_clients - 1)] = np.concatenate(
            tmp_client_data, axis=0)
        # have a final check on the sizes
        tmp_count = 0
        for i in range(num_clients):
            tmp_count += json_data['client_' + str(i)].shape[0]
        assert tmp_count == data_combined.shape[
            0], "Error: data size does not match"
    else:
        raise NotImplementedError

    return json_data


def generate_data(data_input,
                  num_clusters,
                  save_flag=True,
                  filename='processed_data.pkl',
                  num_clients=None,
                  split='iid',
                  k_prime=None):
    """
        Process, split and save data into .pkl files
    """
    json_data = {}
    json_data['full_data'] = data_input
    
    # normalize each dimension
    data_max = json_data['full_data'].max()
    if data_max > 1:
        json_data['full_data'] = json_data['full_data'] / float(data_max)
    
    json_data['num_clusters'] = num_clusters
    split_json = split_data(json_data['full_data'], num_clusters, num_clients, split, k_prime)
    json_data.update(split_json)
    print("data processed!")
    
    if save_flag:
        # save the data to file
        with open(filename, "wb") as fw:
            pickle.dump(json_data, fw)