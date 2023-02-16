#! /usr/bin/env python
# -*- coding: utf-8 -*-

#############################
# This file implements the DC-Kmeans, and it is adapted from the publicly available implementations
# at https://github.com/tginart/deletion-efficient-kmeans.
#############################

import numpy as np
from utils import induced_loss
import pickle
import math
from tqdm import tqdm
from sklearn.cluster import KMeans
from model import MyKmeans
import time


class DCnode(MyKmeans):
    """
        A k-means subproblem for the divide-and-conquer tree
        in DC-k-means algorithm
    """

    def __init__(self, k, iters):
        MyKmeans.__init__(self, k, max_iters=iters)
        self.children = []
        self.parent = None
        self.node_data = set()
        self.data_prop = set()

    def _run_node(self, X):
        self._set_node_data(X)
        self._lloyd_iterations()

    def _set_node_data(self, X):
        self.data = X[list(self.node_data)]
        self._set_data(self.data)


class DCKmeans():

    def __init__(self, ks, widths, iters=10, n_init=10):
        """
            ks - list of k parameter for each layer of DC-tree
            widths - list of width parameter (number of buckets) for each layer
            iters - # of iterations to run
        """
        self.ks = ks
        self.widths = widths
        self.dc_tree = self._init_tree(ks, widths, iters, n_init)
        self.data_partition_table = dict()
        self.data = dict()
        self.dels = set()
        self.valid_ids = []
        self.d = 0
        self.n = 0
        self.h = len(self.dc_tree)
        for i in range(self.h):
            self.data[i] = None

    def run(self, dataset, assignments=True):
        """
            X - numpy matrix, n-by-d, each row is a data point
            assignments (optional) - bool flag, computes assignments and loss
                NOTE: Without assignments flag, this only returns the centroids
            OUTPUT:
            centroids - k-by-d matrix of centroids
                IF assignments FLAG IS SET ALSO RETURNS:
            assignments - Vector of length n, with datapoint to center assignments
            loss - The loss of the final partition
        """
        X = dataset["full_data"]
        self._init_data(X)

        # random partition
        # self._partition_data(X)

        # fixed partition
        self._set_partition_data(dataset)

        self._run()
        if assignments:
            # self.centroids is the root node (server) centroids
            # we should use this to compute the induced loss
            # note that we only consider the two-layer structure here
            if self.h != 2:
                raise NotImplementedError
            # self.assignments is the root node (server) assignment
            # the root node now is a sklearn.KMeans class
            # self.assignments = self.dc_tree[0][0].assignments
            self.assignments = self.dc_tree[0][0].labels_
            local_induced_loss = 0.0
            leaf_start_idx = 0
            global_assignment = []
            for leaf_id in range(len(self.dc_tree[-1])):
                leaf = self.dc_tree[-1][leaf_id]  # DCnode
                leaf_global_assignment = self.assignments[leaf.assignments +
                                                          leaf_start_idx]
                leaf_start_idx += self.ks[1]
                local_induced_loss += induced_loss(leaf.data, self.centroids,
                                                   leaf_global_assignment)
                global_assignment.append(leaf_global_assignment)
            global_assignment = np.concatenate(global_assignment)
            return self.centroids, global_assignment, local_induced_loss
        return self.centroids

    def delete(self, del_idx, assignments=True):
        """
            del_idx is the index in the updated mapping, not the index in the original mapping
            idx = self.valid_ids[del_idx] is the real removed points wrt the original data_full
        """
        start = time.time()
        idx = self.valid_ids[del_idx]
        self.valid_ids.pop(del_idx)
        self.dels.add(idx)
        node = self.dc_tree[-1][self.data_partition_table[idx]]
        node.node_data.remove(idx)
        l = self.h - 1
        self.n -= 1
        while True:
            if node.parent == None:
                # root node, sklearn.KMeans object
                # get the data for root node
                # root_data = np.vstack([self.data[0][root_data_idx] for root_data_idx in range(self.widths[1]*self.ks[1])])
                node.fit(
                    self.data[0]
                )  # fit new data to sklearn.KMeans will reset parameters automatically
                self.centroids = node.cluster_centers_
                break
            # otherwise node is normal DCnode
            node._init_placeholders()  # need to reset manually for MyKmeans
            node._run_node(self.data[l])
            data_prop = list(node.data_prop)
            for c_id in range(len(node.centroids)):
                idx = data_prop[c_id]
                self.data[l - 1][idx] = node.centroids[c_id]
            node = node.parent
            l -= 1
        removal_time = time.time() - start
        # compute the induced loss after update
        if assignments:
            # self.centroids is the root node (server) centroids
            # we should use this to compute the induced loss
            # note that we only consider the two-layer structure here
            if self.h != 2:
                raise NotImplementedError
            # self.assignments is the root node (server) assignment
            self.assignments = self.dc_tree[0][0].labels_
            local_induced_loss = 0.0
            leaf_start_idx = 0
            global_assignment = []
            for leaf_id in range(len(self.dc_tree[-1])):
                leaf = self.dc_tree[-1][leaf_id]  # DCnode
                leaf_global_assignment = self.assignments[leaf.assignments +
                                                          leaf_start_idx]
                leaf_start_idx += self.ks[1]
                local_induced_loss += induced_loss(leaf.data, self.centroids,
                                                   leaf_global_assignment)
                global_assignment.append(leaf_global_assignment)
            global_assignment = np.concatenate(global_assignment)
            return self.centroids, global_assignment, local_induced_loss, removal_time
        return self.centroids, removal_time

    def _init_data(self, X):
        self.n = len(X)
        self.valid_ids = list(range(self.n))
        self.d = len(X[0])
        data_layer_size = self.n
        for i in range(self.h - 1, -1, -1):
            self.data[i] = np.zeros((data_layer_size, self.d))
            data_layer_size = self.ks[i] * self.widths[i]

    def _partition_data(self, X):
        """
            Random data partition for leaves --> layer self.h-1
        """
        self.d = len(X[0])
        num_leaves = len(self.dc_tree[-1])
        for i in range(len(X)):
            leaf_id = np.random.choice(num_leaves)
            leaf = self.dc_tree[-1][leaf_id]
            self.data_partition_table[i] = leaf_id
            leaf.node_data.add(i)
            self.data[self.h - 1][i] = X[i]

    def _set_partition_data(self, dataset):
        """
            This function will set the data for each leaf based on dataset (dict) deterministically instead of random partition 
        """
        num_leaves = len(self.dc_tree[-1])
        data_full = dataset["full_data"]
        print("=" * 5, "preparing data for dc-kmeans", "=" * 5)
        for leaf_id in tqdm(range(num_leaves)):
            client_data = dataset["client_" + str(leaf_id)]
            leaf = self.dc_tree[-1][leaf_id]
            for i_client_data in range(client_data.shape[0]):
                i_in_data_full = np.where(
                    (data_full == client_data[i_client_data]).all(
                        axis=1))[0][0]
                # assert i_in_data_full.size == 1, "there are identical data points in dataset"
                # i_in_data_full = i_in_data_full[0]
                self.data_partition_table[i_in_data_full] = leaf_id
                leaf.node_data.add(i_in_data_full)
                self.data[
                    self.h -
                    1][i_in_data_full] = data_full[i_in_data_full].copy()
                data_full[i_in_data_full] = np.ones(
                    (self.d, )) * 2.0  # 2 will never happen

    def _run(self):
        print("=" * 5, "initial training of dc-kmeans", "=" * 5)
        for l in range(self.h - 1, -1, -1):
            c = 0
            for j in tqdm(range(self.widths[l])):
                subproblem = self.dc_tree[l][j]
                if subproblem.parent == None:
                    # root_data = np.vstack([self.data[0][root_data_idx] for root_data_idx in range(self.widths[1]*self.ks[1])])
                    subproblem.fit(self.data[0])
                    self.centroids = subproblem.cluster_centers_
                else:
                    subproblem._run_node(self.data[l])
                    for c_id in range(len(subproblem.centroids)):
                        subproblem.data_prop.add(c)
                        subproblem.parent.node_data.add(c)
                        self.data[l - 1][c] = subproblem.centroids[c_id]
                        c += 1

    def _init_tree(self, ks, widths, iters, n_init):
        # tree = [[DCnode(ks[0], iters)]] # root node
        root_node = KMeans(n_clusters=ks[0], n_init=n_init, max_iter=iters)
        root_node.parent = None
        root_node.children = []
        root_node.node_data = set()
        tree = [[root_node]]  # use sklearn.KMeans for the root node
        for i in range(1, len(widths)):
            k = ks[i]
            assert widths[i] % widths[i -
                                      1] == 0, "Inconsistent widths in tree"
            merge_factor = int(widths[i] / widths[i - 1])
            level = []
            for j in range(widths[i - 1]):
                parent = tree[i - 1][j]
                for _ in range(merge_factor):
                    child = DCnode(k, iters=iters)
                    child.parent = parent
                    parent.children.append(child)
                    level.append(child)
            tree.append(level)
        return tree
