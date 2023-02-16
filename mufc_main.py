#! /usr/bin/env python
# -*- coding: utf-8 -*-

#############################
# Main running file.
#############################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from model import *
from utils import *
import argparse
import time
from tqdm import tqdm
import copy
import os
import os.path as osp
from kfed import kfed
from dc_kmeans import DCKmeans

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="input data path")
    parser.add_argument("--num_clusters",
                        type=int,
                        default=10,
                        help="K in K-means")
    parser.add_argument("--num_clients",
                        type=int,
                        default=100,
                        help="number of clients/devices")
    parser.add_argument("--num_removes",
                        type=int,
                        default=10,
                        help="number of sequential removal requests")
    parser.add_argument("--client_rounds",
                        type=int,
                        default=1,
                        help="only used for debug")
    parser.add_argument("--client_oversample",
                        type=int,
                        default=1,
                        help="oversampling coefficient on client")
    parser.add_argument("--k_prime",
                        type=int,
                        default=10,
                        help="real number of clusters on devices")
    parser.add_argument("--num_trials",
                        type=int,
                        default=1,
                        help="number of repeated trails")
    parser.add_argument("--max_iters",
                        type=int,
                        default=300,
                        help="max round of Lloyd iterations")
    parser.add_argument("--split", type=str, default='iid', help="iid/non-iid")
    parser.add_argument("--result_path", type=str, default="results")
    parser.add_argument("--compare_kfed",
                        action='store_true',
                        default=False,
                        help="whether to compare with kfed")
    parser.add_argument('--compare_dc',
                        action='store_true',
                        default=False,
                        help="whether to compare with dc-kmeans")
    parser.add_argument(
        '--client_kpp_only',
        action='store_true',
        default=False,
        help=
        "whether to only perform initialization on clients, only used for debug"
    )
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help="verbosity of loss")
    parser.add_argument(
        '--update_centralized_loss',
        action='store_true',
        default=False,
        help="whether to update optimal centralized loss or not")
    args = parser.parse_args()

    assert args.client_rounds == 1, "Do not perform multirounds on clients!"
    # n_init for sklearn.KMeans
    if "femnist" in args.data_path:
        n_init = 3
    else:
        n_init = 10

    if not osp.exists(args.result_path):
        os.makedirs(args.result_path)

    dataset = load_dataset(args.data_path)  # dataset is a json file
    print("Processing dataset:", args.data_path, " kp =", args.k_prime)
    dataset_name = args.data_path.split("/")[-1]  # with .pkl suffix

    print("Use fixed eps")
    # fix quant_eps = 1 / sqrt(data_combined.shape[0]) for now
    quant_eps = 1 / np.sqrt(dataset["full_data"].shape[0])
    print("eps =", quant_eps, "n =", dataset["full_data"].shape[0])

    # mufc
    global_induced_loss = np.zeros((args.num_trials, args.num_removes + 1))
    local_induced_loss = np.zeros((args.num_trials, args.num_removes + 1))
    our_removal_time = np.zeros((args.num_trials, args.num_removes + 1))
    full_retrain_time = np.zeros((args.num_trials, ))
    num_retrain = np.zeros((args.num_trials, ))
    centralized_loss = np.ones(
        (args.num_trials,
         args.num_removes + 1))  # optimal loss after each removal

    # kfed, which does not support unlearning
    kfed_induced_loss = np.zeros((args.num_trials, ))
    kfed_time = np.zeros((args.num_trials, ))

    # dc-kmeans, where each leaf node is a client
    dc_induced_loss = np.zeros((args.num_trials, args.num_removes + 1))
    dc_removal_time = np.zeros((args.num_trials, args.num_removes + 1))

    for i_trail in range(args.num_trials):
        print("=" * 30, i_trail, "=" * 30)
        ###################
        # mufc training
        ###################
        print("=" * 5, "mufc training", "=" * 5)
        centralized_loss[i_trail, 0] = dataset[
            "kmeans_loss"]  # this is the pre-computed optimal loss
        client_worst_time_real = 0
        client_worst_time_baseline = 0
        kmeans_clients = []
        data_server = []
        for i_client in tqdm(range(args.num_clients)):
            # real client computation
            start = time.time()
            if args.split == "iid":
                client_kmeans = MyKmeans(k=int(args.client_oversample *
                                               args.num_clusters),
                                         max_iters=args.max_iters)
            else:
                client_kmeans = MyKmeans(k=int(args.client_oversample *
                                               args.k_prime),
                                         max_iters=args.max_iters)
            # do multirounds on clients or not
            if args.client_rounds == 1:
                if args.client_kpp_only:
                    _, client_assignments, _ = client_kmeans.run_kpp_only(
                        dataset["client_" + str(i_client)])
                else:
                    _, client_assignments, _ = client_kmeans.run(
                        dataset["client_" + str(i_client)])
            else:
                raise NotImplementedError
            # quantize client centroids
            client_quant_centroids = client_kmeans.quantize_centroids(
                quant_eps)

            client_time_used = time.time() - start
            if client_time_used > client_worst_time_real:
                client_worst_time_real = client_time_used
            # generate random samples within each quantization bin
            for i_client_n in range(client_kmeans.n):
                rnd_samples = sample_points_in_bin(
                    client_quant_centroids[client_assignments[i_client_n]], 1,
                    quant_eps)
                data_server.append(rnd_samples)
            # store client information
            client_dict = {}
            client_dict["model"] = client_kmeans
            kmeans_clients.append(client_dict)

            # baseline comparison, always do full K-means++ on clients
            start = time.time()
            if args.split == "iid":
                baseline_kmeans = MyKmeans(k=int(args.client_oversample *
                                                 args.num_clusters),
                                           max_iters=args.max_iters)
            else:
                baseline_kmeans = MyKmeans(k=int(args.client_oversample *
                                                 args.k_prime),
                                           max_iters=args.max_iters)
            baseline_kmeans.run(dataset["client_" + str(i_client)])
            baseline_kmeans.quantize_centroids(quant_eps)
            baseline_time_used = time.time() - start
            if baseline_time_used > client_worst_time_baseline:
                client_worst_time_baseline = baseline_time_used

        # server side computation
        start = time.time()
        # concatenate server data
        data_server = np.concatenate(data_server, axis=0)
        assert data_server.shape[0] == dataset["full_data"].shape[0]

        # run K-means++ on server
        # we use sklearn.KMeans on server side for improved efficiency
        # kmeans_server = MyKmeans(k=args.num_clusters, max_iters=args.max_iters)
        # server_centroids, server_assignments, _ = kmeans_server.run(data_server)

        kmeans_server = KMeans(n_clusters=args.num_clusters,
                               max_iter=args.max_iters,
                               n_init=n_init).fit(data_server)
        server_centroids, server_assignments = kmeans_server.cluster_centers_, kmeans_server.labels_

        # Since client can compute in parallel, we only need to add the client that takes the longest time
        our_removal_time[i_trail,
                         0] = time.time() - start + client_worst_time_real
        full_retrain_time[i_trail] = time.time(
        ) - start + client_worst_time_baseline
        # our_removal_time[i_trail, 0] = client_worst_time
        ###########
        # compute loss
        tmp_count = 0
        for i_client in range(args.num_clients):
            # compute global induced loss
            global_induced_loss[i_trail, 0] += clustering_loss(
                dataset["client_" + str(i_client)], server_centroids)
            local_induced_loss[i_trail, 0] += induced_loss(
                dataset["client_" + str(i_client)], server_centroids,
                server_assignments[tmp_count:tmp_count +
                                   dataset["client_" +
                                           str(i_client)].shape[0]])
            tmp_count += dataset["client_" + str(i_client)].shape[0]

        print("=" * 5, "loss before removal", "=" * 5)
        print(
            f"mufc optimal loss: {centralized_loss[i_trail, 0]}; "
            f"global induced: {global_induced_loss[i_trail, 0]}, ratio: {global_induced_loss[i_trail, 0] / centralized_loss[i_trail, 0]}; "
            f"local induced: {local_induced_loss[i_trail, 0]}, ratio: {local_induced_loss[i_trail, 0] / centralized_loss[i_trail, 0]}"
        )
        print(
            f"mufc time used: {our_removal_time[i_trail, 0]} s, full training: {full_retrain_time[i_trail]} s"
        )

        ###################
        # kfed training
        ###################
        if args.compare_kfed:
            # prepare data for kfed
            kfed_data = []
            for i_client in range(args.num_clients):
                kfed_data.append(dataset["client_" + str(i_client)])
            start = time.time()
            _, _, induced_kfed_loss = kfed(kfed_data,
                                           args.k_prime,
                                           args.num_clusters,
                                           max_iters=args.max_iters)
            kfed_time[i_trail] = time.time() - start
            kfed_induced_loss[i_trail] = induced_kfed_loss

            print(
                f"kfed local induced loss: {kfed_induced_loss[i_trail]}, ratio: {kfed_induced_loss[i_trail] / centralized_loss[i_trail, 0]}"
            )
            print(f"kfed time used: {kfed_time[i_trail]} s")

        ###################
        # DC-kmeans training
        ###################
        if args.compare_dc:
            start = time.time()
            if args.split == "iid":
                dckmeans = DCKmeans([args.num_clusters, args.num_clusters],
                                    [1, args.num_clients],
                                    iters=args.max_iters,
                                    n_init=n_init)
            else:
                dckmeans = DCKmeans([args.num_clusters, args.k_prime],
                                    [1, args.num_clients],
                                    iters=args.max_iters,
                                    n_init=n_init)
            dc_dataset = copy.deepcopy(dataset)
            _, _, dc_induced_loss[i_trail, 0] = dckmeans.run(dc_dataset)
            dc_removal_time[i_trail, 0] = time.time() - start

            print(
                f"dc-kmeans local induced loss: {dc_induced_loss[i_trail, 0]}, ratio: {dc_induced_loss[i_trail, 0] / centralized_loss[i_trail, 0]}"
            )
            print(f"dc-kmeans time used: {dc_removal_time[i_trail, 0]} s")

        ###################
        # unlearning
        ###################
        # set args.num_removes = 0 with disable removal
        print("=" * 5, "unlearning begins", "=" * 5)
        # need to copy the original dataset
        data_combined = copy.deepcopy(dataset["full_data"])
        remove_queue = []

        for i_remove in tqdm(range(args.num_removes)):
            # generate a sequence of points that need to be removed
            # need to ensure that client size always >= model.k
            client_idx_remove = np.random.randint(args.num_clients)
            while kmeans_clients[client_idx_remove][
                    "model"].n < kmeans_clients[client_idx_remove]["model"].k:
                client_idx_remove = np.random.randint(args.num_clients)
            client_data_idx_remove = np.random.randint(
                kmeans_clients[client_idx_remove]["model"].n)
            remove_queue.append([client_idx_remove, client_data_idx_remove])

            ###################
            # mufc unlearning
            ###################

            # perform the unlearning
            client_kmeans = kmeans_clients[client_idx_remove]["model"]
            ori_data_server_size = data_server.shape[0]
            # record the data point that needs to be removed
            data_to_remove = client_kmeans.data[client_data_idx_remove]
            start = time.time()
            retrain_flag = client_kmeans.unlearn_check(client_data_idx_remove)
            # record the removal time
            # our_removal_time[i_trail, i_remove+1] = time.time() - start
            if retrain_flag or not args.client_kpp_only:
                # if true, we need to retrain the client model
                num_retrain[i_trail] += 1
                tmp_data = np.delete(client_kmeans.data,
                                     client_data_idx_remove,
                                     axis=0)
                # start = time.time()
                if args.split == "iid":
                    client_kmeans = MyKmeans(k=int(args.client_oversample *
                                                   args.num_clusters),
                                             max_iters=args.max_iters)
                else:
                    client_kmeans = MyKmeans(k=int(args.client_oversample *
                                                   args.k_prime),
                                             max_iters=args.max_iters)
                if args.client_kpp_only:
                    _, client_assignments, _ = client_kmeans.run_kpp_only(
                        tmp_data)
                else:
                    _, client_assignments, _ = client_kmeans.run(tmp_data)
                # quantize the centroids
                client_quant_centroids = client_kmeans.quantize_centroids(
                    quant_eps)
                # delete the whole client data from server side
                data_server_remove_start = 0
                for before_client_idx_remove in range(client_idx_remove):
                    data_server_remove_start += kmeans_clients[
                        before_client_idx_remove]["model"].n
                data_server_list = []
                if client_idx_remove != 0:
                    data_server_list += [
                        data_server[0:data_server_remove_start, :]
                    ]
                # generate random samples within each quantization bin
                for i_client_n in range(client_kmeans.n):
                    rnd_samples = sample_points_in_bin(
                        client_quant_centroids[client_assignments[i_client_n]],
                        1, quant_eps)
                    data_server_list.append(rnd_samples)
                if client_idx_remove != args.num_clients - 1:
                    data_server_list += [
                        data_server[data_server_remove_start +
                                    tmp_data.shape[0] + 1:, :]
                    ]
                data_server = np.concatenate(data_server_list, axis=0)
                # train the new server model
                kmeans_server = KMeans(n_clusters=args.num_clusters,
                                       max_iter=args.max_iters,
                                       n_init=n_init).fit(data_server)
                server_centroids, server_assignments = kmeans_server.cluster_centers_, kmeans_server.labels_
            else:
                # we do not need to retrain client model
                data_server_remove_start = 0
                for before_client_idx_remove in range(client_idx_remove):
                    data_server_remove_start += kmeans_clients[
                        before_client_idx_remove]["model"].n
                data_server = np.delete(data_server,
                                        data_server_remove_start +
                                        client_data_idx_remove,
                                        axis=0)
                # lazy update
                server_assignments = np.delete(
                    server_assignments,
                    data_server_remove_start + client_data_idx_remove)
            # record the removal time
            our_removal_time[i_trail, i_remove + 1] = time.time() - start
            # update the client model list
            kmeans_clients[client_idx_remove]["model"] = client_kmeans
            # check the size of data_server
            assert ori_data_server_size - data_server.shape[
                0] == 1, "data_server is not correctly updated"

            ###########
            # compute loss
            tmp_count = 0
            for i_client in range(args.num_clients):
                # compute global induced loss
                global_induced_loss[i_trail, i_remove + 1] += clustering_loss(
                    kmeans_clients[i_client]["model"].data, server_centroids)
                local_induced_loss[i_trail, i_remove + 1] += induced_loss(
                    kmeans_clients[i_client]["model"].data, server_centroids,
                    server_assignments[tmp_count:tmp_count +
                                       kmeans_clients[i_client]["model"].data.
                                       shape[0]])
                tmp_count += kmeans_clients[i_client]["model"].data.shape[0]

            global_remove_idx = np.where(
                (data_combined == data_to_remove).all(axis=1))[0][0]
            data_combined = np.delete(data_combined, global_remove_idx, axis=0)
            if args.update_centralized_loss:
                # optimal loss for updated data_combined
                clf = KMeans(n_clusters=args.num_clusters,
                             max_iter=args.max_iters,
                             n_init=n_init).fit(data_combined)
                centralized_loss[i_trail, i_remove + 1] = clf.inertia_

            if args.verbose:
                print(
                    f"{i_remove+1}, number of retrain: {num_retrain[i_trail]}, optimal loss: {centralized_loss[i_trail, i_remove+1]}; "
                    f"global induced: {global_induced_loss[i_trail, i_remove+1]}, ratio: {global_induced_loss[i_trail, i_remove+1] / centralized_loss[i_trail, i_remove+1]}; "
                    f"local induced: {local_induced_loss[i_trail, i_remove+1]}, ratio: {local_induced_loss[i_trail, i_remove+1] / centralized_loss[i_trail, i_remove+1]}"
                )
                print(
                    f"{i_remove+1}, time used: {our_removal_time[i_trail, i_remove+1]} s, ratio: {full_retrain_time[i_trail] / our_removal_time[i_trail, i_remove+1]}"
                )

            ###################
            # dc-kmeans unlearning
            ###################
            if args.compare_dc:
                # global_remove_idx is also the one that we need to remove from dc-kmeans
                _, _, dc_induced_loss[i_trail, i_remove + 1], dc_removal_time[
                    i_trail, i_remove + 1] = dckmeans.delete(global_remove_idx)
                if args.verbose:
                    print(
                        f"dc-kmeans local induced loss: {dc_induced_loss[i_trail, i_remove+1]}, ratio: {dc_induced_loss[i_trail, i_remove+1] / centralized_loss[i_trail, i_remove+1]}"
                    )
                    print(
                        f"dc-kmeans time used: {dc_removal_time[i_trail, i_remove+1]} s, ratio: {full_retrain_time[i_trail] / dc_removal_time[i_trail, i_remove+1]}"
                    )

    # store all the results
    final_res = {}
    final_res["global_induced_loss"] = global_induced_loss
    final_res["local_induced_loss"] = local_induced_loss
    final_res["our_removal_time"] = our_removal_time
    final_res["full_retrain_time"] = full_retrain_time
    final_res["num_retrain"] = num_retrain
    final_res["centralized_loss"] = centralized_loss

    final_res["kfed_induced_loss"] = kfed_induced_loss
    final_res["kfed_time"] = kfed_time

    final_res["dc_induced_loss"] = dc_induced_loss
    final_res["dc_removal_time"] = dc_removal_time

    with open(os.path.join(args.result_path, "full_" + dataset_name),
              "wb") as f:
        pickle.dump(final_res, f)
