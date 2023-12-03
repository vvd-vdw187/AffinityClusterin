from copy import deepcopy
import math
import random
import numpy as np
from sklearn.neighbors import KDTree
from pyspark import SparkConf, SparkContext

class AffinityClustering:
    def __init__(self) -> None:
        pass
   
    def _map_contract_graph(self, _lambda, leader):
        def contraction(adj):
            u, nu = adj
            c, v = u, u
            S = []
            while v not in S:
                S.append(v)
                c = min(c, v)
                v = _lambda[v]
            c = min(c, v)
            A = list(filter(lambda e: leader[e[0]] != c, nu))
            return c, A
        return contraction


    def _reduce_contract_graph(self, leader):
        def reduce_contraction(Nu, A):
            for v, w in A:
                l = leader[v]
                new = True
                for i, e in enumerate(Nu):
                    if l == e[0]:
                        new = False
                        Nu[i] = (l, min(w, e[1]))
                if new:
                    Nu.append((l, w))
            return Nu
        return reduce_contraction


    def _find_best_neighbours(self, adj):
        u, nu = adj
        nn = u
        if len(nu) > 0:
            min_v, min_w = nu[0]
            for v, w in nu:
                if w < min_w:
                    min_v, min_w = v, w
            nn = min_v
        return u, nn


    def _find_leader(self, _lambda):
        def find(adj):
            u, nu = adj
            c, v = u, u
            S = []
            cnt = 0
            while v not in S:
                S.append(v)
                v = _lambda[v]
                cnt += 1
                c = min(c, v)
            return u, c

        return find
    
    def create_distance_matrix(self, dataset, full_dm=False):
        """
        Creates the distance matrix for a dataset with only vertices. Also adds the edges to a dict.
        :param dataset: dataset without edges
        :return: distance matrix, a dict of all edges and the total number of edges
        """
        vertices = []
        size = 0
        three_d = False
        for line in dataset:
            if len(line) == 2:
                vertices.append([line[0], line[1]])
            elif len(line) == 3:
                vertices.append([line[0], line[1], line[2]])
                three_d = True
        if three_d:
            max_weight = 0
            dict = {}
            for i in range(len(dataset)):
                dict2 = {}
                for j in range(i + 1, len(dataset)):
                    value = np.sqrt(np.sum(np.square(dataset[i] - dataset[j])))
                    max_weight = max(value, max_weight)
                    dict2[j] = value
                    size += 1
                dict[i] = dict2
        else:
            d_matrix = scipy.spatial.distance_matrix(vertices, vertices, threshold=1000000)
            dict = {}
            max_weight = 0
            # Run with less edges
            for i in range(len(d_matrix)):
                dict2 = {}
                if full_dm:
                    for j in range(len(d_matrix)):
                        if i != j:
                            size += 1
                            max_weight = max(d_matrix[i][j], max_weight)
                            dict2[j] = d_matrix[i][j]
                    dict[i] = dict2
                else:
                    for j in range(i, len(d_matrix)):
                        if i != j:
                            size += 1
                            max_weight = max(d_matrix[i][j], max_weight)
                            dict2[j] = d_matrix[i][j]
                    dict[i] = dict2
        return dict, size, vertices, max_weight
    
    def affinity_clustering(self, adj, vertex_coordinates, plot_intermediate, num_clusters=3):
        conf = SparkConf().setAppName('MST_Algorithm')
        sc = SparkContext.getOrCreate(conf=conf)
        
        clusters = [[i] for i in range(len(adj))]
        yhats = []
        leaders = []
        graph = deepcopy(adj)
        rdd = sc.parallelize(adj)

        i = 0
        imax = 40
        contracted_leader = [None] * len(adj)
        mst = [None] * len(adj)
        while i < imax:
            if len(graph) <= num_clusters:
                break
            num_edges = sum(map(lambda v: len(v[1]), graph))
            if num_edges == 0:
                break

            rdd1 = rdd.map(self._find_best_neighbours).collect()
            _lambda = [None] * len(adj)
            for line in rdd1:
                _lambda[line[0]] = line[1]

            # Find leader
            leader = [None] * len(adj)
            rdd1 = rdd.map(self._find_leader(_lambda)).collect()
            for line in rdd1:
                leader[line[0]] = line[1]
            leaders.append(leader)

            for j in range(len(adj)):
                l = leader[j]
                if l is not None and not l == j:
                    clusters[l].extend(clusters[j])
                    clusters[j].clear()

            yhat = [None] * len(adj)
            for c, cluster in enumerate(clusters):
                for v in cluster:
                    yhat[v] = c
            yhats.append(yhat)

            for j in range(len(adj)):
                if contracted_leader[j] is None:
                    if yhat[j] != j:
                        contracted_leader[j] = yhat[j]
                        mst[j] = _lambda[j]

            # Contraction
            rdd = (rdd.map(self._map_contract_graph(_lambda=_lambda, leader=leader))
                .foldByKey([], self._reduce_contract_graph(leader)))
            # rdd = rdd.map(map_contract_graph(_lambda=_lambda, leader=leader)).reduceByKey(reduce_contract_graph(leader))

            graph = rdd.collect()

            i += 1

        for j in range(len(adj)):
            if contracted_leader[j] is None:
                contracted_leader[j] = yhat[j]
                mst[j] = yhat[j]

        return i, graph, yhats, contracted_leader, mst
    
            
    def get_nearest_neighbours(self, V, k=5, leaf_size=2, buckets=False):
        def get_sort_key(item):
            return item[1]

        V_copy = deepcopy(V)
        if buckets:
            adj = []
            for key in V:
                nu = []
                sorted_list = sorted(V_copy[key].items(), key=get_sort_key)
                last = -1
                to_shuffle = []
                for i in range(k):
                    if last != sorted_list[i][1]:
                        to_shuffle.append((sorted_list[i][0], sorted_list[i][1]))
                        random.shuffle(to_shuffle)
                        for item in to_shuffle:
                            nu.append(item)
                        to_shuffle = []
                    else:
                        to_shuffle.append((sorted_list[i][0], sorted_list[i][1]))
                    last = sorted_list[i][1]

                random.shuffle(to_shuffle)
                for item in to_shuffle:
                    nu.append(item)
                adj.append((key, nu))
        else:
            kd_tree = KDTree(V, leaf_size=leaf_size)
            dist, ind = kd_tree.query(V, k=k + 1)

            adj = []
            for i in range(len(V)):
                nu = [(ind[i, j], dist[i, j]) for j in range(1, len(dist[i]))]
                adj.append((i, nu))

        return adj

    # num buckets = log_(1 + beta) (W)
    def _create_buckets(self, E, alpha, beta, W):
        num_buckets = math.ceil(math.log(W, (1 + beta)))
        buckets = []
        prev_end = 0
        for i in range(num_buckets):
            now_end = np.power((1 + beta), i) + (np.random.uniform(-alpha, alpha) * np.power((1 + beta), i))
            if i < num_buckets - 1:
                buckets.append((prev_end, now_end))
                prev_end = now_end
            else:
                buckets.append((prev_end, W + 0.00001))

        bucket_counter = [0] * len(buckets)

        for key in E:
            for edge in E[key]:
                bucket_number = 1
                for bucket in buckets:
                    if bucket[0] <= E[key][edge] < bucket[1]:
                        E[key][edge] = bucket_number
                        bucket_counter[bucket_number - 1] += 1
                        break
                    bucket_number += 1
        return E, buckets, bucket_counter


    def shift_edge_weights(self, E, gamma=0.05):
        max_weight = 0
        for key in E:
            for edge in E[key]:
                # TODO: fix shift (remove 100 *)
                if key < edge:
                    E[key][edge] = 100 * max(E[key][edge] + (np.random.uniform(-gamma, gamma)) * E[key][edge], 0)
                    max_weight = max(E[key][edge], max_weight)
                else:
                    E[key][edge] = E[edge][key]
        return E, max_weight


    def find_differences(self, contracted_leader_list):
        diff_matrix = []
        for cl in contracted_leader_list:
            diff = []
            for cl2 in contracted_leader_list:
                diff_count = 0
                for i in range(len(cl2)):
                    if cl[i] != cl2[i]:
                        diff_count += 1
                diff.append(diff_count)
            diff_matrix.append(diff)
        return diff_matrix
    

