from scipy.spatial import distance
import numpy as np

class AgglomerativeClustering:
    SINGLE_LINKAGE = 'single'
    AVERAGE_LINKAGE = 'average'
    AVERAGE_GROUP_LINKAGE = 'average_group'
    COMPLETE_LINKAGE = 'complete'
    LINKAGES = [SINGLE_LINKAGE, AVERAGE_LINKAGE, AVERAGE_GROUP_LINKAGE, COMPLETE_LINKAGE]

    EUCLIDEAN_DISTANCE = 'euclidean'
    MANHATTAN_DISTANCE = 'manhattan'
    AFFINITY_LIST = [EUCLIDEAN_DISTANCE, MANHATTAN_DISTANCE]

    def __init__(self, n_clusters, linkage=SINGLE_LINKAGE, affinity=EUCLIDEAN_DISTANCE):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.affinity = affinity
        self.check_validity()
        self.clusters = []
        self.clusters_distance = []
        self.labels_ = []

    def check_validity(self):
        if (self.n_clusters < 1):
            raise(AttributeError('n_clusters must be positive number'))
        if (self.linkage not in self.LINKAGES):
            raise(AttributeError('Linkage algorithm %s is not available' % self.linkage))
        if (self.affinity not in self.AFFINITY_LIST):
            raise(AttributeError('Affinity %s is not supported' % self.affinity))

    def get_distance(self, obj_one, obj_two):
        if (self.affinity == self.EUCLIDEAN_DISTANCE):
            return distance.euclidean(obj_one, obj_two)
        elif (self.affinity == self.MANHATTAN_DISTANCE):
            dist = 0
            for i in range(len(obj_one)):
                dist += abs(obj_one[i] - obj_two[i])
            return dist

    def initiate(self, data):
        for index in range(len(data)):
            distances = []
            cluster = [index]
            self.clusters.append(cluster)
            for pair in range(len(data)):
                distances.append(self.get_distance(data[index], data[pair]))
            self.clusters_distance.append(distances)
        self.clusters_distance = np.array(self.clusters_distance)

    def fit(self, data):
        self.initiate(data)
        while(len(self.clusters) > self.n_clusters):
            row_idx, col_idx = self.get_next_clustered_indices()
            self.merge_cluster(data, row_idx, col_idx)
        self.convert_labels(data)

    def get_next_clustered_indices(self):
        minimum_distance = np.infty
        row_index = -1
        col_index = -1
        for i in range(len(self.clusters)-1):
            for j in range(i+1, len(self.clusters)):
                if (self.clusters_distance[i][j] < minimum_distance):
                    minimum_distance = self.clusters_distance[i][j]
                    row_index = i
                    col_index = j
        return row_index, col_index

    def merge_cluster(self, data, row_idx, col_idx):
        self.clusters[row_idx] += self.clusters[col_idx]
        self.clusters.pop(col_idx)
        self.compute_distance(data, row_idx, col_idx)

    def compute_distance(self, data, row_idx, col_idx):
        if (self.linkage == self.SINGLE_LINKAGE):
            self.compute_single_distance(data, row_idx, col_idx)
        elif (self.linkage == self.COMPLETE_LINKAGE):
            self.compute_complete_distance(data, row_idx, col_idx)
        elif (self.linkage == self.AVERAGE_LINKAGE):
            self.compute_average_distance(data)
        elif (self.linkage == self.AVERAGE_GROUP_LINKAGE):
            self.compute_average_group_distance(data)
        
    def compute_single_distance(self, data, row_idx, col_idx):
        row = self.clusters_distance[row_idx, :]
        col = self.clusters_distance[:, col_idx]

        min_val = np.minimum(row, col)
        min_val = np.delete(min_val, col_idx, 0)

        self.clusters_distance = np.delete(self.clusters_distance, col_idx, axis=1)
        self.clusters_distance = np.delete(self.clusters_distance, col_idx, axis=0)
        self.clusters_distance = np.delete(self.clusters_distance, row_idx, axis=1)
        self.clusters_distance = np.delete(self.clusters_distance, row_idx, axis=0)

        min_row = np.delete(min_val, row_idx, axis=0).tolist()
        self.clusters_distance = np.insert(self.clusters_distance, row_idx, min_row, axis=0)
        self.clusters_distance = np.insert(self.clusters_distance, row_idx, min_val, axis=1)

    def compute_complete_distance(self, data, row_idx, col_idx):
        row = self.clusters_distance[row_idx, :]
        col = self.clusters_distance[:, col_idx]

        min_val = np.maximum(row, col)
        min_val = np.delete(min_val, col_idx, 0)

        self.clusters_distance = np.delete(self.clusters_distance, col_idx, axis=1)
        self.clusters_distance = np.delete(self.clusters_distance, col_idx, axis=0)
        self.clusters_distance = np.delete(self.clusters_distance, row_idx, axis=1)
        self.clusters_distance = np.delete(self.clusters_distance, row_idx, axis=0)

        min_row = np.delete(min_val, row_idx, axis=0).tolist()
        self.clusters_distance = np.insert(self.clusters_distance, row_idx, min_row, axis=0)
        self.clusters_distance = np.insert(self.clusters_distance, row_idx, min_val, axis=1)
    
    def compute_average_distance(self, data):
        distance_matrix = []
        for i in range(len(self.clusters)):
            row_distance = []
            for j in range(len(self.clusters)):
                row_distance.append(0)
            distance_matrix.append(row_distance)
        for i in range(len(self.clusters)-1):
            for j in range(i+1, len(self.clusters)):
                sum_distance = 0
                for i_member_idx in self.clusters[i]:
                    for j_member_idx in self.clusters[j]:
                        sum_distance += self.get_distance(data[i_member_idx], data[j_member_idx])
                avg_distance = sum_distance/(len(self.clusters[i]) + len(self.clusters[j]))
                distance_matrix[i][j] = avg_distance
                distance_matrix[j][i] = avg_distance
        self.clusters_distance = np.array(distance_matrix)

    def compute_average_group_distance(self, data):
        distance_matrix = []
        attr_len = len(data[0])
        for i in range(len(self.clusters)):
            row_distance = []
            for j in range(len(self.clusters)):
                row_distance.append(0)
            distance_matrix.append(row_distance)
        for i in range(len(self.clusters)-1):
            for j in range(i+1, len(self.clusters)):
                i_sum_member = []
                j_sum_member = []
                for _ in range(attr_len):
                    i_sum_member.append(0)
                    j_sum_member.append(0)
                for i_member_idx in self.clusters[i]:
                    i_sum_member = [x + y for x,y in zip(i_sum_member, data[i_member_idx])]
                for j_member_idx in self.clusters[j]:
                    j_sum_member = [x + y for x,y in zip(j_sum_member, data[j_member_idx])]
                i_cluster_means = [x/len(self.clusters[i]) for x in i_sum_member]
                j_cluster_means = [x/len(self.clusters[j]) for x in j_sum_member]
                cluster_distance = self.get_distance(i_cluster_means, j_cluster_means)
                distance_matrix[i][j] = cluster_distance
                distance_matrix[j][i] = cluster_distance
        self.clusters_distance = np.array(distance_matrix)

    def convert_labels(self, data):
        for data_idx in range(len(data)):
            for cluster_idx in range(len(self.clusters)):
                if data_idx in self.clusters[cluster_idx]:
                    self.labels_.append(cluster_idx)
        self.labels_ = np.array(self.labels_)
        