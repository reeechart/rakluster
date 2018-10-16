from scipy.spatial import distance
import numpy as np

class AgglomerativeClustering:
    SINGLE_LINKAGE = 'single'
    AVERAGE_LINKAGE = 'average'
    AVERAGE_GROUP_LINKAGE = 'average_group'
    COMPLETE_LINKAGE = 'complete'
    LINKAGES = [SINGLE_LINKAGE, AVERAGE_LINKAGE, AVERAGE_GROUP_LINKAGE, COMPLETE_LINKAGE]

    def __init__(self, n_clusters, linkage=SINGLE_LINKAGE):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.check_validity()
        self.clusters = []
        self.clusters_distance = []
        self.labels_ = []

    def check_validity(self):
        if (self.n_clusters < 1):
            raise(AttributeError('n_clusters must be positive number'))
        if (self.linkage not in self.LINKAGES):
            raise(AttributeError('Linkage algorithm %s is not available' % self.linkage))

    def initiate(self, data):
        for index in range(len(data)):
            distances = []
            cluster = [index]
            self.clusters.append(cluster)
            for pair in range(len(data)):
                distances.append(distance.euclidean(data[index], data[pair]))
            self.clusters_distance.append(distances)
        self.clusters_distance = np.array(self.clusters_distance)

    def fit(self, data):
        self.initiate(data)
        while(len(self.clusters) > self.n_clusters):
            row_idx, col_idx = self.get_next_clustered_indices()
            print(row_idx, col_idx)
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
        print(len(self.clusters_distance))

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
        print(len(self.clusters_distance))
    
    def compute_average_distance(self, data):
        print('y')

    def compute_average_group_distance(self, data):
        print('z')

    def convert_labels(self, data):
        for data_idx in range(len(data)):
            for cluster_idx in range(len(self.clusters)):
                if data_idx in self.clusters[cluster_idx]:
                    self.labels_.append(cluster_idx)