from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
import numpy
class DBSCAN:
    def __init__(self, eps= 0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        if(metric == 'euclidean'):
            self.metric_func = euclidean_distances
        elif(metric == 'manhattan'):
            self.metric_func = manhattan_distances
        elif(callable(metric)):
            self.metric_func = metric
        else:
            self.metric_func = euclidean_distances
        self.metric = metric
    
    def expand(self, index, root):
        for i in self.core_list[root]:
            if(self.result[i] == -1):
                self.result[i] = index
                if(i in self.core_list):
                    self.expand(index, i)

    def fit(self, X):
        print("Start Fitting")
        self.result = {}
        for i in range(len(X)):
            self.result[i] = -1
        distances_matrix = self.metric_func(X)
        #Finding Core
        core_list = {}
        for i in range(len(distances_matrix)):
            incluster = []
            for j in range(len(distances_matrix[i])):
                if(distances_matrix[i][j] <= self.eps):
                    incluster.append(j)
            if(len(incluster) >= self.min_samples):
                core_list[i] = incluster
        self.core_list = core_list
        #Expand Cluster
        idx = 0
        for i in core_list:
            if(self.result[i] == -1):
                self.expand(idx,i)
                idx = idx + 1
        self.labels_ = numpy.array(list(self.result.values()))
        return self
