import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroid = []
        self.next_centroid = []
        self.label = []
    
    def init_centroid(self,data,n_clusters):
        idx = np.sort(np.random.choice(len(data), n_clusters, replace=False))
        self.centroid = np.array(data)[idx].tolist()

        return self.centroid

    def euclidean_dst(self,x,y):
        distance = 0
        for i in range(len(x)):
            distance += (x[i] - y[i]) ** 2

        return np.sqrt(distance)

    def get_cluster(self, n_cluster, instances, centroid):
        distance = list(self.euclidean_dst(instances,centroid[i]) for i in range(n_cluster))
        
        return distance.index(min(distance))
    
    def find_cluster_member(self,label,cluster):
        indices = list(i for i, x in enumerate(label) if x == cluster)
        
        return indices
    
    def calculate_new_centroid(self,data,indices):
        sum_point = []
        for i in range(len(data[0])):
            total = 0
            for j in range(len(indices)):
                total += data[indices[j]][i]
            sum_point.append(total)
                
        for i in range(len(sum_point)):
            sum_point[i] /= len(indices)
            
        return sum_point
        
    def fit(self, data):
        self.centroid = self.init_centroid(data,self.n_clusters)
        convergance = False
        iteration = 0

        while not convergance:
            if (iteration > 0) :
                self.centroid = self.next_centroid.copy()
                
            self.next_centroid = []
            self.label = []
            
            self.label = list(self.get_cluster(self.n_clusters,data[i],self.centroid) for i in range(len(data)))    
            self.next_centroid = list(self.calculate_new_centroid(data,self.find_cluster_member(self.label,i)) for i in range(self.n_clusters))
            
            iteration += 1
            convergance = (self.centroid == self.next_centroid or iteration >=100)
            
        return self.label
    
    def predict(self, instances):
        return self.get_cluster(self.n_clusters, instances, self.centroid)