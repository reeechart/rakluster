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
        centroid = np.array(data)[idx].tolist()
        print(centroid)
        return centroid

    def euclidean_dst(self,x,y):
        distance = 0
        for i in range(len(x)):
            distance += (x[i] - y[i]) ** 2

        return np.sqrt(distance)

    def get_cluster(self, n_cluster, instances, centroid):
        distance = []
        for i in range(n_cluster):
            distance.append(self.euclidean_dst(instances, centroid[i]))
        
        return distance.index(min(distance))
    
    def find_cluster_member(self,label,cluster):
        indices = []
        for i in range(len(label)):
            if (label[i] == cluster):
                indices.append(i)
        
        return indices
    
    def calculate_new_centroid(self,data,indices):
        sum_point = []
        total = 0
        for i in range(len(data[0])):
            for j in range(len(indices)):
                total += data[indices[j]][i]
            sum_point.append(total)
                
        for i in range(len(sum_point)):
            sum_point[i] /= len(indices)
            
        return sum_point
        
    def fit(self, data):
        print(self.n_clusters)
        self.centroid = self.init_centroid(data,self.n_clusters)
        convergance = False
        iteration = 0

        while not convergance:
            if (iteration > 0) :
                self.centroid = self.next_centroid.copy()
                
            self.next_centroid = []
            self.label = []
            
            print(self.centroid)
            for i in range(len(data)):
                self.label.append(self.get_cluster(self.n_clusters,data[i],self.centroid))
                
            for j in range(self.n_clusters):
                self.next_centroid.append(self.calculate_new_centroid(data,self.find_cluster_member(label,j)))
            
            iteration += 1
            convergance = (self.centroid == self.next_centroid or iteration >=100)
            print(iteration)
            print(self.next_centroid)
            
        return self.label

