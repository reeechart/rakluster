import numpy as np

class KMedoids:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.idx_next_centroid = []
        self.idx_centroid = []
        self.centroid = []
        self.label = []
        self.cost = 0
        self.next_cost = 0
    
#     def init_centroid(self,data,n_clusters):
#         idx = np.sort(np.random.choice(len(data), n_clusters, replace=False))
#         self.centroid = np.array(data)[idx].tolist()
#         print(self.centroid)
#         return self.centroid

    def init_centroid(self,data,n_clusters):
        idx = np.sort(np.random.choice(len(data), n_clusters, replace=False))
        return idx
    
    def manhattan_dst(self,x,y):
        distance = 0
        for i in range(len(x)):
            distance += abs(x[i]-y[i])
        
        return distance
    
    def get_cluster(self, n_cluster, instances, centroid, data):
        distance = list(self.manhattan_dst(instances,data[centroid[i]]) for i in range(n_cluster))
        
        return distance.index(min(distance))
    
    def find_cluster_member(self,label,cluster):
        indices = list(i for i, x in enumerate(label) if x == cluster)
        
        return indices
    
    def new_medoids(self,label,curr_medoid,n_clusters):
        new_medoid = curr_medoid
        change_med = np.random.choice(n_clusters,1,replace=False)
        ran_med = np.random.choice(self.find_cluster_member(label,change_med),1,replace=False)
        
        while (new_medoid[change_med[0]] == ran_med[0]):
            ran_med = np.random.choice(self.find_cluster_member(label,change_med),1,replace=False)
            
        new_medoid[change_med[0]] = ran_med[0]

        return new_medoid
    
    def count_cost(self,label,data,centroid):
        cost = 0
        
        for i in range(len(data)):
            cost += self.manhattan_dst(data[i],data[centroid[label[i]]])
            
        return cost
    
    def fit(self,data):
        self.idx_centroid = self.init_centroid(data,self.n_clusters)
        self.label = list(self.get_cluster(self.n_clusters,data[i],self.idx_centroid,data) for i in range(len(data)))
        self.cost = self.count_cost(self.label,data,self.idx_centroid)
        self.next_cost = self.cost
        convergance = False
        iteration = 0
        
        while not convergance:
            if (self.cost != self.next_cost):
                self.cost = self.next_cost
                self.idx_centroid = self.idx_next_centroid
                
            self.idx_next_centroid = self.new_medoids(label,self.idx_centroid,self.n_clusters)
            self.label = list(self.get_cluster(self.n_clusters,data[i],self.idx_next_centroid,data) for i in range(len(data)))
            self.next_cost = self.count_cost(self.label,data,self.idx_next_centroid)
            
            iteration += 1
            convergance = (self.cost <= self.next_cost or iteration >= self.max_iter)
        
        self.label = list(self.get_cluster(self.n_clusters,data[i],self.idx_next_centroid,data) for i in range(len(data)))
        self.centroid = list(data[self.idx_next_centroid[i]] for i in range(self.n_clusters))
        
        return self.label
    
    def predict(self,instances):
        distance = list(self.manhattan_dst(instances,self.centroid[i]) for i in range(self.n_clusters))
        
        return distance.index(min(distance))