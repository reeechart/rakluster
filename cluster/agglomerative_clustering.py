class AgglomerativeClustering:
    SINGLE_LINKAGE = 'single'
    AVERAGE_LINKAGE = 'average'
    AVERAGE_GROUP_LINKAGE = 'average_group'
    COMPLETE_LINKAGE = 'complete'
    LINKAGES = [SINGLE_LINKAGE, AVERAGE_LINKAGE, AVERAGE_GROUP_LINKAGE, COMPLETE_LINKAGE]

    clusters = []

    def __init__(self, n_clusters, linkage=SINGLE_LINKAGE):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.check_validity()

    def check_validity(self):
        if (self.n_clusters < 1):
            raise(AttributeError('n_clusters must be positive number'))
        if (self.linkage not in self.LINKAGES):
            raise(AttributeError('Linkage algorithm %s is not available' % self.linkage))

    def initiate(self, data):
        for index in range(len(data)):
            cluster = [index]
            self.clusters.append(cluster)

    def fit(self, data):
        self.initiate(data)
        if (self.linkage == self.SINGLE_LINKAGE):
            self.single_fit(data)
        elif (self.linkage == self.COMPLETE_LINKAGE):
            self.complete_fit(data)
        elif (self.linkage == self.AVERAGE_LINKAGE):
            self.average_fit(data)
        elif (self.linkage == self.AVERAGE_GROUP_LINKAGE):
            self.average_group_fit(data)
    
    def single_fit(self, data):
        print('a')

    def complete_fit(self, data):
        print('b')

    def average_fit(self, data):
        print('c')
    
    def average_group_fit(self, data):
        print('d')