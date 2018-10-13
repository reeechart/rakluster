class AgglomerativeClustering:
    SINGLE_LINKAGE = 'single'
    AVERAGE_LINKAGE = 'average'
    AVERAGE_GROUP_LINKAGE = 'average_group'
    COMPLETE_LINKAGE = 'complete'

    def __init__(self, n_clusters, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage