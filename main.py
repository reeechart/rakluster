from cluster.kmeans import KMeans
from cluster.agglomerative_clustering import AgglomerativeClustering
from cluster.dbscan import DBSCAN

kmeans = KMeans(3, 100)
print(kmeans.n_clusters)
print(kmeans.max_iter)

agglo = AgglomerativeClustering(n_clusters=3, linkage='complete')
print(agglo.n_clusters)
print(agglo.linkage)

dbscan = DBSCAN(0.5, 4)
print(dbscan.epsilon)
print(dbscan.min_pts)