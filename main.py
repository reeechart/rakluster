from cluster.kmeans import KMeans
from cluster.agglomerative_clustering import AgglomerativeClustering
from cluster.dbscan import DBSCAN

from sklearn import datasets

iris = datasets.load_iris()

kmeans = KMeans(3, 100)
print(kmeans.n_clusters)
print(kmeans.max_iter)

agglo = AgglomerativeClustering(n_clusters=3, linkage=AgglomerativeClustering.SINGLE_LINKAGE)
print(agglo.n_clusters)
print(agglo.linkage)
agglo.fit(iris.data)

dbscan = DBSCAN(0.5, 4)
print(dbscan.epsilon)
print(dbscan.min_pts)