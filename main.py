from cluster.kmeans import KMeans
from cluster.kmedoids import KMedoids
from cluster.agglomerative_clustering import AgglomerativeClustering
from cluster.dbscan import DBSCAN
from cluster.metrics import purity
from sklearn import datasets

iris = datasets.load_iris()

kmeans = KMeans(n_clusters=3, max_iter=100)
kmeans.fit(iris.data)
print(kmeans.labels_)
print(purity(kmeans.labels_, iris.target))

kmedoids = KMedoids(n_clusters=3, max_iter=100)
kmedoids.fit(iris.data)
print(kmedoids.labels_)
print(purity(kmedoids.labels_, iris.target))

single_agglo = AgglomerativeClustering(n_clusters=3, linkage=AgglomerativeClustering.SINGLE_LINKAGE, affinity=AgglomerativeClustering.EUCLIDEAN_DISTANCE)
single_agglo.fit(iris.data)
print(single_agglo.labels_)
print(purity(single_agglo.labels_, iris.target))

complete_agglo = AgglomerativeClustering(n_clusters=3, linkage=AgglomerativeClustering.COMPLETE_LINKAGE, affinity=AgglomerativeClustering.EUCLIDEAN_DISTANCE)
complete_agglo.fit(iris.data)
print(complete_agglo.labels_)
print(purity(complete_agglo.labels_, iris.target))

average_agglo = AgglomerativeClustering(n_clusters=3, linkage=AgglomerativeClustering.AVERAGE_LINKAGE, affinity=AgglomerativeClustering.EUCLIDEAN_DISTANCE)
average_agglo.fit(iris.data)
print(average_agglo.labels_)
print(purity(average_agglo.labels_, iris.target))

average_group_agglo = AgglomerativeClustering(n_clusters=3, linkage=AgglomerativeClustering.AVERAGE_GROUP_LINKAGE, affinity=AgglomerativeClustering.EUCLIDEAN_DISTANCE)
average_group_agglo.fit(iris.data)
print(average_group_agglo.labels_)
print(purity(average_group_agglo.labels_, iris.target))

dbscan = DBSCAN(eps=0.5, min_samples=4)
dbscan.fit(iris.data)
print(dbscan.labels_)
print(purity(dbscan.labels_, iris.target))