from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import pandas as pn
import numpy as np
from sklearn.preprocessing import MinMaxScaler



data = pn.read_csv("data.csv")
data = data[["Happiness Rank", "Year_2016"]]

############################################
#
# Getting eplsion
#
# neigh = NearestNeighbors(n_neighbors=2)
# nbrs = neigh.fit(data)
# distances, indices = nbrs.kneighbors(data)
# distances = np.sort(distances, axis=0)
# distances = distances[:,1]
# plt.plot(distances)
# plt.show()
#
#We determine that eplsilon is 5000000
############################################

dbs = DBSCAN(eps=500000, min_samples=3)
labels = dbs.fit_predict(data)

plt.scatter(data["Happiness Rank"], data["Year_2016"], c=labels)#, cmap='plasma')
plt.show()