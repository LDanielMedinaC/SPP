import pandas as pn
from sklearn.cluster import DBSCAN, AgglomerativeClustering, MeanShift, KMeans, estimate_bandwidth
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt     

data = pn.read_csv("data.csv")
data = data[["Happiness Rank", "Year_2016"]]


###############################################
# Code for choosing a k     
#
# wcss = []
# for i in range(1,11):
#     kmeans = KMeans(i)
#     kmeans.fit(data)
#     wcss.append(kmeans.inertia_)

# plt.plot(range(1,11),wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')

# km = KMeans(n_clusters = 2)
# labels = km.fit_predict(data)

# plt.scatter(data["Happiness Rank"], data["Year_2016"], c=labels)#, cmap='plasma')
#   
#    we this code we selected a k = 3 
# 
###############################################


km = KMeans(n_clusters = 3)
labels = km.fit_predict(data)
scaler = MinMaxScaler()
data[["Year_2016"]] = scaler.fit_transform(data[["Year_2016"]])
plt.scatter(data["Happiness Rank"], data["Year_2016"], c=labels)#, cmap='plasma')
plt.show()