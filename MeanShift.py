import pandas as pn
from sklearn.cluster import DBSCAN, AgglomerativeClustering, MeanShift, KMeans, estimate_bandwidth
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt     

data = pn.read_csv("data.csv")
data = data[["Happiness Rank", "Year_2016"]]

bandwidth = estimate_bandwidth(data)
print(bandwidth)
ms = MeanShift(bandwidth=bandwidth)
labels = ms.fit_predict(data)
scaler = MinMaxScaler()
data[["Year_2016"]] = scaler.fit_transform(data[["Year_2016"]])
plt.scatter(data["Happiness Rank"], data["Year_2016"], c=labels)#, cmap='plasma')
plt.show()