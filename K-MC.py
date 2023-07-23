from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

# Load the data from Kmeans.txt
df = pd.read_csv("Kmeans.txt")
# Visualize the original data
plt.scatter(df['rollno'], df['marks'])
plt.xlabel('Rollno')
plt.ylabel('Marks')
plt.title('Original Data')
plt.show()
# scale the marks and rollno column using MinMaxScaler
scale = MinMaxScaler()
df[['marks', 'rollno']] = scale.fit_transform(df[['marks', 'rollno']])
# Perform K-Means clustering
km = KMeans(n_clusters=3, n_init=10)
predicted = km.fit_predict(df[['rollno', 'marks']])
df['cluster'] = predicted
# Separating data for each clusters
df1 = df[df['cluster'] == 0]
df2 = df[df['cluster'] == 1]
df3 = df[df['cluster'] == 2]
# Plot the data points
plt.scatter(df1['rollno'], df1['marks'], color='green', label='Cluster1')
plt.scatter(df2['rollno'], df2['marks'], color='blue', label='Cluster2')
plt.scatter(df3['rollno'], df3['marks'], color='red', label='Cluster3')
#Plot the cluster center
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='black', marker='*', s=200, label='Clusters Center')

plt.xlabel('rollno')
plt.ylabel('marks')
plt.title('K-Means Clustering')
plt.show()

