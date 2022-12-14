import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import csv

train_data = pd.read_csv("train.csv").drop("song_id", axis=1)
test_data = pd.read_csv("test_3000.csv")

processed_train_data = train_data
processed_train_data.drop(columns=['duration_ms', 'popularity'])

#processed_train_data = processed_train_data.filter(items=["Feature 1"], axis=1)

# for column in processed_train_data.columns:
#     print("Column: ", column)
#     print("Unique values: ", processed_train_data[column].unique())
#     print()



# dummy = pd.get_dummies(processed_train_data["Feature 13"],  prefix="Feature 13", drop_first=True)
# processed_train_data = pd.concat([processed_train_data, dummy], axis=1)
# processed_train_data = processed_train_data.drop("Feature 13", axis=1)

le = LabelEncoder()
processed_train_data['key'] = le.fit_transform(processed_train_data['key'])
processed_train_data = processed_train_data.values

scaler = MinMaxScaler(feature_range=(0, 1))
processed_train_data = scaler.fit_transform(processed_train_data)

# pca = PCA(n_components=.95)
# principalComponents = pca.fit_transform(processed_train_data) 



# wcss = []
# for i in range(1,15):
#   kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
#   kmeans.fit(scaled)
#   wcss.append(kmeans.inertia_)
# plt.plot(range(1,15), wcss, 'o')
# plt.plot(range(1 , 15) , wcss , '-' , alpha = 0.5)
# plt.title('Elbow Method')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.savefig('Elbow_Method.png')
# plt.show()
# print(wcss)

#scaled.columns = cols

#processed_train_data.to_csv('./test.csv',index=False)

gmm = GaussianMixture(n_components=2).fit(processed_train_data)
labels = gmm.predict(processed_train_data)

# fig, ax = plt.subplots(figsize=(13,11))
# ax = fig.add_subplot(111, projection='3d')
# plt.scatter(scaled[y_kmeans == 0,0],scaled[y_kmeans == 0,1], s= 50, c= 'red',label= 'Cluster 1')
# plt.scatter(scaled[y_kmeans == 1,0], scaled[y_kmeans == 1,1], s= 50, c= 'blue', label= 'Cluster 2')
# plt.scatter(scaled[y_kmeans == 2,0], scaled[y_kmeans == 2,1], s= 50, c= 'green', label= 'Cluster 3')
# plt.scatter(scaled[y_kmeans == 3,0], scaled[y_kmeans == 3,1], s= 50, c= 'cyan', label= 'Cluster 4')
# plt.scatter(scaled[y_kmeans == 4,0], scaled[y_kmeans == 4,1], s= 50, c= 'magenta', label= 'Cluster 5')
# plt.scatter(scaled[y_kmeans == 5,0], scaled[y_kmeans == 5,1], s= 50, c= 'gray', label= 'Cluster 6')
# plt.scatter(scaled[y_kmeans == 6,0], scaled[y_kmeans == 6,1], s= 50, c= 'purple', label= 'Cluster 7')
# plt.scatter(scaled[y_kmeans == 7,0], scaled[y_kmeans == 7,1], s= 50, c= 'pink', label= 'Cluster 8')
# plt.scatter(scaled[y_kmeans == 8,0], scaled[y_kmeans == 8,1], s= 50, c= 'silver', label= 'Cluster 9')

# # centroids
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s= 300, c= 'yellow', label= 'Centroids')
# plt.title('Clusters')
# plt.legend()
# plt.savefig('clusters.png')
# plt.show()


# separate the features

# pca = PCA(n_components=17)
# pca.fit(X_std)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_)
# print(pca.n_components_) 
# X_transformed = pca.transform(X_std)

# kmeans = KMeans(n_clusters=13,init='k-means++',random_state=42)
# kmeans.fit(X_std)

# kmeans.labels_


output = labels
for i in list(set(output)):
    print(i,output.tolist().count(i))
ans = []
for index, row in test_data.iterrows():
    if(output[row['col_1']]==output[row['col_2']]):
        #print(output[row['col_1']],output[row['col_2']])
        ans.append([str(index),str(1)])
    else:
        ans.append([str(index),str(0)])
        
        
with open('ans'+str(0)+'.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['id', 'ans'])
    writer.writerows(ans)


# view the cluster labels
# print(model.labels_)
# print(len(model.labels_))

# view the cluster centers
#print(model.cluster_centers_)