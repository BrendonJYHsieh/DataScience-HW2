import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import csv

train_data = pd.read_csv("train.csv")
train_data = train_data.drop("song_id", axis=1)
test_data = pd.read_csv("test_3000.csv")

groups = train_data.groupby(train_data.columns[1:].to_list())

processed_train_data = train_data
#processed_train_data = processed_train_data.drop("Feature 13", axis=1)

dummy = pd.get_dummies(processed_train_data["Feature 13"],  prefix="Feature 13", drop_first=True)
processed_train_data = pd.concat([processed_train_data, dummy], axis=1)
processed_train_data = processed_train_data.drop("Feature 13", axis=1)

# le = LabelEncoder()
# processed_train_data['Feature 13'] = le.fit_transform(processed_train_data['Feature 13'])

# scaler = MinMaxScaler(feature_range=(0, 1))
# processed_train_data = pd.DataFrame(scaler.fit_transform(processed_train_data), columns=processed_train_data.columns)


scaler = StandardScaler()
X_std = scaler.fit_transform(processed_train_data)

processed_train_data.to_csv('./test.csv',index=False)




# separate the features

pca = PCA(n_components=17)
pca.fit(X_std)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)
X_transformed = pca.transform(X_std)

wcss = []
for i in range(1,21):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X_transformed)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(10,8))
plt.plot(range(1,21),wcss,marker='o',linestyle='--')
plt.xlabel('Num of Clusters')
plt.ylabel('WCSS')
plt.show()

# create and fit the k-means model
# for i in range(7,13):
#     ans = []
#     print(i)
#     gm = GaussianMixture(n_components=i).fit(X_transformed)
#     centers = gm.means_
#     pred = gm.predict(X_transformed)

#     for index, row in test_data.iterrows():
#         if(pred[row['col_1']]==pred[row['col_2']]):
#             ans.append([str(index),str(1)])
#         else:
#             ans.append([str(index),str(0)])
            
            
#     with open('ans'+str(i)+'.csv', 'w', newline='') as outfile:
#         writer = csv.writer(outfile)
#         writer.writerow(['id', 'ans'])
#         writer.writerows(ans)


# view the cluster labels
# print(model.labels_)
# print(len(model.labels_))

# view the cluster centers
#print(model.cluster_centers_)