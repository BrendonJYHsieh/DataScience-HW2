import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import csv

train_data = pd.read_csv("train.csv").drop("song_id", axis=1)
test_data = pd.read_csv("test_3000.csv")

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

kmeans = KMeans(n_clusters=13,init='k-means++',random_state=42)
kmeans.fit(X_transformed)

kmeans.labels_

ans = []
for index, row in test_data.iterrows():
    if(kmeans.labels_[row['col_1']]==kmeans.labels_[row['col_2']]):
        print(kmeans.labels_[row['col_1']],kmeans.labels_[row['col_2']])
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