from sklearn.cluster import MiniBatchKMeans,KMeans,HDBSCAN,DBSCAN,Birch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
import numpy as np

def try_kmeans(data,labels):
    x=data
    y=labels
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    X_train_std = StandardScaler().fit_transform(X_train)

    inertias = []
    silhouette_scores = []
    db_scores = []
    K_range = range(1, 10)  # Adjust the range based on your dataset

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=11, n_init='auto').fit(X_train_std)
        inertias.append(kmeans.inertia_)
        if(k != 1):
            silhouette_scores.append(silhouette_score(X_train_std, kmeans.labels_))
            db_scores.append(davies_bouldin_score(X_train_std, kmeans.labels_))
    print(silhouette_scores)
     


def try_minibatch_kmeans(data,labels,clusters):
    x=data
    y=labels
    inertias = []
    silhouette_scores = []
    
    X_train_std = StandardScaler().fit_transform(data)
    
    mbk = MiniBatchKMeans(init ='k-means++',max_iter=20, n_clusters = clusters,
                            batch_size = 2048, n_init = 10,
                            max_no_improvement = 10, verbose = 0)
    mbk.fit(X_train_std)

   
    print(silhouette_score(X_train_std, mbk.labels_))
 
    df = pd.DataFrame({'real_labels': labels, 'clusters': mbk.labels_})

    return mbk.labels_,df

def try_dbscan(data,labels):
    x=data
    y=labels
    inertias = []
    silhouette_scores = []

    X_train_std = StandardScaler().fit_transform(data)
    
    dbs = DBSCAN(n_jobs=-1,min_samples=50,eps=1)
    dbs.fit(X_train_std)

   
    print(f'DBSCAN silouette score --->{silhouette_score(X_train_std, dbs.labels_)}')

    df = pd.DataFrame({'real_labels': labels, 'clusters': dbs.labels_})

    return dbs.labels_,df

    
def try_birch(data,labels):
    X_train_std = StandardScaler().fit_transform(data)
    birch = Birch(n_clusters=12,threshold = 0.01,branching_factor=5)
    birch.fit(X_train_std)
    print(f'BIRCH silouette score --->{silhouette_score(X_train_std, birch.labels_)}')

    df = pd.DataFrame({'real_labels': labels, 'clusters': birch.labels_})

    return birch.labels_,df