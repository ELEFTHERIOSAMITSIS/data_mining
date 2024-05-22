from sklearn.cluster import MiniBatchKMeans,KMeans,DBSCAN,Birch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
import numpy as np

def kmeans(data,labels):
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
     


def minibatch_kmeans(data,labels,clusters):
        
    mbk = MiniBatchKMeans(init ='k-means++',max_iter=20, n_clusters = clusters,
                            batch_size = 512, n_init = 10,
                            max_no_improvement = 10, verbose = 0)
    mbk.fit(data)

    silhouette_score_val=silhouette_score(data, mbk.labels_)
    print(silhouette_score_val)
 
    df = pd.DataFrame({'real_labels': labels, 'clusters': mbk.labels_})

    return mbk.labels_,df,silhouette_score_val

def dbscan(data,labels):
   
    dbs = DBSCAN(n_jobs=-1,min_samples=50,eps=1)
    dbs.fit(data)

    silhouette_score_val=silhouette_score(data, dbs.labels_)
    print(f'DBSCAN silouette score --->{silhouette_score_val}')

    df = pd.DataFrame({'real_labels': labels, 'clusters': dbs.labels_})

    return dbs.labels_,df,silhouette_score_val

    
def birch(data,labels,clusters,th,bf):
   
    birch = Birch(n_clusters=clusters,threshold = th,branching_factor=bf)
    birch.fit(data)
    silhouette_score_val=silhouette_score(data, birch.labels_)
    print(f'BIRCH silouette score --->{silhouette_score_val}')

    df = pd.DataFrame({'real_labels': labels, 'clusters': birch.labels_})

    return birch.labels_,df,silhouette_score_val