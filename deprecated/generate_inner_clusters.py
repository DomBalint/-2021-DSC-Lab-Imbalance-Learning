import numpy as np 
import pandas as pd 
import os
from sklearn.cluster import KMeans,AgglomerativeClustering,MiniBatchKMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
import pickle


def generated_distance_features(df,centroid_map):
    df=df.drop(["id"],axis=1)
    col=df.columns
    for cls_index,centroids in centroid_map.items():
        for centroid in centroids:
            print(centroid)
            print(centroid.shape)
            for i in range(len(df)):
                index=np.where(centroids==centroid)
                col_name=f"dict_{cls_index}_{index[0][0]}"
                norm_diff=np.linalg.norm(np.array(df[col].iloc[i])-centroid)
                if col_name not in df.columns:
                    print(f"{col_name} not found")
                    df[col_name]=0
                    
                df[col_name].iloc[i]=norm_diff
                
    return df

centroids_map=pickle.load(open("centroids_map.sav","rb"))
df_valid=pd.read_csv("tabular_valid.csv")
df_test=pd.read_csv("tabular_test.csv")


df_valid_generated=generated_distance_features(df_valid,centroids_map)
df_test_generated=generated_distance_features(df_test,centroids_map)

df_valid_generated.to_csv("df_valid_5.csv",index=False)
df_test_generated.to_csv("df_test_5.csv",index=False)

