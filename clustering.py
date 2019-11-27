import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

def train_clustering(train_data,method='ward'):
  Z = shc.linkage(train_data.drop(["Freq_sinistre"],axis=1,inplace=False), method=method)
  clusters = fcluster(Z, 8.45, criterion='distance')
  train_data["clusters"]=clusters
  c=np.array(train_data["clusters"].unique())
  centroids = [ np.array(np.mean(train_data[train_data.clusters==i].drop(["Freq_sinistre","clusters"],axis=1,inplace=False))) for i in c ]
  mean_frequencies = [np.mean(train_data.Freq_sinistre[train_data.clusters==i]) for  i in c]
  std_frequencies = [np.std(train_data.Freq_sinistre[train_data.clusters==i]) for  i in c]
  return train_data,centroids,mean_frequencies,std_frequencies,c
  
  
def predict_clustering(test_data,centroids,clusters):
  predicted_clusters = np.ones(len(test_data))
  for i in range (len(test_data)):
    pi = np.array(test_data.drop(["Freq_sinistre"],axis=1,inplace=False))[i]
    predicted_clusters[i] = clusters[np.argmin(np.array([np.linalg.norm(pi - centroids[i]) for i in range (len(centroids))]))]
  test_data["clusters"]=predicted_clusters
  return test_data
  
def stabilite(train_data,test_data,mean_frequencies,std_frequencies,clusters):
  stab = np.zeros((len(clusters),4))
  stab[:,2]=clusters.astype('int')
  stab[:,-1]=std_frequencies
  new_data = pd.concat([train_data,test_data],ignore_index=True)
  for i in range (len(clusters)):
    stab[i,0]=mean_frequencies[i]*10
    stab[i,1]=np.mean(new_data.Freq_sinistre[new_data.clusters==clusters[i]])*10
  return stab
