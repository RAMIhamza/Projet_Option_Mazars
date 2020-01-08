# DATA PREPROCESSING
import pandas as pd
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import fcluster
import scipy.spatial.distance as sd

import numpy as np
def classe_age(x):
    if x == '<= 1 ans':
        return 1
    elif x == '1 - 2 ans':
        return 2
    elif x == '2 - 3 ans':
        return 3
    elif x == '3 - 4 ans':
        return 4
    elif x == '4 - 5 ans':
        return 5
    elif x == '> 5 ans':
        return 6
    
    
def franchise_(x):
    return int(x[0])



def preprocessing(data, balance=True, train_size=0.5):
    import pandas as pd
    if balance : 
      data_non_nul=data[data.nombre_de_sinistre>0]
      data_nul=data[data.nombre_de_sinistre==0]
      data_nul_1,data_nul_2=train_test_split(data_nul,train_size=len(data_non_nul)/len(data_nul))
      data_clustering=pd.concat([data_non_nul,data_nul_1])
    else :
      data_clustering = data
    columns_conduc=["Classe_Age_Situ_Cont","Type_Apporteur","Activite"]
    columns_contrat=["Mode_gestion","Zone","Fractionnement","franchise","FORMULE",'Exposition_au_risque']
    columns_vehi=["Age_du_vehicule","ValeurPuissance","Freq_sinistre"]
    data_clustering=data_clustering[columns_conduc+columns_contrat+columns_vehi]    
    data_clustering["Classe_Age_Situ_Cont"]=data_clustering["Classe_Age_Situ_Cont"].apply(classe_age)
    data_clustering["franchise"]=data_clustering["franchise"].apply(franchise_)
    data_clustering[["Type_Apporteur","Activite","Zone","franchise","FORMULE","ValeurPuissance"]].astype('str')
    data_clustering_d=pd.get_dummies(data_clustering)
    data_scaled = normalize(data_clustering_d)
    data_scaled = pd.DataFrame(data_scaled, columns=data_clustering_d.columns)
    if train_size < 1 :
        train_data,test_data = train_test_split(data_scaled,train_size=train_size)
    else :
        train_data,test_data = data_scaled, None
    return train_data,test_data


def build_similarity_graph(X, var=0.01):
    """
    Computes the similarity matrix for a given dataset of samples.
     
    :param X: (n x m) matrix of m-dimensional samples
    :param var: the sigma value for the exponential function, already squared
    :return: W: (n x n) dimensional matrix representing the adjacency matrix of the graph
    """

    n = X.shape[0]

    dists = sd.squareform(sd.pdist(X, "sqeuclidean"))
    W = np.exp(-dists / var)

    return W


def compute_y_hat(Y, W):

  """
  Computes the estimated corrected value for eveyr frequency
     
  :param Y: observed frequency
  :return: Y_hat
           err : mean squared error
  """
  n=Y.shape[0]
  vecteur_1 = np.array([1]*n).reshape(n,1)
  Y_ = Y.reshape(1,n)
  V = np.dot((vecteur_1), Y_)
  R = np.transpose(V)-V

  Y_hat =[]
  for i in range(n):
      y_hat = max(0,Y_[0,i] + np.dot(W[i,:], R[:,i]))
      Y_hat.append(y_hat)
  Y_hat=np.array(Y_hat)
  err = np.sqrt(np.mean((Y-Y_hat)**2))

  return Y_hat, err


def count_zeros(Y_hat,thresh=10**(-10)):
    """
    Count the number of elements inferior to the threshold
     
    :param Y:  frequency
    :return: proportion of values inferior to the threshold
    """
    count = np.sum(Y_hat<thresh)
    return count/len(Y_hat)
