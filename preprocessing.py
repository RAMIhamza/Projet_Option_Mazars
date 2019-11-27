from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import fcluster
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

def preprocessing(data, balance=True):
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
  data_clustering_d=pd.get_dummies(data_clustering)

  data_scaled = normalize(data_clustering_d)
  data_scaled = pd.DataFrame(data_scaled, columns=data_clustering_d.columns)
  train_data,test_data = train_test_split(data_scaled,train_size=.5)
  return train_data,test_data
