
from preprocessing import *
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def load_DB(path, decimal='.') :
    # path = "./train_contrats_approx.csv"
    data = pd.read_csv(path, sep=";", engine="python", decimal=decimal)
    return data

def clean_sin(data_sinistres) :

    data_sinistres_ = data_sinistres.drop('Unnamed: 0', axis=1,)
    n = len(data_sinistres_)
    data_sinistres_ = data_sinistres_.drop_duplicates()
    p = len(data_sinistres_)
    cols = ["IMMAT","CHARGE_SINISTRE", "Date_Deb_Situ", "ANNEE", "Date_Fin_Situ", "Num_contrat"]
    data_sinistres_ = data_sinistres_[cols]
    data_sinistres_ = data_sinistres_.groupby(["IMMAT", "Date_Deb_Situ", "ANNEE", "Date_Fin_Situ", "Num_contrat"], as_index=False).sum()
    print(f"Data sinistre : Dropped {n-p} duplicates, merged {p-len(data_sinistres_)} same profils")
    return data_sinistres_

def clusterize(data, method):
    labels = None

    if method == "gmm":
        print("GMM: Optimal number of clusters : 8 (ARI)")
        estimator = GaussianMixture(n_components=8,
                                    covariance_type='full', max_iter=500, random_state=0)
        labels = estimator.fit_predict(data)
        # data_ = data.copy()
        # data_["cluster"] = GMM_labels

    if method == "kmeans":
        print("KMeans: Optimal number of clusters : 3 (ARI)")
        estimator = KMeans(n_clusters=3, random_state=0).fit(data)
        labels = estimator.labels_

    return estimator, labels

def calcul_prime(data, data_pre):
    data_pre["Freq_sinistre"] = data["Freq_sinistre"]
    data_pre["CHARGE_SINISTRE"] = data["CHARGE_SINISTRE"]
    data_pre["count"] = 1
    data_pre["count_charge_sinistre"] = (data["CHARGE_SINISTRE"] != 0).astype('int')
    # On prend frequence moyenne, et la charge moyenne calculee sur les charges non nulles
    data_clust = data_pre.groupby("cluster", as_index=False).sum()

    # Calculer prime pour chaque cluster
    data_clust["CHARGE_SINISTRE"] = data_clust["CHARGE_SINISTRE"] / data_clust["count_charge_sinistre"]
    data_clust["Freq_sinistre"] = data_clust["Freq_sinistre"] / data_clust["count"]
    data_clust["Prime"] = data_clust["CHARGE_SINISTRE"] * data_clust["Freq_sinistre"]

    return data_clust

def main(path, path_sin, path_test, method):

# Preprocessing
    data = load_DB(path)
    data["Freq_sinistre"] = data["nombre_de_sinistre"] / data["Exposition_au_risque"]
    data_preprocessed = preprocessing(data, balance=False, train_size=1)[0]
    data_pre = data_preprocessed.copy()
    print("Train data loaded and preprocessed")

# Obtenir le clustering
    estimator, labels = clusterize(data_pre, method)
    data["cluster"] = labels
    data_pre["cluster"] = labels

    # Merge la bdd des charges sinistre

    data_sinistres = load_DB(path_sin,decimal=",")
    data_sinistres = clean_sin(data_sinistres)
    data = data.merge(data_sinistres, on=["IMMAT", "Date_Deb_Situ", "ANNEE", "Date_Fin_Situ", "Num_contrat"], how="left")
    data.fillna({"CHARGE_SINISTRE":0},inplace=True)

# Calculer freq moyen et charge pour chaque cluster
    data_prime = calcul_prime(data, data_pre)
    data_prime = data_prime[["cluster", "Prime"]]

# Output les primes pour chaque somehow
    data_test = load_DB(path_test)
    data_test_pre = preprocessing(data_test, balance=False, train_size=1)[0]
    print("Test data loaded and preprocessed")

    for column in data_preprocessed.columns :
        if column not in data_test.columns :
            data_test_pre[column] = 0
    preds = estimator.predict(data_test_pre)
    data_test["cluster"]=preds
    data_test = data_test.merge(data_prime, on="cluster", sort=False, how="inner").sort_values("Unnamed: 0").reset_index(drop=True)

    print('End')
    return data_test

if __name__ == "__main__" :

    path_approx = "./train_contrats_approx.csv"
    path_original = "./train_contrats.csv"
    path_sin = "./train_sinistres.csv"
    path_test = "./test_contrats.csv"
    method = "kmeans" # gmm / kmeans

    # data_test_approx = main(path_approx, path_sin, path_test, method)
    data_test_original = main(path_original, path_sin, path_test, method)

