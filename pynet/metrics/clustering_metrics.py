import sklearn.metrics as sk_metrics
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np

def NMI(y, y_pred) :
    return sk_metrics.normalized_mutual_info_score(y, y_pred)

def ARI(y, y_pred) :
    return sk_metrics.adjusted_rand_score(y, y_pred)

def V_measure(y, y_pred) :
    return sk_metrics.v_measure_score(y, y_pred)

def silhouette_score(X_rep, y_pred):
    return sk_metrics.silhouette_score(X_rep, y_pred)

def davies_bouldin(X_rep, y_pred) :
    return sk_metrics.davies_bouldin_score(X_rep, y_pred)
