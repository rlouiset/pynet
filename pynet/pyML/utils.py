import pandas as pd
import numpy as np
import scipy
from sklearn.cluster import KMeans
import math

def load_features_data(root, data_dirs, features_dict) :
    ## import features and their names
    data_df = pd.read_csv(root + data_dirs['metadata_path'], sep='\t')
    header = np.array(list(data_df.keys()))
    features_index = [i for i in range(len(header)) if header[i] in features_dict["general_features"]]
    features_names = [header[i] for i in range(len(header)) if header[i] in features_dict["general_features"]]

    ## build the features array
    for key in features_dict.keys() :
        if key == "roi" :
            roi_index = [i for i in range(len(header)) if header[i][:4]=='roi_']
            features_index.extend(roi_index)
            features_names.extend(header[roi_index])
        if key == "physiologic_features" :
            for feature_name in features_dict[key] :
                physio_index = [i for i in range(len(header)) if header[i][:len(feature_name)]==feature_name]
                features_index.extend(physio_index)
                features_names.extend(header[physio_index])
    return np.array(data_df)[:, features_index], features_names

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def reconfigure_params_to_config_standard(params, ml_method_config) :
    config_standard = {'DR_method': {'name': ml_method_config['DR_method']['name'], 'hyperparameters': {}},
                   'ML_method': {'name': ml_method_config['ML_method']['name'], 'hyperparameters': {}}}
    for key in params.keys():
        if key in list(ml_method_config['DR_method']['hyperparameters'].keys()):
            config_standard['DR_method']['hyperparameters'][key] = params[key]
        elif key in list(ml_method_config['ML_method']['hyperparameters'].keys()):
            config_standard['ML_method']['hyperparameters'][key] = params[key]
    return config_standard


def proportional_assign(l, d):
    """
    Proportional assignment based on margin
    :param l: int
    :param d: int
    :return:
    """
    np.seterr(divide='ignore', invalid='ignore')
    invL = np.divide(1, l)
    idx = np.isinf(invL)
    invL[idx] = d[idx]

    for i in range(l.shape[0]):
        pos = np.where(invL[i, :] > 0)[0]
        neg = np.where(invL[i, :] < 0)[0]
        if pos.size != 0:
            invL[i, neg] = 0
        else:
            invL[i, :] = np.divide(invL[i, :], np.amin(invL[i, :]))
            invL[i, invL[i, :] < 1] = 0

    S = np.multiply(invL, np.divide(1, np.sum(invL, axis=1))[:, np.newaxis])

    return S

def sample_dpp(evalue, evector, k=None):
    """
    sample a set Y from a dpp.  evalue, evector are a decomposed kernel, and k is (optionally) the size of the set to return
    :param evalue: eigenvalue
    :param evector: normalized eigenvector
    :param k: number of cluster
    :return:
    """
    if k == None:
        # choose eigenvectors randomly
        evalue = np.divide(evalue, (1 + evalue))
        evector = np.where(np.random.random(evalue.shape[0]) <= evalue)[0]
    else:
        v = sample_k(evalue, k)  ## v here is a 1d array with size: k

    k = v.shape[0]
    v = v.astype(int)
    v = [i - 1 for i in
         v.tolist()]  ## due to the index difference between matlab & python, here, the element of v is for matlab
    V = evector[:, v]

    ## iterate
    y = np.zeros(k)
    for i in range(k, 0, -1):
        ## compute probabilities for each item
        P = np.sum(np.square(V), axis=1)
        P = P / np.sum(P)

        # choose a new item to include
        y[i - 1] = np.where(np.random.rand(1) < np.cumsum(P))[0][0]
        y = y.astype(int)

        # choose a vector to eliminate
        j = np.where(V[y[i - 1], :])[0][0]
        Vj = V[:, j]
        V = np.delete(V, j, 1)

        ## Update V
        if V.size == 0:
            pass
        else:
            V = np.subtract(V, np.multiply(Vj, (V[y[i - 1], :] / Vj[y[i - 1]])[:,
                                               np.newaxis]).transpose())  ## watch out the dimension here

        ## orthogonalize
        for m in range(i - 1):
            for n in range(m):
                V[:, m] = np.subtract(V[:, m], np.matmul(V[:, m].transpose(), V[:, n]) * V[:, n])

            V[:, m] = V[:, m] / np.linalg.norm(V[:, m])

    y = np.sort(y)

    return y

def sample_k(lambda_value, k):
    """
    Pick k lambdas according to p(S) \propto prod(lambda \in S)
    :param lambda_value: the corresponding eigenvalues
    :param k: the number of clusters
    :return:
    """

    ## compute elementary symmetric polynomials
    E = elem_sym_poly(lambda_value, k)

    ## ietrate over the lambda value
    num = lambda_value.shape[0]
    remaining = k
    S = np.zeros(k)
    while remaining > 0:
        # compute marginal of num given that we choose remaining values from 0:num-1
        if num == remaining:
            marg = 1
        else:
            marg = lambda_value[num - 1] * E[remaining - 1, num - 1] / E[remaining, num]

        # sample marginal
        if np.random.rand(1) < marg:
            S[remaining - 1] = num
            remaining = remaining - 1
        num = num - 1
    return S

def elem_sym_poly(lambda_value, k):
    """
    given a vector of lambdas and a maximum size k, determine the value of
    the elementary symmetric polynomials:
    E(l+1,n+1) = sum_{J \subseteq 1..n,|J| = l} prod_{i \in J} lambda(i)
    :param lambda_value: the corresponding eigenvalues
    :param k: number of clusters
    :return:
    """
    N = lambda_value.shape[0]
    E = np.zeros((k + 1, N + 1))
    E[0, :] = 1

    for i in range(1, k + 1):
        for j in range(1, N + 1):
            E[i, j] = E[i, j - 1] + lambda_value[j - 1] * E[i - 1, j - 1]

    return E

def consensus_clustering(clustering_results, k):
    num_pt = clustering_results.shape[0]
    cooccurence_matrix = np.zeros((num_pt, num_pt))

    for i in range(num_pt - 1):
        for j in range(i + 1, num_pt):
            cooccurence_matrix[i, j] = sum(clustering_results[i, :] == clustering_results[j, :])

    cooccurence_matrix = np.add(cooccurence_matrix, cooccurence_matrix.transpose())
    ## here is to compute the Laplacian matrix
    Laplacian = np.subtract(np.diag(np.sum(cooccurence_matrix, axis=1)), cooccurence_matrix)

    Laplacian_norm = np.subtract(np.eye(num_pt), np.matmul(
        np.matmul(np.diag(1 / np.sqrt(np.sum(cooccurence_matrix, axis=1))), cooccurence_matrix),
        np.diag(1 / np.sqrt(np.sum(cooccurence_matrix, axis=1)))))
    ## replace the nan with 0
    Laplacian_norm = np.nan_to_num(Laplacian_norm)

    ## check if the Laplacian norm is symmetric or not, because matlab eig function will automatically check this, but not in numpy or scipy
    evalue, evector = scipy.linalg.eigh(Laplacian_norm)

    ## check if the eigen vector is complex
    if np.any(np.iscomplex(evector)):
        evalue, evector = scipy.linalg.eigh(Laplacian)

    ## create the kmean algorithm with sklearn
    kmeans = KMeans(n_clusters=k, n_init=20).fit(evector.real[:, 0: k])
    final_predict = kmeans.labels_

    return final_predict
