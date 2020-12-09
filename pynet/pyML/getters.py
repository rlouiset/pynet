from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from .ML_methods import *
from .DR_methods import *

from tqdm import tqdm
import numpy as np
import pprint
import math

from .utils import reconfigure_params_to_config_standard

def get_ML_method(ML_method_config) :
    if ML_method_config['name'] == 'k_means' :
        return KMeans(**ML_method_config['hyperparameters'])
    if ML_method_config['name'] == 'HDBSCAN' :
        return HDBSCAN(**ML_method_config['hyperparameters'])
    elif ML_method_config['name'] == 'GaussianMixture' :
        return GaussianMixture(**ML_method_config['hyperparameters'])
    elif ML_method_config['name'] == 'SVM' :
        return SVC(**ML_method_config['hyperparameters'])
    elif ML_method_config['name'] == 'random_forest' :
        return RandomForestClassifier(**ML_method_config['hyperparameters'])
    elif ML_method_config['name'] == 'xgboost' :
        return xgb.XGBClassifier(**ML_method_config['hyperparameters'])
    elif ML_method_config['name'] == 'HYDRA':
        return HYDRA(**ML_method_config['hyperparameters'])
    return NotImplemented

def get_DR_method(DR_method_config) :
    if DR_method_config['name'] is None :
        return BaseDR()
    if DR_method_config['name'] == 'PCA' :
        return PCA(**DR_method_config['hyperparameters'])
    if DR_method_config['name'] == 'ICA' :
        return FastICA(**DR_method_config['hyperparameters'])
    if DR_method_config['name'] == 'RFE' :
        return RFE_DR(**DR_method_config['hyperparameters'])
    if DR_method_config['name'] == 'SHAP' :
        return SHAP(**(DR_method_config['hyperparameters']))
    return NotImplemented


def get_best_params_w_GS(data_manager, dataset_config, ml_method_config, features, label_map):
    if len(ml_method_config['DR_method']['hyperparameters'])==0 and len(ml_method_config['ML_method']['hyperparameters'])==0 :
        return reconfigure_params_to_config_standard({}, ml_method_config)

    ## Train the model and pick the best hyperparameters
    params_grid = ParameterGrid(
        dict(ml_method_config['DR_method']['hyperparameters'], **(ml_method_config['ML_method']['hyperparameters'])))
    best_score, best_params = -math.inf, None
    print('Performing Grid Search on Dimensionality Reduction and Machine Learning methods Hyperparameters.')
    ## perform huge Grid Search
    for params in tqdm(params_grid):
        gs_params_config = reconfigure_params_to_config_standard(params, ml_method_config)
        score = 0
        for fold in range(dataset_config['NB_FOLD']):
            ## get fold train and val set
            train_indices = data_manager.dataset['train'][fold].indices
            val_indices = data_manager.dataset['validation'][fold].indices
            X_train, X_val = features[train_indices, :], features[val_indices, :]
            y_train, y_val = np.array([label_map(label) for label in data_manager.labels[train_indices]]), np.array(
                [label_map(label) for label in data_manager.labels[val_indices]])
            ## rescale the features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val= scaler.transform(X_val)

            ## compute parameters score and update the average one
            DR_method = get_DR_method(gs_params_config['DR_method'])
            X_train = DR_method.fit_transform(X_train, y_train)
            ML_method = get_ML_method(gs_params_config['ML_method'])
            ML_method.fit(X_train, y_train)
            score += ML_method.score(DR_method.transform(X_val), y_val)
        if best_score < (score / dataset_config['NB_FOLD']):
            best_score = (score / dataset_config['NB_FOLD'])
            best_params = gs_params_config

    print("The best parameters we got are :")
    pprint.pprint(best_params)
    return best_params