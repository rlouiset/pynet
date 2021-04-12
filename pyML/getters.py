import os
import sys

sys.path.append('../pynet')
sys.path.append(os.path.abspath('/home/robin/Desktop/rl264746/UCSL'))

import shap
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import FastICA, PCA
from ucsl.ucsl_classifier import UCSL_C
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import xgboost as xgb
import numpy as np


def get_ML_method(ML_method_config):
    if ML_method_config['name'] == 'KMeans':
        return KMeans(**ML_method_config['hyperparameters'])
    elif ML_method_config['name'] == 'SVM':
        return SVC(**ML_method_config['hyperparameters'])
    elif ML_method_config['name'] == 'GaussianMixture':
        return GaussianMixture(**ML_method_config['hyperparameters'])
    elif ML_method_config['name'] == 'xgboost':
        return xgb.xgboost(**ML_method_config['hyperparameters'])
    elif ML_method_config['name'] == 'UCSL_C':
        return UCSL_C(**ML_method_config['hyperparameters'])
    return NotImplemented


def get_DR_method(DR_method_config):
    if DR_method_config['name'] is None :
        return Identity()
    if DR_method_config['name'] == 'SHAP':
        return SHAP(**DR_method_config['hyperparameters'])
    if DR_method_config['name'] == 'PCA':
        return PCA(**DR_method_config['hyperparameters'])
    if DR_method_config['name'] == 'ICA':
        return FastICA(**DR_method_config['hyperparameters'])
    return NotImplemented

class Identity(object):
    """ Base Machine Learning classifier / clustering method
    """
    def __init__(self, name="Base empty Machine Learning object"):
        self.name = name

    def fit(self, X_train, y_train):
        pass

    def fit_transform(self, X_train, y_train=None):
        return X_train

    def transform(self, X_val):
        return X_val

class SHAP(object):
    """ Computes and stores the average and current value.
    """
    def __init__(self, name):
        classifier_config = {'name':name, 'hyperparameters':{}}
        self.classifier = RandomForestClassifier()
        self.k_explainer = None

    def fit(self, X_train, y_train):
        self.classifier.fit(X_train, self.preprocess(y_train))

    def fit_transform(self, X_train, y_train):
        self.classifier.fit(X_train, self.preprocess(y_train))
        shap_values = shap.TreeExplainer(self.classifier).shap_values(X_train)
        return shap_values[1]

    def transform(self, X_val):
        shap_values = shap.TreeExplainer(self.classifier).shap_values(X_val)
        return shap_values[1]

    def preprocess(self, y):
        y_binary = np.copy(y)
        y_binary[y_binary>=1] = 1
        return y_binary
