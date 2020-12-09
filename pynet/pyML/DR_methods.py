import shap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import xgboost as xgb
import numpy as np
from sklearn.svm import SVC

class BaseDR(object) :
    def __init__(self):
        self.name = "Base empy dimensionality reduction object"

    def fit(self, X_train, y_train):
        pass

    def fit_transform(self, X_train, y_train=None):
        return X_train

    def transform(self, X_val):
        return X_val

class RFE_DR(object) :
    def __init__(self, name, n_features_to_select):
        self.name = "RFE reduction object"
        self.classifier = RandomForestClassifier()
        self.selector = RFE(self.classifier, n_features_to_select=n_features_to_select, step=1)

    def fit(self, X_train, y_train):
        self.selector.fit(X_train, y_train)

    def fit_transform(self, X_train, y_train=None):
        X_train = self.selector.fit_transform(X_train, y_train)
        return X_train

    def transform(self, X_val):
        return self.selector.transform(X_val)

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
