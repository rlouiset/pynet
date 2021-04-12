import numpy as np
import pandas as pd
from scipy.special import logsumexp


def load_features_from_csv(dataset_config):
    features_df = pd.read_csv("CAT12_metadata_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv", sep='\t')
    columns_to_keep = [col for col in features_df.columns if col not in dataset_config["FEATURES_FILTER"]["KEYWORDS"]]
    for sub_keyword in dataset_config["FEATURES_FILTER"]["SUB_KEYWORDS"]:
        columns_to_keep = [col for col in columns_to_keep if sub_keyword not in col]
    features = np.array(features_df[columns_to_keep]).astype(np.float)
    return features


def apply_label(y, LABEL_DICT_CONF):
    return [LABEL_DICT_CONF[y_i] for y_i in y]


def select_labels(y_pred, y_true, labels):
    right_idx = [(y_i in labels) for y_i in y_true]
    y_pred = np.array(y_pred)[right_idx]
    y_true = np.array(y_true)[right_idx]
    right_labelling = list(np.unique(y_true))
    y_true = np.array([right_labelling.index(y_i) for y_i in y_true])
    return y_pred, y_true


def one_hot_encode(y, n_classes=None):
    ''' utils function in order to turn a label vector into a one hot encoded matrix '''
    if n_classes is None:
        n_classes = np.max(y) + 1
    y_one_hot = np.copy(y)
    return np.eye(n_classes)[y_one_hot]


def sigmoid(x, lambda_=5):
    return 1 / (1 + np.exp(-lambda_ * x))


def py_softmax(x, axis=None):
    """stable softmax"""
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))
