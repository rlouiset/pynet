import sys
sys.path.append('..')

from pynet.datasets.core import DataManager
from pynet.metrics.core import MetricManager
from pynet.transforms import LabelMapping
from sklearn.preprocessing import StandardScaler
from .utils import *
from .ML_methods import *
from .getters import *

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import shap
import matplotlib.pyplot as plt

display= False

def train_linear_model(xp_name, dataset_config, ml_method_config, metrics_config) :
    print("Let us launch the training of the Linear model ", xp_name)
    ## get label dict, metric and data managers
    label_map = LabelMapping(**dataset_config['LABEL_DICT'])
    data_manager = DataManager(None, dataset_config['ROOT'] + dataset_config['DATA_DIRS']['metadata_path'],
                          number_of_folds=dataset_config['NB_FOLD'],
                          labels_transforms=label_map,
                          labels=['diagnosis'],
                          custom_stratification=dataset_config['CUSTOM_STRATIFICATION_DICT'],
                          sep='\t',
                          N_train_max=dataset_config['N_TRAIN_MAX'],
                          N_train_max_per_label=dataset_config['N_TRAIN_MAX_PER_LABEL'])

    ## load features
    features, features_names = load_features_data(dataset_config['ROOT'], dataset_config['DATA_DIRS'] , dataset_config['FEATURES_DICT'])
    metric_manager = MetricManager(metrics_config, xp_name, features_names)

    ## load test features and labels
    scaler = StandardScaler()
    test_indices = data_manager.dataset['test'].indices
    y_test, X_test = np.array([label_map(label) for label in data_manager.labels[test_indices]]), features[test_indices, :]
    X_test = scaler.fit_transform(X_test)

    print(np.sum(y_test==0))
    print(np.sum(y_test==1))

    ## get best parameters with Grid Search algorithm
    #best_params = get_best_params_w_GS(data_manager, dataset_config, ml_method_config, features, label_map)

    best_params = {'ML_method':{'name':ml_method_config['ML_method']['name'], 'hyperparameters':{}}, 'DR_method':{'name':ml_method_config['DR_method']['name'], 'hyperparameters':{}}}
    for key in ml_method_config['ML_method']['hyperparameters'].keys() :
        best_params['ML_method']['hyperparameters'][key] = ml_method_config['ML_method']['hyperparameters'][key][0]
    for key in ml_method_config['DR_method']['hyperparameters'].keys() :
        best_params['DR_method']['hyperparameters'][key] = ml_method_config['DR_method']['hyperparameters'][key][0]


    for fold in range(dataset_config['NB_FOLD']):
        train_indices = data_manager.dataset['train'][fold].indices
        val_indices = data_manager.dataset['validation'][fold].indices

        X_train, X_val = features[train_indices, :], features[val_indices, :]
        y_train, y_val = np.array([label_map(label) for label in data_manager.labels[train_indices]]), np.array(
            [label_map(label) for label in data_manager.labels[val_indices]])

        ## scale the features
        scaler = StandardScaler()
        X_train, X_val = scaler.fit_transform(X_train), scaler.fit_transform(X_val)

        ## fit Dimensionality reduction method and Machine Learning method on the train features with Grid Search
        DR_method = get_DR_method(best_params['DR_method'])
        X_train_rep = DR_method.fit_transform(X_train, y_train)
        X_val_rep = DR_method.transform(X_val)
        X_test_rep = DR_method.transform(X_test)
        ML_method = get_ML_method(best_params['ML_method'])
        ML_method.fit(X_train_rep, y_train)

        ## compute validation and test metrics
        metric_manager.update_metrics(y_val, X_val_rep, ML_method, phase='VAL')
        metric_manager.update_metrics(y_test, X_test_rep, ML_method, phase='TEST')

    if display :
        shap_values = shap.TreeExplainer(ML_method).shap_values(X_train_rep)
        shap.summary_plot(shap_values, X_train_rep, features_names, plot_size=(18, 10) , show=False)
        plt.savefig('shap_explanation_values_bipolarity.png')

    metric_manager.display_metrics(phase_to_display=['VAL', 'TEST'])
