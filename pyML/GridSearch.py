from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from  tqdm import tqdm
from getters import *
import numpy as np
import logging
import math

def GridSearch(data_manager, dataset_config, ml_method_config, features, label_map):
    ''' Perform Grid Search on hyperparameters space. '''

    # Fix the empty DR_METHOD case
    if ml_method_config["DR_METHOD"] is None:
        ml_method_config["DR_METHOD"] = {'name': None, 'hyperparameters': {}}

    # Get Hyperparameters to tune (set as a list in the config file)
    ml_method_hyperparameters_to_tune = {'ML_' + key: value for (key, value) in
                                         ml_method_config["ML_METHOD"]["hyperparameters"].items()
                                         if isinstance(value, list)}
    dr_method_hyperparameters_to_tune = {'DR_' + key: value for (key, value) in
                                         ml_method_config["DR_METHOD"]["hyperparameters"].items()
                                         if isinstance(value, list)}

    # Set the other constant Hyperparameters
    ml_method_hyperparameters = {key: value for (key, value) in ml_method_config["ML_METHOD"]["hyperparameters"].items()
                                 if not isinstance(value, list)}
    dr_method_hyperparameters = {key: value for (key, value) in ml_method_config["DR_METHOD"]["hyperparameters"].items()
                                 if not isinstance(value, list)}

    # Train the model and pick the best hyperparameters
    parameters_grid = ParameterGrid(dict(ml_method_hyperparameters_to_tune, **(dr_method_hyperparameters_to_tune)))

    best_score = -math.inf
    best_ml_method_hyperparameters = ml_method_hyperparameters.copy()
    best_dr_method_hyperparameters = dr_method_hyperparameters.copy()

    logging.info('----------------------------------------------------------')
    logging.info('Performing Grid Search on Dimensionality Reduction and Machine Learning methods Hyperparameters.')

    # Define Standard Scaler
    scaler = StandardScaler()

    # Perform huge Grid Search
    if len(parameters_grid) > 1 :
        for parameters in tqdm(parameters_grid):

            ml_method_hyperparameters = update_parameters(ml_method_hyperparameters, parameters, prefix='ML_')
            dr_method_hyperparameters = update_parameters(dr_method_hyperparameters, parameters, prefix='DR_')

            ml_method_config['ML_METHOD'].update({'hyperparameters': ml_method_hyperparameters})
            ml_method_config['DR_METHOD'].update({'hyperparameters': dr_method_hyperparameters})

            scores = []
            for fold in range(dataset_config['N_FOLD']):
                train_indices = data_manager.dataset['train'][fold].indices
                val_indices = data_manager.dataset['validation'][fold].indices

                # Get train/val features and labels
                X_train, X_val = features[train_indices, :], features[val_indices, :]
                y_train, y_val = np.array([label_map(label) for label in data_manager.labels[train_indices]]), np.array(
                    [label_map(label) for label in data_manager.labels[val_indices]])

                # Normalize the features
                X_train, X_val = scaler.fit_transform(X_train), scaler.fit_transform(X_val)

                # fit Dimensionality reduction method and Machine Learning method on the train features with Grid Search
                DR_method = get_DR_method(ml_method_config['DR_METHOD'])
                X_train_rep = DR_method.fit_transform(X_train, y_train)
                X_val_rep = DR_method.transform(X_val)

                ML_method = get_ML_method(ml_method_config['ML_METHOD'])
                ML_method.fit(X_train_rep, y_train)
                scores.append(ML_method.score(DR_method.transform(X_val_rep), y_val))

            score = np.mean(scores)
            if best_score < score:
                best_score = score
                best_ml_method_hyperparameters = ml_method_hyperparameters.copy()
                best_dr_method_hyperparameters = dr_method_hyperparameters.copy()
    else :
        best_ml_method_hyperparameters = update_parameters(ml_method_hyperparameters, parameters_grid[0], prefix='ML_')
        best_dr_method_hyperparameters = update_parameters(dr_method_hyperparameters, parameters_grid[0], prefix='DR_')

    ml_method_config["ML_METHOD"]["hyperparameters"] = best_ml_method_hyperparameters
    ml_method_config["DR_METHOD"]["hyperparameters"] = best_dr_method_hyperparameters

    logging.info("Grid Search yielded the following ML hyperparameters : : %s", best_ml_method_hyperparameters)
    logging.info("Grid Search yielded the following DR hyperparameters : : %s", best_dr_method_hyperparameters)
    logging.info('----------------------------------------------------------')

    return ml_method_config


def update_parameters(prefix_method_hyperparameters, parameters, prefix='ML_'):
    for (key, value) in parameters.items():
        if key[:3] == prefix:
            prefix_method_hyperparameters.update({key[3:]: value})
    return prefix_method_hyperparameters
