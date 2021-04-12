import sys
import os

sys.path.append('../pynet')
sys.path.append(os.path.abspath('/home/robin/Desktop/rl264746/pylearn-mulm'))
sys.path.append(os.path.abspath('/home/robin/Desktop/rl264746/nitk'))

from ML_configs.general_config import STRATIFICATION_CONFIGS
from nitk.stats.stats_residualizer import Residualizer
from pynet.datasets.core import DataManager
from pynet.metrics.core import MetricManager
from pynet.transforms import LabelMapping
from utils.save_utils import *
from shutil import copyfile
from GridSearch import *
import pandas as pd
import numpy as np
import logging
import argparse
import json
import yaml


def train_linear_model(xp_config, save=False):
    """ Train a Machine Learning model whose parameters have been given in input. """

    # Get configurations for each object
    dataset_config = xp_config["DATASET_CONFIG"]
    metrics_config = xp_config["METRICS_CONFIG"]
    ml_method_config = xp_config["ML_METHOD_CONFIG"]

    # initialize Label Mapping
    label_map = LabelMapping(**dataset_config['LABEL_DICT'])

    # Initialize Data Paths
    input_paths = [dataset_config['DATA_DIRS']['ROOT'] + input_path_i for input_path_i in
                   dataset_config['DATA_DIRS']['input_path']]
    metadata_paths = [dataset_config['DATA_DIRS']['ROOT'] + metadata_path_i for metadata_path_i in
                      dataset_config['DATA_DIRS']['metadata_path']]

    # Define Data Manager
    data_manager = DataManager(input_paths,
                               metadata_paths,
                               number_of_folds=dataset_config['N_FOLD'],
                               labels_transforms=label_map,
                               labels=['diagnosis'],
                               custom_stratification=STRATIFICATION_CONFIGS[dataset_config['STRATIFICATION']],
                               N_train_max=dataset_config['N_TRAIN_MAX'],
                               N_train_max_per_label=dataset_config['N_TRAIN_MAX_PER_LABEL'],
                               train_size=dataset_config['TRAIN_SIZE'])

    # Get features array and corresponding features names
    features_df = pd.concat(data_manager.inputs, ignore_index=True, sort=False)
    columns_to_keep = [col for col in features_df.columns if col not in dataset_config["FEATURES_FILTER"]["KEYWORDS"]]
    if dataset_config["FEATURES_FILTER"]["SUB_KEYWORDS"] is not None:
        for sub_keyword in dataset_config["FEATURES_FILTER"]["SUB_KEYWORDS"]:
            columns_to_keep = [col for col in columns_to_keep if sub_keyword not in col]
    features_df = features_df[columns_to_keep+['participant_id']]
    features_names = features_df.columns
    confound_features_df = data_manager.metadata_df[["participant_id", "site", "age", "sex", "diagnosis"]]

    # Residuals features (for example : regress out "age + site + sex"  w.r.t "age + sex + site + diagnosis")
    if dataset_config['RESIDUALS'] is not None:
        res_spl = Residualizer(features_df.merge(confound_features_df, on='participant_id'),
                               formula_res=dataset_config['RESIDUALS']['REGRESS_OUT'],
                               formula_full=dataset_config['RESIDUALS']['WITH_RESPECT_TO'])

    # Get Metrics manager
    metric_manager = MetricManager(metrics_config, xp_name, features_names)

    # load test features and labels
    test_indices = data_manager.dataset['test'].indices
    y_test = np.array([label_map(label) for label in data_manager.labels[test_indices]])

    # Apply Grid Search algorithm in order to find best Hyper-parameters for both Dimensionality Reduction and Machine Learning algo
    ml_method_config = GridSearch(data_manager, dataset_config, ml_method_config, np.array(features_df[columns_to_keep]).astype(np.float), label_map)

    # Initialize dictionary
    dict_of_predictions = {str(fold): {} for fold in range(dataset_config['N_FOLD'])}

    # Cross validation
    for fold in range(dataset_config['N_FOLD']):
        # Get train/val features and labels
        train_indices = data_manager.dataset['train'][fold].indices
        val_indices = data_manager.dataset['validation'][fold].indices
        y_train, y_val = np.array([label_map(label) for label in data_manager.labels[train_indices]]), np.array(
            [label_map(label) for label in data_manager.labels[val_indices]])

        X_test_df = features_df.iloc[test_indices, :].merge(confound_features_df.iloc[test_indices, :], on='participant_id')
        X_val_df = features_df.iloc[val_indices, :].merge(confound_features_df.iloc[val_indices, :], on='participant_id')
        X_train_df = features_df.iloc[train_indices, :].merge(confound_features_df.iloc[train_indices, :], on='participant_id')

        # Residuals features (for example : regress out "age + site + sex"  w.r.t "age + sex + site + diagnosis")
        if dataset_config['RESIDUALS'] is not None:
            X_train = res_spl.fit_transform(np.array(X_train_df[columns_to_keep]).astype(np.float), res_spl.get_design_mat()[train_indices, :])
            X_val = res_spl.transform(np.array(X_val_df[columns_to_keep]).astype(np.float), res_spl.get_design_mat()[val_indices, :])
            X_test = res_spl.transform(np.array(X_test_df[columns_to_keep]).astype(np.float), res_spl.get_design_mat()[test_indices, :])

        # Normalize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # fit Dimensionality reduction method and Machine Learning method on the train features with Grid Search
        DR_method = get_DR_method(ml_method_config['DR_METHOD'])
        X_train_rep = DR_method.fit_transform(X_train, y_train)
        X_val_rep = DR_method.transform(X_val)
        X_test_rep = DR_method.transform(X_test)

        # train the ML method
        ML_method = get_ML_method(ml_method_config['ML_METHOD'])
        ML_method.fit(X_train_rep, y_train)

        # Compute Validation/Test metrics
        metric_manager.update_metrics(y_val, X_val_rep, ML_method, phase='VAL')
        metric_manager.update_metrics(y_test, X_test_rep, ML_method, phase='TEST')

        # Save Clustering and Classification results
        if save:
            dict_of_predictions = fullfill_predictions(dict_of_predictions, ml_method_config['ML_METHOD']['name'],
                                                       ML_method, fold, data_manager, features_df, features_names,
                                                       {'TRAIN': X_train_rep, 'TEST': X_test_rep, 'VAL': X_val_rep})

    metric_manager.display_metrics(phase_to_display=['VAL', 'TEST'])

    if save:
        with open("/".join([xp_config['SAVE_FOLDER'], "predictions_file.json"]), "w") as json_file:
            json.dump(dict_of_predictions, json_file)


if __name__ == "__main__":
    # Parse Terminal Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg_file", type=str, dest="cfg_file")
    parser.add_argument("--save", type=bool, dest="save_results", default=False)
    args = parser.parse_args()

    # .yaml file safe load
    with open(args.cfg_file, 'r') as input_cfg:
        try:
            xp_config = yaml.safe_load(input_cfg)
        except yaml.YAMLError as error:
            logging.error(error)

    # Define xp name and logger
    xp_name = xp_config['XP_NAME']

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    if args.save_results:
        folder_name = "/".join(["records", xp_name])
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        sub_folder_name = "_".join([xp_name, str(len(os.listdir(folder_name)))])
        sub_folder_path = "/".join([folder_name, sub_folder_name])
        os.mkdir(sub_folder_path)
        logger.addHandler(logging.FileHandler("/".join([sub_folder_path, "terminal.log"])))
        copyfile(args.cfg_file, "/".join([sub_folder_path, "cfg_file.yaml"]))
        xp_config['SAVE_FOLDER'] = sub_folder_path

    logging.info("Let us launch the training of the experiment " + xp_name)
    train_linear_model(xp_config, save=args.save_results)
