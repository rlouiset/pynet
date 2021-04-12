from pyML.main import train_linear_model

## root and data directory paths
ROOT_CONF = "/home/robin/Desktop/rl264746/neurospin/psy_sbox/"
DATA_DIRS = {'input_path': '',
             'metadata_path':'CAT12_metadata_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv',
             'features_path':'CAT12_metadata_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'
             }

## define folds, labels cropping in order to equilibrate the dataset
N_FOLD = 10
N_TRAIN_MAX = 940
N_TRAIN_MAX_PER_LABEL = 500


## features and labels config
FEATURES_CONF = {'general_features': ['participant_id', ], # 'wm', 'gm', 'csf', 'sex', 'age', 'tiv'
                 #'FS': ['FS_volume', 'FS_subcort_vol'],  # 'thickness', 'surface_area'
                 'CAT12': ['cat12_ROI'],
                 }

LABEL_DICT_CONF = {'relative of proband with psychotic bipolar disorder':6, 'relative of proband with schizoaffective disorder':5, 'relative of proband with schizophrenia':4,
                    'schizoaffective disorder':3, 'psychotic bipolar disorder':2, 'bipolar disorder':2, 'FEP':1, 'schizophrenia':1, 'control':0}

CUSTOM_STRATIFICATION_CONF = {
            'train': {'study': ['SCHIZCONNECT-VIP', 'BIOBD'],
                      'diagnosis': ['control', 'schizophrenia', 'bipolar disorder']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'schizophrenia' , 'psychotic bipolar', 'psychotic bipolar disorder', 'bipolar disorder']}
            }


## define metrics, ml_method and dataset config
METRICS_CONFIG = [
                {'name': 'AUC', 'phase':['VAL', 'TEST'], 'testing_label_mapping': {0:0, 1:1, 2:1}, 'predict_testing_label_mapping' : False},
                {'name': 'bacc', 'phase':['VAL', 'TEST'], 'testing_label_mapping': {0:0, 1:1, 2:1}, 'predict_testing_label_mapping' : False},
                {'name': 'acc', 'phase':['VAL', 'TEST'], 'testing_label_mapping': {0:0, 1:1, 2:1}, 'predict_testing_label_mapping' : False},
                {'name': 'ARI', 'phase': ['VAL', 'TEST'], 'testing_label_mapping': {1:0, 2:1}, 'predict_testing_label_mapping' : False},
                {'name': 'V_measure', 'phase':['VAL', 'TEST'], 'testing_label_mapping': {1:0, 2:1}, 'predict_testing_label_mapping' : False},
                {'name': 'Cluster Consistency', 'phase':['TEST'], 'testing_label_mapping': {1:0, 2:1}, 'predict_testing_label_mapping' : False},
                ]

ML_METHOD_CONFIG = {
                    'DR_method' : { 'name': None, 'hyperparameters' : {} },
                    'ML_method' : { 'name':'HYDRA',
                                                'hyperparameters' :
                                                {
                                                'C':[0.1],
                                                'n_consensus':[10],
                                                'n_iterations':[10],
                                                'clustering_strategy':['original'],
                                                'initialization_type':['DPP'],
                                                'consensus':['direction'],

                                                'n_clusters_per_label':[{0:2, 1:2}],
                                                'which_label_clusters_we_predict':[1],

                                                'training_label_mapping':[{0:0, 1:1, 2:1}]
                                                }
                                    },
                    }

DATASET_CONFIG = {'ROOT':ROOT_CONF, 'DATA_DIRS':DATA_DIRS,
                  'CUSTOM_STRATIFICATION_DICT':CUSTOM_STRATIFICATION_CONF, 'LABEL_DICT':LABEL_DICT_CONF, 'FEATURES_DICT':FEATURES_CONF,
                  'NB_FOLD':N_FOLD, 'N_TRAIN_MAX':N_TRAIN_MAX, 'N_TRAIN_MAX_PER_LABEL':N_TRAIN_MAX_PER_LABEL}

## define general config
CONFIG = {'xp_name':'', 'dataset_config' : DATASET_CONFIG, 'ml_method_config' : ML_METHOD_CONFIG, 'metrics_config' : METRICS_CONFIG}

train_linear_model(**CONFIG)

