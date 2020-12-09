from pynet.pyML.main import train_linear_model

## root and data directory paths
ROOT_CONF = "/home/robin/Desktop/rl264746/neurospin/psy_sbox"
DATA_DIRS = {'input_path': '',
             'metadata_path': '/all_t1mri_mwp1_participants_merged_features.tsv'}

## define folds, labels cropping in order to equilibrate the dataset
N_FOLD = 10
N_TRAIN_MAX = 600
N_TRAIN_MAX_PER_LABEL = 250


## features and labels config
FEATURES_CONF = {'general_features': ['wm', 'wmh', 'gm', 'csf', 'sex', 'age', 'tiv'], #
                 #'physiologic_features': ['surface_area',  'thickness', 'volume'],
                 'roi': [],
                 }

CUSTOM_STRATIFICATION_CONF = {
            'train': {'study': ['SCHIZCONNECT-VIP'],
                      'diagnosis': ['control', 'schizophrenia']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'schizophrenia']}
            }

LABEL_DICT_CONF = {'schizophrenia':1, 'control':0}


## define metrics, ml_method and dataset config
METRICS_CONFIG = [
                {'name': 'AUC', 'phase':['VAL', 'TEST']},
                {'name': 'bacc', 'phase':['VAL', 'TEST']},
                {'name': 'acc', 'phase':['VAL', 'TEST']},
                ]

ML_METHOD_CONFIG = {
                    'DR_method' : { 'name': None, 'hyperparameters' : {} },
                    'ML_method' : { 'name':'SVM',
                                                'hyperparameters' : {'probability':[True], 'kernel':['rbf']}}, #'max_depth': [2, 3, 4, 5, 10], 'eta':[0.01, 0.05, 0.1, 0.2, 0.5, 1, 3]}
                    }

DATASET_CONFIG = {'ROOT':ROOT_CONF, 'DATA_DIRS':DATA_DIRS,
                  'CUSTOM_STRATIFICATION_DICT':CUSTOM_STRATIFICATION_CONF, 'LABEL_DICT':LABEL_DICT_CONF, 'FEATURES_DICT':FEATURES_CONF,
                  'NB_FOLD':N_FOLD, 'N_TRAIN_MAX':N_TRAIN_MAX, 'N_TRAIN_MAX_PER_LABEL':N_TRAIN_MAX_PER_LABEL}

## define general config
CONFIG = {'xp_name':'tiny_scz_kfolds', 'dataset_config' : DATASET_CONFIG, 'ml_method_config' : ML_METHOD_CONFIG, 'metrics_config' : METRICS_CONFIG}

train_linear_model(**CONFIG)
