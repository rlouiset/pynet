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
FEATURES_CONF = {'general_features': ['sex', 'age', 'tiv', 'wm', 'wmh', 'gm', 'csf'],
                 'physiologic_features': ['surface_area', 'volume', 'thickness'],
                 'roi': [],
                 }

CUSTOM_STRATIFICATION_CONF = {
            'train': {'study': ['BIOBD'],
                      'diagnosis': ['control', 'bipolar disorder']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'psychotic bipolar disorder']}
            }

LABEL_DICT_CONF = {'psychotic bipolar disorder':1, 'bipolar disorder':1, 'control':0}


## define metrics, ml_method and dataset config
METRICS_CONFIG = [
                {'name': 'AUC', 'phase':['VAL', 'TEST'], 'labels': {'control':0, 'bipolar disorder':1}},
                {'name': 'bacc', 'phase':['VAL', 'TEST']},
                {'name': 'acc', 'phase':['VAL', 'TEST']},
                ]

ML_METHOD_CONFIG = {
                    'DR_method' : { 'name': None, 'hyperparameters' : {} },
                    'ML_method' : { 'name':'SVM',
                                                'hyperparameters' : {'probability':[True], 'kernel':['linear']} },
                    }

DATASET_CONFIG = {'ROOT':ROOT_CONF, 'DATA_DIRS':DATA_DIRS,
                  'CUSTOM_STRATIFICATION_DICT':CUSTOM_STRATIFICATION_CONF, 'LABEL_DICT':LABEL_DICT_CONF, 'FEATURES_DICT':FEATURES_CONF,
                  'NB_FOLD':N_FOLD, 'N_TRAIN_MAX':N_TRAIN_MAX, 'N_TRAIN_MAX_PER_LABEL':N_TRAIN_MAX_PER_LABEL}

## define general config
CONFIG = {'xp_name':'tiny_bip_kfolds', 'dataset_config' : DATASET_CONFIG, 'ml_method_config' : ML_METHOD_CONFIG, 'metrics_config' : METRICS_CONFIG}

train_linear_model(**CONFIG)
