from pynet.pyML.main import train_linear_model

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

## root and data directory paths
ROOT_CONF = "/home/robin/Desktop/rl264746/neurospin/psy_sbox"
DATA_DIRS = {'input_path': '',
             'metadata_path': '/all_t1mri_mwp1_participants_merged_features.tsv'}

## define folds, labels cropping in order to equilibrate the dataset
N_FOLD = 10
N_TRAIN_MAX = 270
N_TRAIN_MAX_PER_LABEL = 250


## features and labels config
FEATURES_CONF = {'general_features': ['wm', 'wmh', 'gm', 'csf'], # 'sex', 'age', 'tiv'
                 #'physiologic_features': ['surface_area', 'volume', 'thickness'],
                 'roi': [],
                 }

LABEL_DICT_CONF = {'relative of proband with psychotic bipolar disorder':5, 'relative of proband with schizoaffective disorder':4, 'relative of proband with schizophrenia':3,
                    'schizoaffective disorder':3, 'psychotic bipolar disorder':2, 'bipolar disorder':2, 'FEP':1, 'schizophrenia':1, 'control':0}

CUSTOM_STRATIFICATION_CONF = {
            'train': {'study': ['SCHIZCONNECT-VIP',], #  'BIOBD'
                      'diagnosis': ['schizophrenia',]}, #  'bipolar disorder'
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['schizophrenia', #'psychotic bipolar disorder', #'schizoaffective disorder',
                                   #'relative of proband with schizophrenia',
                                   #'relative of proband with schizoaffective disorder',
                                   #'relative of proband with psychotic bipolar disorder'
                                  ]}
            }


## define metrics, ml_method and dataset config
METRICS_CONFIG = [
                #{'name': 'ARI', 'phase': ['VAL', 'TEST']},
                #{'name': 'V_measure', 'phase': ['VAL', 'TEST']},
                {'name': 'DaviesBouldin', 'phase': ['VAL', 'TEST']},
                {'name': 'SilhouetteScore', 'phase': ['VAL', 'TEST']},
                ]

ML_METHOD_CONFIG = {'ML_method' : { 'name':'k_means',
                                                'hyperparameters' : {'n_clusters':[3]} },
                    'DR_method' : { 'name': None, 'hyperparameters' : {} }, #'name': ['RFE'], 'n_features_to_select':[100]
                    }

DATASET_CONFIG = {'ROOT':ROOT_CONF, 'DATA_DIRS':DATA_DIRS,
                  'CUSTOM_STRATIFICATION_DICT':CUSTOM_STRATIFICATION_CONF, 'LABEL_DICT':LABEL_DICT_CONF, 'FEATURES_DICT':FEATURES_CONF,
                  'NB_FOLD':N_FOLD, 'N_TRAIN_MAX':N_TRAIN_MAX, 'N_TRAIN_MAX_PER_LABEL':N_TRAIN_MAX_PER_LABEL}

## define general config
CONFIG = {'xp_name':'clustering', 'dataset_config' : DATASET_CONFIG, 'ml_method_config' : ML_METHOD_CONFIG, 'metrics_config' : METRICS_CONFIG}

train_linear_model(**CONFIG)
