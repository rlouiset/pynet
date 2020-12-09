from pynet.pyML.main import train_linear_model

## root and data directory paths
ROOT_CONF = "/home/robin/Desktop/rl264746/neurospin/psy_sbox"
DATA_DIRS = {'input_path': '',
             'metadata_path': '/all_t1mri_mwp1_participants_merged_features.tsv'}

## define folds, labels cropping in order to equilibrate the dataset
N_FOLD = 10
N_TRAIN_MAX = 1250
N_TRAIN_MAX_PER_LABEL = 500


## features and labels config
FEATURES_CONF = {'general_features': ['wm', 'wmh', 'gm', 'csf'], # ,'sex', 'age', 'tiv'
                 'physiologic_features': ['surface_area', 'volume', 'thickness'],
                 'roi': [],
                 }

LABEL_DICT_CONF = {'relative of proband with psychotic bipolar disorder':6, 'relative of proband with schizoaffective disorder':5, 'relative of proband with schizophrenia':4,
                    'schizoaffective disorder':3, 'psychotic bipolar disorder':2, 'bipolar disorder':2, 'FEP':1, 'schizophrenia':1, 'control':0}

CUSTOM_STRATIFICATION_CONF = {
            'train': {'study': ['SCHIZCONNECT-VIP', 'BIOBD'],
                      'diagnosis': ['control', 'schizophrenia', 'bipolar disorder']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'schizophrenia', 'schizoaffective disorder', 'psychotic bipolar disorder',
                                   'relative of proband with schizophrenia',
                                   'relative of proband with schizoaffective disorder',
                                   'relative of proband with psychotic bipolar disorder']}
            }

## define metrics, ml_method and dataset config
METRICS_CONFIG = [
                #{'name': 'accuracy', 'phase':['VAL']},
                {'name': 'AUC', 'labels': {'control':0, 'schizophrenia':1, 'bipolar disorder':2}, 'phase':['VAL', 'TEST']},
                {'name': 'bacc', 'labels': {'control':0, 'schizophrenia':1, 'bipolar disorder':2}, 'phase': ['TEST']},
                {'name': 'AUC_scz_hc', 'labels': {'control':0, 'schizophrenia':1}, 'phase':['TEST']},
                {'name': 'AUC_bip_hc', 'labels': {'control':0, 'psychotic bipolar disorder':2}, 'phase':['TEST']},
                {'name': 'multi_class_confusion_matrix', 'labels': {'control':0, 'schizophrenia':1, 'bipolar disorder':2}, 'phase':'TEST'},
                ]

ML_METHOD_CONFIG = {
                    'DR_method': {'name': None,
                                                'hyperparameters': {'name': ['']}},
                    'ML_method' : { 'name':'SVM',
                                                'hyperparameters' : {'probability': [True]} },
                    }

DATASET_CONFIG = {'ROOT':ROOT_CONF, 'DATA_DIRS':DATA_DIRS,
                  'CUSTOM_STRATIFICATION_DICT':CUSTOM_STRATIFICATION_CONF, 'LABEL_DICT':LABEL_DICT_CONF, 'FEATURES_DICT':FEATURES_CONF,
                  'NB_FOLD':N_FOLD, 'N_TRAIN_MAX':N_TRAIN_MAX, 'N_TRAIN_MAX_PER_LABEL':N_TRAIN_MAX_PER_LABEL}

## define general config
CONFIG = {'xp_name':'clustering', 'dataset_config' : DATASET_CONFIG, 'ml_method_config' : ML_METHOD_CONFIG, 'metrics_config' : METRICS_CONFIG}

train_linear_model(**CONFIG)
