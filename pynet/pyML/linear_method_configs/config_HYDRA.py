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
                 #'physiologic_features': ['surface_area', 'volume', 'thickness'],
                 'roi': [],
                 }

LABEL_DICT_CONF = {'relative of proband with psychotic bipolar disorder':6, 'relative of proband with schizoaffective disorder':5, 'relative of proband with schizophrenia':4,
                    'schizoaffective disorder':3, 'psychotic bipolar disorder':1, 'bipolar disorder':1, 'FEP':2, 'schizophrenia':2, 'control':0}

CUSTOM_STRATIFICATION_CONF = {
            'train': {'study': ['BIOBD'],  # 'SCHIZCONNECT-VIP'
                      'diagnosis': ['control', 'bipolar disorder']}, # 'schizophrenia'
            'test': {'study': ['BSNIP'],
                     'diagnosis': [ 'control', 'psychotic bipolar disorder', # 'schizophrenia', 'schizoaffective disorder',
                                   #'relative of proband with schizophrenia',
                                   #'relative of proband with schizoaffective disorder',
                                   #'relative of proband with psychotic bipolar disorder'
                                  ]}
            }


## define metrics, ml_method and dataset config
METRICS_CONFIG = [
                #{'name': 'ARI', 'phase': ['VAL', 'TEST']},
                {'name': 'AUC', 'phase':['VAL', 'TEST'], 'labels': {'control':0, 'scz':1}},
                {'name': 'bacc', 'phase':['VAL', 'TEST'], 'labels': {'control':0, 'scz':1}},
                {'name': 'acc', 'phase':['VAL', 'TEST']},
                #{'name': 'SilhouetteScore', 'phase':['VAL', 'TEST']},
                #{'name': 'DaviesBouldin', 'phase':['VAL', 'TEST']},
                #{'name': 'V_measure', 'phase':['VAL', 'TEST']},
                ]

ML_METHOD_CONFIG = {
                    'DR_method' : { 'name': None, 'hyperparameters' : {} },
                    'ML_method' : { 'name':'HYDRA',
                                                'hyperparameters' : {'C':[1], 'n_consensus':[1], 'n_iterations':[1],
                                                'n_clusters_per_label':[{0:1, 1:1}], 'labels':[{0:'control', 1:'bip'}]}},
                    }

DATASET_CONFIG = {'ROOT':ROOT_CONF, 'DATA_DIRS':DATA_DIRS,
                  'CUSTOM_STRATIFICATION_DICT':CUSTOM_STRATIFICATION_CONF, 'LABEL_DICT':LABEL_DICT_CONF, 'FEATURES_DICT':FEATURES_CONF,
                  'NB_FOLD':N_FOLD, 'N_TRAIN_MAX':N_TRAIN_MAX, 'N_TRAIN_MAX_PER_LABEL':N_TRAIN_MAX_PER_LABEL}

## define general config
CONFIG = {'xp_name':'', 'dataset_config' : DATASET_CONFIG, 'ml_method_config' : ML_METHOD_CONFIG, 'metrics_config' : METRICS_CONFIG}

train_linear_model(**CONFIG)

