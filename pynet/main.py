import argparse
from train import BaseTrainer

## root and data directory paths
ROOT_CONF = {'input_path': "/home/robin/Desktop/rl264746/neurospin/psy_sbox/",
             'metadata_path': "/home/robin/Desktop/rl264746/neurospin/psy_sbox/"}

DATA_DIRS = {'input_path': [
                            ROOT_CONF['input_path']+'/'+'icaar-start_t1mri_mwp1_gs-raw_data64.npy',
                            ROOT_CONF['input_path']+'/'+'bsnip_t1mri_mwp1_gs-raw_data64.npy',
                            ROOT_CONF['input_path']+'/'+'schizconnect-vip_t1mri_mwp1_gs-raw_data64.npy',
                            ROOT_CONF['input_path']+'/'+'biobd_t1mri_mwp1_gs-raw_data64.npy'
                            ],
             'metadata_path': [
                            ROOT_CONF['metadata_path']+'/'+'icaar-start_t1mri_mwp1_participants.csv',
                            ROOT_CONF['metadata_path']+'/'+'bsnip_t1mri_mwp1_participants.csv',
                            ROOT_CONF['metadata_path']+'/'+'schizconnect-vip_t1mri_mwp1_participants.csv',
                            ROOT_CONF['metadata_path']+'/'+'biobd_t1mri_mwp1_participants.csv',
                            ]
             }

## define folds, labels cropping in order to equilibrate the dataset
N_FOLD = 10
N_TRAIN_MAX = 600
N_TRAIN_MAX_PER_LABEL = 250


CUSTOM_STRATIFICATION_CONF = {
            'train': {'study': ['SCHIZCONNECT-VIP'],
                      'diagnosis': ['control', 'schizophrenia']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'schizophrenia']}
            }

LABEL_DICT_CONF = {'schizophrenia':1, 'control':0}


## define metrics, ml_method and dataset config
METRICS_CONFIG = [
                {'name': 'AUC', 'phase':['VAL', 'TEST'], 'testing_label_mapping': {0:0, 1:1}, 'predict_testing_label_mapping' : False},
                {'name': 'bacc', 'phase':['VAL', 'TEST'], 'testing_label_mapping': {0:0, 1:1}, 'predict_testing_label_mapping' : False},
                {'name': 'acc', 'phase':['VAL', 'TEST'], 'testing_label_mapping': {0:0, 1:1}, 'predict_testing_label_mapping' : False},
                ]

ML_METHOD_CONFIG = None
FEATURES_CONFIG = {'preproc':'cat12', 'N_train_max':N_TRAIN_MAX, 'db':'tiny_scz_kfolds', 'labels':['diagnosis'],
                   'stratify_label':'age', 'nb_folds':N_FOLD, 'metrics':['accuracy'], 'nb_epochs':10}

MODEL_CONFIG = {'pretrained_path':None, 'freeze_until_layer':None, 'dropout':0, 'num_classes':2, 'da':['noise'],
                'net':'fc', 'batch_size':8, 'pin_mem':False, 'drop_last':True, 'cuda':False, 'num_cpu_workers':4}  # densenet121

TEST_CONFIG = {'outfile_name':None, 'checkpoint_dir':None, 'nb_epochs_per_saving':5, 'bayesian':False, 'concrete_dropout':0, 'add_input':False}
OPTIMIZER_CONFIG = {'gamma_scheduler':0.00001, 'step_size_scheduler':1, 'lr':0.0001, 'sampler':'random', 'exp_name':'First try', 'folds':None}
LOSS_CONFIG = {'loss':'L1', 'loss_param':None}

DATASET_CONFIG = {'ROOT':ROOT_CONF, 'DATA_DIRS':DATA_DIRS,
                  'CUSTOM_STRATIFICATION_DICT':CUSTOM_STRATIFICATION_CONF, 'LABEL_DICT':LABEL_DICT_CONF, 'FEATURES_DICT':FEATURES_CONFIG,
                  'NB_FOLD':N_FOLD, 'N_TRAIN_MAX':N_TRAIN_MAX, 'N_TRAIN_MAX_PER_LABEL':N_TRAIN_MAX_PER_LABEL}

xp_name = 'tiny_scz_kfolds'

## define general config
CONFIG = {'xp_name':'tiny_scz_kfolds', 'dataset_config' : DATASET_CONFIG, 'ml_method_config' : ML_METHOD_CONFIG, 'metrics_config' : METRICS_CONFIG}

args={}
for CONFIG in [FEATURES_CONFIG, MODEL_CONFIG, TEST_CONFIG, OPTIMIZER_CONFIG, LOSS_CONFIG, DATA_DIRS] :
    args.update(CONFIG)
args['xp_name'] = xp_name


parser = argparse.ArgumentParser()
for key, value in args.items() :
    parser.add_argument('-'+key, '--'+key, action="store", required=False, default=value)
args = parser.parse_args()

trainer = BaseTrainer(args)
trainer.run()



