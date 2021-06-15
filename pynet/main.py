import argparse
from train import BaseTrainer

# root and data directory paths
ROOT_CONF = {'input_path':  "/tsi/clusterhome/rlouiset/psy", # "/home/robin/Desktop/rl264746/neurospin/psy",
             'metadata_path': "/tsi/clusterhome/rlouiset/psy"}

DATA_DIRS = {'input_path': [
    # ROOT_CONF['input_path']+'/'+'icaar-start_t1mri_mwp1_gs-raw_data64.npy',
    ROOT_CONF['input_path'] + '/' + 'bsnip1_cat12vbm_mwp1-gs.npy',
    # ROOT_CONF['input_path'] + '/' + 'schizconnect-vip-prague_cat12vbm_mwp1-gs.npy',
    ROOT_CONF['input_path'] + '/' + 'biobd_cat12vbm_mwp1-gs.npy'
],
    'metadata_path': [
        # ROOT_CONF['metadata_path']+'/'+'icaar-start_t1mri_mwp1_participants.csv',
        ROOT_CONF['metadata_path'] + '/' + 'bsnip1_cat12vbm_participants.csv',
        # ROOT_CONF['metadata_path'] + '/' + 'schizconnect-vip-prague_cat12vbm_participants.csv',
        ROOT_CONF['metadata_path'] + '/' + 'biobd_cat12vbm_participants.csv',
    ]
}


# define folds, labels cropping in order to equilibrate the dataset
N_FOLD = 5
N_TRAIN_MAX = 600
N_TRAIN_MAX_PER_LABEL = 300

CUSTOM_STRATIFICATION_CONF = {
    'train': {'study': ['BIOBD'],
              'diagnosis': ['control', 'bipolar disorder']},
    'test': {'study': ['BSNIP'],
             'diagnosis': ['control', 'psychotic bipolar disorder', 'psychotic bipolar']}
}

LABEL_DICT_CONF = {'bipolar disorder': 1, 'control': 0}

ML_METHOD_CONFIG = None
FEATURES_CONFIG = {'preproc': 'cat12', 'N_train_max': N_TRAIN_MAX, 'db': 'tiny_bip_kfolds', 'labels': ['diagnosis'],
                   'stratify_label': 'diagnosis', 'nb_folds': N_FOLD, 'metrics': ['accuracy'], 'nb_epochs': 50}

MODEL_CONFIG = {'pretrained_path': None, 'freeze_until_layer': None, 'dropout': 0, 'num_classes': 1, 'da': [],
                'net': 'densenet121', 'batch_size': 8, 'pin_mem': False, 'drop_last': True, 'cuda': True,
                'num_cpu_workers': 4}  # densenet121

TEST_CONFIG = {'outfile_name': None, 'checkpoint_dir': '/tsi/clusterhome/rlouiset/records/exp2/', 'nb_epochs_per_saving': 5, 'bayesian': False,
               'concrete_dropout': 0, 'add_input': False}
OPTIMIZER_CONFIG = {'gamma_scheduler': 0.2, 'step_size_scheduler': 10, 'lr': 0.0001, 'sampler': 'random',
                    'exp_name': 'BIP_vs_CONTROL', 'folds': None}
LOSS_CONFIG = {'loss': 'BCE', 'loss_param': None}

DATASET_CONFIG = {'ROOT': ROOT_CONF, 'DATA_DIRS': DATA_DIRS,
                  'CUSTOM_STRATIFICATION_DICT': CUSTOM_STRATIFICATION_CONF, 'LABEL_DICT': LABEL_DICT_CONF,
                  'FEATURES_DICT': FEATURES_CONFIG,
                  'NB_FOLD': N_FOLD, 'N_TRAIN_MAX': N_TRAIN_MAX, 'N_TRAIN_MAX_PER_LABEL': N_TRAIN_MAX_PER_LABEL}

args = {}
for CONFIG in [FEATURES_CONFIG, MODEL_CONFIG, TEST_CONFIG, OPTIMIZER_CONFIG, LOSS_CONFIG, DATA_DIRS]:
    args.update(CONFIG)

parser = argparse.ArgumentParser()
for key, value in args.items():
    parser.add_argument('-' + key, '--' + key, action="store", required=False, default=value)
args = parser.parse_args()

import torch
print(torch.cuda.device_count())
print(torch.cuda.is_available())

trainer = BaseTrainer(args)
trainer.run()
