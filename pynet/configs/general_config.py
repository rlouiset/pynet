import socket, re, os

CONFIG = {
    'db': {
        'healthy': {
            'train': {'study': ['HCP', 'IXI'], 'diagnosis': 'control'},
            'validation': {'study': 'BIOBD', 'diagnosis': 'control'},
            'test': {'study': ['BSNIP'], 'diagnosis': ['control']}
            },

        'big_healthy': {
            'train': {'study': ['OASIS3', 'CoRR', 'HCP', 'ABIDE1', 'GSP', 'RBP', 'ABIDE2', 'IXI', 'LOCALIZER',
                                'MPI-LEIPZIG', 'ICBM', 'NPC', 'NAR'], 'diagnosis': 'control'},
            'validation': {'study': 'BIOBD', 'diagnosis': 'control'},
            'test': {'study': 'BSNIP', 'diagnosis': 'control'}
            },

        'clustering': {
            'train': {'study': ['SCHIZCONNECT-VIP', 'BIOBD'],
                      'diagnosis': ['control', 'schizophrenia', 'bipolar disorder']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'schizophrenia', 'schizoaffective disorder', 'psychotic bipolar disorder',
                                   'relative of proband with schizophrenia',
                                   'relative of proband with schizoaffective disorder',
                                   'relative of proband with psychotic bipolar disorder']}
            },

        'tiny_scz_kfolds': {
            'train': {'study': ['SCHIZCONNECT-VIP'],
                      'diagnosis': ['control', 'schizophrenia']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'schizophrenia']}
            },

        'tiny_bip_kfolds': {
            'train': {'study': ['BIOBD'],
                      'diagnosis': ['control', 'bipolar disorder']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'psychotic bipolar disorder']}
            },

    },

    'optimizer': {
        'Adam': {'weight_decay': 5e-5}
    },
    'scheduler': {
        'StepLR': {} # By default step_size = 10
    },
    'cat12': {},
    'quasi_raw': {}
}
ALL_DATASETS = [('hcp_t1mri_mwp1_gs-raw_data64.npy', 'hcp_t1mri_mwp1_participants.csv'),
                ('ixi_t1mri_mwp1_gs-raw_data64.npy', 'ixi_t1mri_mwp1_participants.csv'),
                ('npc_t1mri_mwp1_gs-raw_data64.npy', 'npc_t1mri_mwp1_participants.csv'),
                ('nar_t1mri_mwp1_gs-raw_data64.npy', 'nar_t1mri_mwp1_participants.csv'),
                ('rbp_t1mri_mwp1_gs-raw_data64.npy', 'rbp_t1mri_mwp1_participants.csv'),
                ('oasis3_t1mri_mwp1_gs-raw_data64.npy', 'oasis3_t1mri_mwp1_participants.csv'),
                ('gsp_t1mri_mwp1_gs-raw_data64.npy', 'gsp_t1mri_mwp1_participants.csv'),
                ('icbm_t1mri_mwp1_gs-raw_data64.npy', 'icbm_t1mri_mwp1_participants.csv'),
                ('abide1_t1mri_mwp1_gs-raw_data64.npy', 'abide1_t1mri_mwp1_participants.csv'),
                ('abide2_t1mri_mwp1_gs-raw_data64.npy', 'abide2_t1mri_mwp1_participants.csv'),
                ('localizer_t1mri_mwp1_gs-raw_data64.npy', 'localizer_t1mri_mwp1_participants.csv'),
                ('mpi-leipzig_t1mri_mwp1_gs-raw_data64.npy', 'mpi-leipzig_t1mri_mwp1_participants.csv'),
                ('corr_t1mri_mwp1_gs-raw_data64.npy', 'corr_t1mri_mwp1_participants.csv'),
                ## Datasets with scz
                ('candi_t1mri_mwp1_gs-raw_data64.npy', 'candi_t1mri_mwp1_participants.csv'),
                ('cnp_t1mri_mwp1_gs-raw_data64.npy', 'cnp_t1mri_mwp1_participants.csv'),
                ('biobd_t1mri_mwp1_gs-raw_data64.npy', 'biobd_t1mri_mwp1_participants.csv'),
                ('bsnip_t1mri_mwp1_gs-raw_data64.npy', 'bsnip_t1mri_mwp1_participants.csv'),
                ('schizconnect-vip_t1mri_mwp1_gs-raw_data64.npy', 'schizconnect-vip_t1mri_mwp1_participants.csv'),
                ]