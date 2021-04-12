import socket, re, os

STRATIFICATION_CONFIGS = {

        'HYDRA_SKZ_VS_BIP': {
            'train': {'study': ['SCHIZCONNECT-VIP', 'BIOBD'],
                      'diagnosis': ['control', 'schizophrenia', 'bipolar disorder']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'schizophrenia', 'psychotic bipolar disorder']}
            },

        'HYDRA_UHR_C_vs_0.617±0.0NC': {
            'train': {'study': ['SCHIZCONNECT-VIP', 'BIODB', 'BSNIP'],
                      'diagnosis': ['control', 'schizophrenia', 'FEP']},
            'test': {'study': ['ICAAR_EUGEI_START'],
                     'diagnosis': ['UHR-C', 'UHR-NC']}
            },

        'CLASSIF_UHR': {
            'train': {'study': ['ICAAR_EUGEI_START'],
                      'diagnosis': ['UHR-C', 'UHR-NC']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'schizophrenia']}
        },

        'HYDRA_UHR_C_vs_NC_multiclass': {
            'train': {'study': ['SCHIZCONNECT-VIP', 'ICAAR_EUGEI_START'],
                      'diagnosis': ['control', 'schizophrenia', 'FEP', 'UHR-C', 'UHR-NC']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'schizophrenia']}
        },

        'CLASSIF_SKZ_HC': {
            'train': {'study': ['SCHIZCONNECT-VIP', 'ICAAR_EUGEI_START'],
                      'diagnosis': ['control', 'schizophrenia']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'schizophrenia']}
            },

        'CLASSIF_BIP_HC': {
            'train': {'study': ['BIOBD'],
                      'diagnosis': ['control', 'bipolar disorder']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['control', 'psychotic bipolar disorder']}
        },

        'CLASSIF_SKZ_BIP': {
            'train': {'study': ['SCHIZCONNECT-VIP', 'BIOBD'],
                      'diagnosis': ['schizophrenia', 'bipolar disorder']},
            'test': {'study': ['BSNIP'],
                     'diagnosis': ['schizophrenia', 'psychotic bipolar disorder']}
            },
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