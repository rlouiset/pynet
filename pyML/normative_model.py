import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

# root and data directory paths
ROOT_DATA = "/home/robin/Desktop/rl264746/neurospin/psy/"
ROOT_PREDICTIONS = "/home/robin/Desktop/rl264746/pynet/records/"


PHENOTYPE_FILES = ["biobd_cat12vbm_participants.csv",
                   "bsnip1_cat12vbm_participants.csv",
                   "schizconnect-vip-prague_cat12vbm_participants.csv",
                   "start-icaar-eugei_cat12vbm_participants.csv"]

FEATURES_FILES = ["biobd_cat12vbm_rois-gs.csv",
                  "bsnip1_cat12vbm_rois-gs.csv",
                  "schizconnect-vip-prague_cat12vbm_rois-gs.csv",
                  "start-icaar-eugei_cat12vbm_rois-gs.csv"]

PREDICTIONS_FILE = "HYDRA_UHR_C_vs_NC_multiclass/HYDRA_UHR_C_vs_NC_multiclass_2/"

# load phenotype and features data
all_features_df = [pd.read_csv(ROOT_DATA + p, sep=',') for p in FEATURES_FILES]
features_df = pd.concat(all_features_df, ignore_index=True, sort=False)
all_phenotype_df = [pd.read_csv(ROOT_DATA + p, sep=',') for p in PHENOTYPE_FILES]
phenotype_df = pd.concat(all_phenotype_df, ignore_index=True, sort=False)

features_df = features_df.merge(phenotype_df, on='participant_id', how='inner', suffixes=[None, '_'])

diagnosis = {'train': ['schizophrenia', 'control'],
             'test': ['UHR-C', 'UHR-NC']}

study = {'train': ['SCHIZCONNECT-VIP'],
        'test': ['ICAAR_EUGEI_START']}

biological_markers = ['rCbeLoCbe6-7_GM_Vol'] #
# 'lInfFroAngGy_GM_Vol', 'rCbeLoCbe6-7_GM_Vol', 'rSupTemGy_GM_Vol', 'lInfFroGy_GM_Vol' 'rMedPoCGy_GM_Vol', 'lInfFroAngGy_GM_Vol'

# first build the training statistics
studies = list(features_df['study'])
train_sites_indexes = np.array([i for (i, s) in enumerate(studies) if s in study['train']])
train_features_df = features_df.iloc[train_sites_indexes]

diagnosises = list(train_features_df['diagnosis'])

# get train mean and std
train_control_indexes = np.array([i for i, d in enumerate(diagnosises) if d=='control'])
control_features_df = train_features_df.iloc[train_control_indexes]
control_mean_per_biomarker = {biomarker:np.mean(control_features_df[biomarker]) for biomarker in biological_markers}
control_std_per_biomarker = {biomarker:np.std(control_features_df[biomarker]) for biomarker in biological_markers}

# get train features
train_sites_indexes = np.array([i for i, d in enumerate(diagnosises) if d in diagnosis['train']])
train_features_df = train_features_df.iloc[train_sites_indexes]


# list all ages
train_ages = np.rint(train_features_df['age'])
all_train_ages = np.unique(train_ages)
train_diagnosises = np.array(train_features_df['diagnosis'])

train_statistics = {d: {'mean': [0] * len(all_train_ages), 'std': [0] * len(all_train_ages)} for d in
                    diagnosis['train']}

for d in diagnosis['train']:
    for i, age in enumerate(all_train_ages):
        diagnosis_indexes = np.where(train_diagnosises == d)
        train_features_df_indexed = train_features_df.iloc[diagnosis_indexes]
        age_indexes = np.where(train_features_df_indexed['age'] == age)
        train_features_df_indexed = train_features_df_indexed.iloc[age_indexes]

        train_statistics[d]['mean'][i] = np.mean([ (np.mean(train_features_df_indexed[biomarker])-control_mean_per_biomarker[biomarker])/control_std_per_biomarker[biomarker]
                                                  for biomarker in biological_markers])
        train_statistics[d]['std'][i] = np.mean([(np.std(train_features_df_indexed[biomarker])-control_std_per_biomarker[biomarker])/control_std_per_biomarker[biomarker]
                                                 for biomarker in biological_markers])




# first build the training statistics
studies = list(features_df['study'])
test_sites_indexes = np.array([i for (i, s) in enumerate(studies) if s in study['test']])
test_features_df = features_df.iloc[test_sites_indexes]

diagnosises = list(test_features_df['diagnosis'])
test_d_indexes = np.array([i for i, d in enumerate(diagnosises) if d in diagnosis['test']])
test_features_df = test_features_df.iloc[test_d_indexes]

# list all ages
test_ages = np.rint(test_features_df['age'])
test_statistics = {d: {'age': [], 'Z': []} for d in diagnosis['test']}
test_diagnosises = np.array(test_features_df['diagnosis'])

for d in diagnosis['test']:
    diagnosis_indexes = np.where(test_diagnosises == d)
    test_features_df_indexed = test_features_df.iloc[diagnosis_indexes]
    for i in range(0, len(test_features_df_indexed)):
        test_statistics[d]['age'].extend(list(test_features_df_indexed['age']))
        bio_markers = np.sum([(test_features_df_indexed[biomarker]-control_mean_per_biomarker[biomarker])/control_std_per_biomarker[biomarker]
                              for biomarker in biological_markers], 0)
        test_statistics[d]['Z'].append(list(bio_markers))




# plot
std_factors_list = [1, 2]
fig, ax = plt.subplots()
# plot the training statistics
for d in diagnosis['train']:
    # corrrect no-samples points and apply moving average
    for stat in ['mean', 'std'] :
        train_statistics[d][stat] = np.nan_to_num(train_statistics[d][stat])
        no_nan_stats = train_statistics[d][stat].copy()
        mean_t_stats = np.nanmedian(no_nan_stats[no_nan_stats>0])
        for i, point in enumerate(train_statistics[d][stat]) :
            if train_statistics[d][stat][i] == 0 :
                train_statistics[d][stat][i] = mean_t_stats
        train_statistics[d][stat] = gaussian_filter1d(train_statistics[d][stat], 1)

    # plot training curve
    ax.plot(all_train_ages, train_statistics[d]['mean'], label=d)
    if d == 'control':
        for std_factor in std_factors_list:
            ci = np.array(train_statistics[d]['std']).astype(np.float) * std_factor
            ax.fill_between(all_train_ages, (train_statistics[d]['mean'] - ci),
                            (train_statistics[d]['mean'] + ci), color='r', alpha=.1)

for d in diagnosis['test']:
    ax.scatter(test_statistics[d]['age'], test_statistics[d]['Z'], label=d, s=5, alpha=0.8)

plt.legend()
plt.savefig('normative_model_test.png')
