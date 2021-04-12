from pyML.utils.explainabilty import *
import json
import matplotlib.pyplot as plt

# root and data directory paths
ROOT_DATA = "/home/robin/Desktop/rl264746/neurospin/psy/"
ROOT_PREDICTIONS = "/home/robin/Desktop/rl264746/pynet/records/"

PHENOTYPE_FILES = ["bsnip1_cat12vbm_participants.csv",
                   "schizconnect-vip-prague_cat12vbm_participants.csv",
                   "start-icaar-eugei_cat12vbm_participants.csv"]

PREDICTIONS_FILE = "CLASSIF_SKZ_HC/CLASSIF_SKZ_HC_1/"

PHENOTYPE_FEATURES = ['sex', 'age', 'percentage', 'GM_Vol_', 'WM_Vol_', 'CSF_Vol_', 'ymrstot', 'MADRS']
                      # 'PANSS_total', 'PANSS_positive', 'PANSS_negative', 'PANSS_psychopatho', 'PANSS_desorganisation', 'SANS', 'SAPS', 'MADRS', 'SOFAS', 'NSS', 'BPRS']

# load predictions dictionary
with open(ROOT_PREDICTIONS + PREDICTIONS_FILE + "predictions_file.json") as f:
    predictions_dict = json.load(f)

# load phenotype data
all_df = [pd.read_csv(ROOT_DATA + p, sep=',') for p in PHENOTYPE_FILES]
phenotype_df = pd.concat(all_df, ignore_index=True, sort=False)

clusters = [0, 1]
labels = ['control', 'schizophrenia']
label_mapping = {'control': 0, 'schizophrenia': 1}
label_to_cluster = 1


# important features
important_features = []
for fold in predictions_dict.keys():
    important_features.extend(predictions_dict[fold]['important_features'][str(label_to_cluster)])
important_features = list(np.unique(important_features))

# aggregate results for all fold
for phase in ['TEST']:
    cluster_phenotype_stats = {metadata: {cluster: {'mean': [], 'std': []} for cluster in clusters} for metadata in
                               PHENOTYPE_FEATURES}
    label_phenotype_stats = {metadata: {label: {'mean': [], 'std': []} for label in labels} for metadata in
                             PHENOTYPE_FEATURES}
    cluster_important_features_stats = {features: {cluster: {'mean': [], 'std': []} for cluster in clusters} for
                                        features in important_features}
    label_important_features_stats = {features: {label: {'mean': [], 'std': []} for label in labels} for features in
                                      important_features}
    for fold in predictions_dict.keys():
        metadata_features_df_phase = pd.read_json(predictions_dict[fold]['dict_of_metadata'][phase])
        participant_ids = np.array(metadata_features_df_phase['participant_id'])

        phenotype_df_phase = metadata_features_df_phase.merge(phenotype_df, on='participant_id',
                                                              how='inner',
                                                              suffixes=[None, '_'])

        diagnosis = np.array(metadata_features_df_phase['diagnosis'])
        y = np.array([label_mapping[d] for d in diagnosis])

        classification_predictions_phase = predictions_dict[fold]['classification_predictions'][phase]
        clustering_predictions_phase = predictions_dict[fold]['clustering_predictions'][phase]

        # project on label to cluster only
        label_indexes = np.where(y == label_to_cluster)
        label_diagnosis = diagnosis[label_indexes]


        for cluster in clusters:
            cluster_predictions_phase_cluster = np.argmax(np.array(clustering_predictions_phase[str(label_to_cluster)]), 1)
            cluster_predictions_phase_cluster = cluster_predictions_phase_cluster[label_indexes]

            indexes = np.where(cluster_predictions_phase_cluster == cluster)
            phenotype_df_phase_indexed = phenotype_df_phase.iloc[label_indexes].iloc[indexes]

            if len(phenotype_df_phase_indexed) / len(cluster_predictions_phase_cluster) in [0, 1] :
                continue

            mean_MADRS = np.mean(phenotype_df_phase.iloc[label_indexes]['MADRS'])
            print(str(mean_MADRS))
            print(str(np.mean(phenotype_df_phase_indexed['MADRS'])))
            print('')
            if np.mean(phenotype_df_phase_indexed['MADRS']) > mean_MADRS :
                mapped_cluster = 0
            else :
                mapped_cluster = 1
            """diagnosis_cluster = np.array(phenotype_df_phase_indexed['diagnosis'])
            if np.sum(diagnosis_cluster=="UHR-NC") / np.sum(np.array(phenotype_df_phase['diagnosis'])=="UHR-NC") > 0.5 :
                mapped_cluster = 0
            else :
                mapped_cluster = 1"""

            for metadata in PHENOTYPE_FEATURES:
                if metadata == 'percentage' :
                    percentage = len(phenotype_df_phase_indexed)/len(cluster_predictions_phase_cluster)
                    cluster_phenotype_stats[metadata][mapped_cluster]['mean'].append(percentage)
                else :
                    cluster_phenotype_stats[metadata][mapped_cluster]['mean'].append(np.mean(phenotype_df_phase_indexed[metadata]))
            for feature in important_features:
                cluster_important_features_stats[feature][mapped_cluster]['mean'].append(
                    np.mean(phenotype_df_phase_indexed[feature]))
                cluster_important_features_stats[feature][mapped_cluster]['std'].append(
                    np.std(phenotype_df_phase_indexed[feature]))

        print('-------------')

        for label in labels:
            indexes = np.where(diagnosis == label)
            phenotype_df_phase_indexed = phenotype_df_phase.iloc[indexes]
            for metadata in PHENOTYPE_FEATURES:
                if metadata == 'percentage':
                    percentage = len(phenotype_df_phase_indexed) / len(label_diagnosis)
                    label_phenotype_stats[metadata][label]['mean'].append(percentage)
                else:
                    label_phenotype_stats[metadata][label]['mean'].append(np.nanmean(phenotype_df_phase_indexed[metadata]))
            for feature in important_features:
                label_important_features_stats[feature][label]['mean'].append(
                    np.nanmean(phenotype_df_phase_indexed[feature]))
                label_important_features_stats[feature][label]['std'].append(
                    np.nanstd(phenotype_df_phase_indexed[feature]))

    #print(debug)
    # print final scores for clustering
    print('Clustering statistics : ')
    print('Phenotype')
    for metadata in PHENOTYPE_FEATURES:
        print('Metadata : ', metadata)
        for cluster in clusters:
            mean_values = cluster_phenotype_stats[metadata][cluster]['mean']
            to_print = ('Cluster ' + str(cluster) + ' : ' + str(np.nanmean(mean_values))[:5] + '+/-' + str(
                np.nanstd(mean_values))[:5])
            print(to_print)
        print('')

    print('')
    print('Important features')
    for feature in important_features:
        print('Feature : ', feature)
        for cluster in clusters:
            mean_values = cluster_important_features_stats[feature][cluster]['mean']
            std_values = cluster_important_features_stats[feature][cluster]['std']
            to_print = ('Cluster ' + str(cluster) + ' : ' + str(np.nanmean(mean_values))[:5] + '+/-' + str(
                np.nanmean(std_values))[:5])
            print(to_print)
        print('')

    print('---------------------------------------------------------')
    print('')

    # print final scores for ground-truth labels
    print('Ground-truth Label statistics : ')
    print('Phenotype')
    for metadata in PHENOTYPE_FEATURES:
        print('Metadata : ', metadata)
        for label in labels:
            mean_values = label_phenotype_stats[metadata][label]['mean']
            to_print = ('Label ' + label + ' : ' + str(np.nanmean(mean_values))[:5] + '+/-' + str(np.nanstd(mean_values))[:5])
            print(to_print)
        print('')
    print('')

    print('')
    print('Important features')
    for feature in important_features:
        print('Feature : ', feature)
        for label in labels:
            mean_values = label_important_features_stats[feature][label]['mean']
            std_values = label_important_features_stats[feature][label]['std']
            to_print = ('Label ' + str(label) + ' : ' + str(np.nanmean(mean_values))[:5] + '+/-' + str(
                np.nanmean(std_values))[:5])
            print(to_print)
        print('')
