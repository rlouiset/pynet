import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.utils import resample
import pandas as pd

def display_cluster_statistics(y_pred_clusters, y_ground_truth, mapping=[0,1,2]) :
    y_pred_clusters_mapped = np.array(mapping)[y_pred_clusters]
    unique_labels_pred = np.unique(y_pred_clusters_mapped)
    for label in unique_labels_pred :
        print('The proportion of the label ' + str(label) + ' is : ' + str(np.sum(y_pred_clusters_mapped==label)) + '/' + str(len(y_pred_clusters_mapped)))
    print('')

def find_right_cluster_labelling(y_pred_clusters, y_ground_truth) :
    best_mapping, best_bacc = 0, 0
    mappings = [[0, 1], [1, 0]]
    for mapping in mappings :
        y_pred_clusters_mapped = np.array(mapping)[y_pred_clusters]
        print(mapping)
        b_acc, acc =[], []
        print(balanced_accuracy_score(y_pred_clusters_mapped, y_ground_truth))
        for n in range(100) :
            bootstrap_y_pred_clusters_mapped, bootstrap_y_ground_truth = resample(y_pred_clusters_mapped, y_ground_truth)
            b_acc.append(balanced_accuracy_score(bootstrap_y_pred_clusters_mapped, bootstrap_y_ground_truth))
            acc.append(accuracy_score(bootstrap_y_pred_clusters_mapped, bootstrap_y_ground_truth))
        print('Balanced accuracy : ' + str(np.mean(b_acc)) + ' +/- ' + str(np.std(b_acc)))
        print('Accuracy : ' + str(np.mean(acc)) + ' +/- ' + str(np.std(acc)))
        print('')
        if balanced_accuracy_score(y_ground_truth, y_pred_clusters_mapped) > best_bacc :
            best_bacc = balanced_accuracy_score(y_ground_truth, y_pred_clusters_mapped)
            best_mapping = mapping
    return best_mapping

def save_test_clustering_prediction(ML_method, X_test, participant_ids, test_indices, mapping=[0, 1]) :
    # predict cluster labels
    y_pred_clusters = np.argmax(ML_method.predict_cluster_assignement(X_test), 1)
    y_pred_clusters_mapped = np.array(mapping)[y_pred_clusters]
    # get participant ids
    participant_ids_test = participant_ids[test_indices]

    a = np.concatenate((participant_ids_test[:,0][:,None], y_pred_clusters_mapped[:,None]), axis=1)
    a_df = pd.DataFrame(a, columns=["participant_id", "cluster_pred"])
    a_df.to_csv("HYDRA_clustering.csv", index=False)
    print('Clustering prediction saved')


def print_importance_features_directions(HYDRA_method, features_names, X_test, y_test, mapping=[0,1,2]) :
    ctrl, skz, bip = X_test[y_test == 0], X_test[y_test == 1], X_test[y_test == 2]
    directions = HYDRA_method.coefficients[0]
    k = 10

    for cluster_i in range(len(directions)):
        print('LABEL ', mapping[cluster_i])
        indices = np.abs(directions[cluster_i][0]).argsort()[-k:][::-1]
        for idx in indices:
            print(features_names[idx] + ' : ' + str(directions[cluster_i][0][idx])[:6])
            #var_ctrl = np.sqrt(sum([(x - np.mean(ctrl[:, idx])) ** 2 for x in ctrl[:, idx]]) / (len(ctrl[:, idx])))
            #var_skz = np.sqrt(sum([(x - np.mean(skz[:, idx])) ** 2 for x in skz[:, idx]]) / (len(skz[:, idx])))
            #var_bip = np.sqrt(sum([(x - np.mean(bip[:, idx])) ** 2 for x in bip[:, idx]]) / (len(bip[:, idx])))
            #print('Means ctrl : ', str(np.mean(ctrl[:, idx]))[:5] + '+/-' + str(var_ctrl)[:6])
            #print('Means skz : ', str(np.mean(skz[:, idx]))[:5] + '+/-' + str(var_skz)[:6])
            #print('Means bip : ', str(np.mean(bip[:, idx]))[:5] + '+/-' + str(var_bip)[:6])
            #print('')
        print('')
        print('')

def tree_method_shap_plot() :
    return NotImplemented