from pynet.metrics.metrics import confusion_matrix, balanced_accuracy, AUC, accuracy_score
from pynet.metrics.clustering_metrics import *
import numpy as np
import logging

def one_hot_encode(y, n_classes=None):
    ''' utils function in order to turn a label vector into a one hot encoded matrix '''
    if n_classes is None:
        n_classes = np.max(y) + 1
    y_one_hot = np.copy(y)
    return np.eye(n_classes)[y_one_hot]

class MetricManager(object):
    """ Data manager used to compute and display metrics
    """

    def __init__(self, metrics_config, xp_name, features_name):

        self.xp_name = xp_name
        self.metrics_config = metrics_config

        # Define both set of metrics : CLUSTERING and CLASSIFICATION
        if 'CLASSIFICATION' in metrics_config.keys():
            self.classification_metrics = {phase: [Metric(metric, phase, **metrics_config['CLASSIFICATION']['hp'])
                                                   for metric in metrics_config['CLASSIFICATION']['metrics']
                                                   if phase in metrics_config['CLASSIFICATION']['phase']]
                                           for phase in ['TEST', 'VAL']}

        if 'CLUSTERING' in metrics_config.keys():
            self.clustering_metrics = {phase: [Metric(metric, phase, **metrics_config['CLUSTERING']['hp'])
                                               for metric in metrics_config['CLUSTERING']['metrics']
                                               if phase in metrics_config['CLUSTERING']['phase']]
                                       for phase in ['TEST', 'VAL']}

        self.features_names = features_name

    def update_metrics(self, y, X_rep, ML_method, phase='VAL'):
        if 'CLASSIFICATION' in self.metrics_config.keys():
            try :
                y_pred_proba = ML_method.predict_proba(X_rep)
            except :
                y_pred = ML_method.predict(X_rep).astype(np.int)
                y_pred_proba = np.zeros((len(y_pred), 2))
                for i, y_i in enumerate(y_pred) :
                    y_pred_proba[i, y_i] = 1
            for metric in self.classification_metrics[phase]:
                metric.update(y, y_pred_proba, X_rep)

        if 'CLUSTERING' in self.metrics_config.keys():
            try :
                y_cluster_pred = ML_method.predict_clusters(X_rep)
            except :
                y_cluster_pred = one_hot_encode(ML_method.predict(X_rep))
            for metric in self.clustering_metrics[phase]:
                metric.update(y, y_cluster_pred, X_rep)

    def display_metrics(self, phase_to_display=None):
        if 'CLASSIFICATION' in self.metrics_config.keys():
            logging.info('Let us print the CLASSIFICATION metrics : ')
            for phase in phase_to_display:
                logging.info(phase)
                for metric in self.classification_metrics[phase]:
                    metric.display()
        logging.info('----------------------------------------------------------')
        if 'CLUSTERING' in self.metrics_config.keys():
            logging.info('Let us print the CLUSTERING metrics : ')
            for phase in phase_to_display:
                logging.info(phase)
                for metric in self.clustering_metrics[phase]:
                    metric.display()


class Metric(object):
    """ Metric Object with update function, name, preprocessing functions...
    """

    def __init__(self, name, phase='VAL', testing_label_mapping=None, label_to_cluster=None):
        self.testing_label_mapping = testing_label_mapping
        self.label_to_cluster = label_to_cluster
        self.phase = phase
        self.name = name
        self.values = []

    def select_labels_mapping(self, y, y_pred):
        """ From one-hot encoded y matrix label and probability prediction matrix y_pred,
        return the selection over the labels specified in the metric definition (self.labels)"""
        if self.label_to_cluster is not None:
            y_pred_mapped = np.array([y_pred[self.label_to_cluster][i] for i in range(len(y)) if y[i] in self.testing_label_mapping.keys()])
        else:
            y_pred_mapped = np.array([y_pred[i] for i in range(len(y)) if y[i] in self.testing_label_mapping.keys()])
        y_mapped = np.array([self.testing_label_mapping[y_i] for y_i in y if y_i in self.testing_label_mapping.keys()])
        return y_mapped, y_pred_mapped

    def update(self, y, y_pred, X_rep):
        ''' search the right metric and compute it with respect to its Hyperparameters '''

        # the following metrics are for unsupervised methods only
        if self.name == 'SilhouetteScore':
            self.values.append(silhouette_score(X_rep, y_pred))
        if self.name == 'DaviesBouldin':
            self.values.append(davies_bouldin(X_rep, y_pred))
        if self.name == 'ClusteringStability':
            _, y_pred_cluster_select = self.select_labels_mapping(y, y_pred)
            self.values.append(y_pred_cluster_select)

        # the following metrics are for hybrid methods
        if self.name == 'V_measure':
            y_select, y_pred_cluster_select = self.select_labels_mapping(y, y_pred)
            self.values.append(V_measure(y_select, np.argmax(y_pred_cluster_select, 1)))
        if self.name == 'ARI':
            y_select, y_pred_cluster_select = self.select_labels_mapping(y, y_pred)
            self.values.append(ARI(y_select, np.argmax(y_pred_cluster_select, 1)))
        if self.name == 'ClusteringBACC':
            y_select, y_pred_cluster_select = self.select_labels_mapping(y, y_pred)
            y_pred_cluster_select = np.argmax(y_pred_cluster_select, 1)
            clustering_accuracy = max(balanced_accuracy(y_select, y_pred_cluster_select),
                                      balanced_accuracy(1 - y_select, y_pred_cluster_select))
            self.values.append(clustering_accuracy)

        # the following metrics for supervised methods only
        if self.name == 'ACC':
            y_select, y_pred_proba_select = self.select_labels_mapping(y, y_pred)
            self.values.append(accuracy_score(y_select, np.argmax(y_pred_proba_select, 1)))
        if self.name == 'BACC':
            y_select, y_pred_proba_select = self.select_labels_mapping(y, y_pred)
            self.values.append(balanced_accuracy(y_select, np.argmax(y_pred_proba_select, 1)))
        if self.name == 'AUC':
            y_select, y_pred_proba_select = self.select_labels_mapping(y, y_pred)
            self.values.append(AUC(y_select, y_pred_proba_select[:,1]))

    def display(self):
        values = np.array(self.values)

        if self.name == 'ClusteringStability':
            ARI_list = []
            for fold_clustering_i in range(len(values)):
                for fold_clustering_j in range(fold_clustering_i + 1, len(values)):
                    ARI_list.append(
                        ARI(np.argmax(values[fold_clustering_i], 1), np.argmax(values[fold_clustering_j], 1)))
            confidence_interval_bound = np.sqrt(
                sum([(x - np.mean(ARI_list)) ** 2 for x in ARI_list]) / (len(ARI_list)))
            print(self.name + ' : ' + str(np.mean(ARI_list))[:5] + '+/-' + str(confidence_interval_bound))

        else:
            confidence_interval_bound = np.sqrt(sum([(x - np.mean(values)) ** 2 for x in values]) / (len(values)))
            to_print = (self.name + ' : ' + str(np.mean(values))[:5] + '+/-' + str(confidence_interval_bound))
            logging.info(to_print)
