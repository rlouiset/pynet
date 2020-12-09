from pynet.metrics.metrics import confusion_matrix, balanced_accuracy, AUC, accuracy_score
from pynet.metrics.clustering_metrics import *
import sklearn.preprocessing as sk_prep
import numpy as np
from statistics import variance
import shap
import matplotlib.pyplot as plt

class MetricManager(object):
    """ Data manager used to compute and display metrics
    """
    def __init__(self, metrics_config, xp_name, features_name):
        self.xp_name = xp_name
        self.metrics_computer = {phase : [Metric(**metric) for metric in metrics_config if phase in metric['phase']] for phase in ['TEST', 'VAL']}
        self.features_names = features_name

    def update_metrics(self, y, X_rep, ML_method, phase='VAL'):
        y_pred = ML_method.predict(X_rep)
        try :
            y_pred_proba = ML_method.predict_proba(X_rep)
        except :
            y_pred_proba = y_pred
        try :
            y_pred_cluster = ML_method.predict_cluster_assignement(X_rep)
        except :
            y_pred_cluster = y_pred

        for metric in self.metrics_computer[phase] :
            metric.update(y, y_pred, y_pred_proba, y_pred_cluster, X_rep)

    def display_metrics(self, phase_to_display=['VAL']):
        for phase in phase_to_display :
            print('Let us print our metrics for the phase : ', phase)
            for metric in self.metrics_computer[phase]:
                metric.display()
            print('')

class Metric(object) :
    """ Metric Object with update function, name, preprocessing functions...
    """
    def __init__(self, name, phase='VAL', labels=None):
        if labels is None:
            labels = {'control': 0, 'case': 1}
        self.labels = labels
        self.phase = phase
        self.name = name
        self.values = []

    def one_hot_encode(self, y):
        ''' utils function in order to turn a label vector into a one hot encoded matrix '''
        y_one_hot = np.copy(y)
        return np.eye(np.max(y_one_hot)+1)[y_one_hot]

    def renormalize(self, y_proba):
        ''' normalize probability vector so that each row sum is equal to one '''
        y_proba_norm = np.copy(y_proba)
        return y_proba_norm / np.sum(y_proba_norm, 1)[:, None]

    def select_labels(self, y, y_pred) :
        ''' From one-hot encoded y matrix label and probablity prediction matrix y_pred,
        return the selection over the labels specified in the metric definition (self.labels)'''
        y_select, y_pred_select = np.copy(y), np.copy(y_pred)

        select_index = np.array([i for i, y_i in enumerate(y_select) if np.argmax(y_i) in list(self.labels.values())])
        select_label = np.array(list(self.labels.values())).astype(np.int)

        y_select, y_pred_select= y_select[select_index], y_pred_select[select_index, :]
        y_select, y_pred_select = y_select[:, select_label], y_pred_select[:, select_label]

        return y_select, y_pred_select

    def update(self, y, y_pred, y_pred_proba, y_pred_cluster, X_rep):
        ''' search the right metric and compute it with respect to its Hyperparameters '''

        ## the following metrics are for unsupervised methods only
        if self.name == 'SilhouetteScore' :
            self.values.append(silhouette_score(X_rep, y_pred_cluster))
        if self.name == 'DaviesBouldin' :
            self.values.append(davies_bouldin(X_rep, y_pred_cluster))

        ## the following metrics are for hybrid methods
        if self.name == 'V_measure' :
            self.values.append(V_measure(y, y_pred_cluster))
        if self.name == 'ARI' :
            self.values.append(ARI(y, y_pred_cluster))
        if self.name == 'NMI' :
            self.values.append(NMI(y, y_pred_cluster))

        ## the following metrics for supervised methods only and are also usable in multiclass setting
        if self.name == 'multi_class_confusion_matrix' :
            y_select, y_pred_proba_select = self.select_labels(self.one_hot_encode(y), y_pred_proba)
            self.values.append(confusion_matrix(np.argmax(y_select,1), np.argmax(y_pred_proba_select,1), list(self.labels.values())))
        if self.name[:3] == 'acc':
            y_select, y_pred_proba_select = self.select_labels(self.one_hot_encode(y), y_pred_proba)
            self.values.append(balanced_accuracy(np.argmax(y_select,1), np.argmax(y_pred_proba_select,1)))
        if self.name[:4] == 'bacc':
            y_select, y_pred_proba_select = self.select_labels(self.one_hot_encode(y), y_pred_proba)
            self.values.append(accuracy_score(np.argmax(y_select,1), np.argmax(y_pred_proba_select,1)))
        if self.name[:3] == 'AUC' :
            y_select, y_pred_proba_select = self.select_labels(self.one_hot_encode(y), y_pred_proba)
            if len(self.labels)==2 :
                self.values.append(AUC(y_select[:,1], y_pred_proba_select[:,1]))
            else :
                self.values.append(AUC(y_select, y_pred_proba_select, multi_class='ovo'))

    def display(self):
        values = np.array(self.values)
        if self.name == 'multi_class_confusion_matrix':
            values_mean = np.mean(np.array(values), 0)
            values_var = np.var(np.array(values), 0)
            print('Multi-class confusion matrix : ')
            for i in range(len(values_mean)) :
                line_to_print = ''
                for j in range(len(values_mean)) :
                    line_to_print += (str(values_mean[i][j])[:5] + '+/-' + str(values_var[i][j])[:5] + '   ')
                print(line_to_print)
        else :
            confidence_intervall_bound = np.sqrt(sum([(x-np.mean(values))**2 for x in values]) / (len(values)))
            print(self.name + ' : ' + str(np.mean(values))[:5] + '+/-' + str(confidence_intervall_bound))