import hdbscan
from sklearn.mixture import GaussianMixture
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from .utils import *


class BaseML(object):
    """ Base Machine Learning classifier / clustering method
    """
    def __init__(self, name="Base empty Machine Learning object"):
        self.name = name

    def fit(self, X_train, y_train):
        pass

    def fit_transform(self, X_train, y_train=None):
        return X_train

    def predict(self, X_val):
        return X_val

class HDBSCAN(BaseML) :
    """ Hierarchical DBSCAN algorithm, it has a predict function and a displaying of hierarchical clustering which is very interesting for us """
    def __init__(self, min_cluster_size=15, name="HDBSCAN"):
        super().__init__(name)
        self.HDBSCAN = hdbscan.HDBSCAN(min_cluster_size, prediction_data=True)

    def fit(self, X_train, y_train):
        self.HDBSCAN.fit(X_train)

    def fit_transform(self, X_train, y_train=None):
        self.HDBSCAN.fit(X_train)
        y_pred = self.HDBSCAN.labels_
        return y_pred

    def predict(self, X_val):
        return hdbscan.approximate_predict(self.HDBSCAN, X_val)[0]

    def score(self, X_val, y_val):
        y_pred_val = self.predict(X_val)
        return silhouette_score(X_val, y_pred_val)


class HYDRA(BaseML):
    """ Computes and stores the average and current value.
    """
    def __init__(self, C, n_clusters_per_label=None, labels=None, initialization_type="DPP", n_consensus=5,
                 n_iterations=5, tolerance=0.0001, name="HYDRA"):
        super().__init__(name)
        if n_clusters_per_label is None:
            n_clusters_per_label = {0: 1, 1: 1} # control patient have only one cluster, schizophrenic will be divided into 2 clusters by default
        if labels is None:
            labels = {0: 'control', 1: 'schizophrenia'}  # {0:'control', 1:'scz'}
        self.C = C
        self.n_consensus = n_consensus
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.initialization_type = initialization_type

        self.labels = list(labels.keys())
        self.n_clusters_per_label = n_clusters_per_label
        self.coefficients = {label:{cluster_i:None for cluster_i in range(n_clusters_per_label[label])} for label in self.labels}
        self.intercepts = {label:{cluster_i:None for cluster_i in range(n_clusters_per_label[label])} for label in self.labels}

    def fit(self, X_train, y_train):
        for label in self.labels :
            self.main(X_train, y_train, idx_outside_polytope=label)

    def predict(self, X):
        if len(self.labels) == 2 :
            y_pred = self.predict_binary_proba(X)[:,1]
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred < 0.5] = 0
        else :
            y_pred = self.predict_proba(X)
            y_pred = np.argmax(y_pred, 1)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def predict_binary_proba(self, X):
        SVM_scores_dict = {label: np.zeros((len(X), self.n_clusters_per_label[label])) for label in self.labels}
        y_pred = np.zeros((len(X), 2))
        for label in self.labels:
            ## fullfill the SVM score matrix
            for cluster_i in range(self.n_clusters_per_label[label]):
                SVM_coefficient, SVM_intercept = self.coefficients[label][cluster_i], self.intercepts[label][cluster_i]
                SVM_scores_dict[label][:, cluster_i] = (np.matmul(SVM_coefficient, X.transpose()) + SVM_intercept).transpose().squeeze()

        ## fullfill each cluster score
        for i in range(len(X)):
            y_pred[i][1] = sigmoid(np.max(SVM_scores_dict[1][i, :]) - np.max(SVM_scores_dict[0][i, :]))
            y_pred[i][0] = 1-y_pred[i][1]
        return y_pred

    def predict_proba(self, X):
        ''' '''
        ## predict cluster distance for each label
        if len(self.labels) == 2 :
            y_pred_proba = self.predict_binary_proba(X)
        else :
            cluster_predictions = self.predict_SVM_distances(X)
            y_pred_proba = np.zeros((len(X), len(cluster_predictions)))
            for sample in range(len(X)) :
                for label_i in self.labels :
                    y_pred_proba[sample][label_i] = cluster_predictions[label_i][sample, 0]
        return y_pred_proba

    def predict_SVM_distances(self, X) :
        ''' '''
        SVM_scores_dict = {label : np.zeros((len(X), self.n_clusters_per_label[label])) for label in self.labels}
        cluster_predictions = {label : np.zeros((len(X), self.n_clusters_per_label[label]+1)) for label in self.labels}
        for label in self.labels :
            ## fullfill the SVM score matrix
            for cluster_i in range(self.n_clusters_per_label[label]):
                SVM_coefficient, SVM_intercept = self.coefficients[label][cluster_i], self.intercepts[label][cluster_i]
                SVM_scores_dict[label][:, cluster_i] = (np.matmul(SVM_coefficient, X.transpose()) + SVM_intercept).transpose().squeeze()

            ## fullfill each cluster score
            for i in range(len(X)) :
                if np.max(SVM_scores_dict[label][i, :]) < 0:
                    cluster_predictions[label][i, 0] = sigmoid(np.mean(SVM_scores_dict[label][i, :])) # P(y=label)
                else :
                    for cluster_i in range(self.n_clusters_per_label[label]):
                        SVM_scores_dict[label][i, cluster_i] = max(0, SVM_scores_dict[label][i, cluster_i])
                    cluster_predictions[label][i, 0] = sigmoid(np.sum(SVM_scores_dict[label][i, :]))
                    for cluster_i in range(self.n_clusters_per_label[label]):
                        cluster_predictions[label][i, cluster_i+1] = SVM_scores_dict[label][i, cluster_i] / np.sum(SVM_scores_dict[label][i, :])     # P(cluster=i|y=label)
        return cluster_predictions

    def predict_cluster_assignement(self, X):
        y_pred = self.predict_proba(X)
        cluster_predictions = self.predict_SVM_distances(X)
        y_pred_cluster = np.zeros((len(X), np.sum([cluster_size_i for cluster_size_i in self.n_clusters_per_label.values()])))

        c = 0
        for label in self.labels :
            for cluster in range(self.n_clusters_per_label[label]) :
                for sample_i in range(len(X)) :
                    if y_pred[sample_i][label] > 0.5 :
                        if cluster_predictions[label][sample_i, cluster+1] == np.max(cluster_predictions[label][sample_i, 1:]) :
                            y_pred_cluster[sample_i, c] = 1
                c += 1

        return np.argmax(y_pred_cluster, 1)


    def main(self, X, y, idx_outside_polytope, epsilon=0.001):
        n_clusters = self.n_clusters_per_label[idx_outside_polytope]
        ## put the label idx_center_polytope at the center of the polytope by setting it to positive labels
        y_polytope = np.copy(y)
        y_polytope[y_polytope!=idx_outside_polytope] = -1    ## if label is inside of the polytope, the distance is negative and the label is not divided into
        y_polytope[y_polytope==idx_outside_polytope] = 1     ## if label is outside of the polytope, the distance is positive and the label is clustered

        censensus_assignment = np.zeros((len(y_polytope[y_polytope == 1]), self.n_consensus))  ## only consider the positive index
        cluster_index = censensus_assignment[0,:]

        index_positives = np.where(y_polytope == 1)[0]  # index for Positive Labels
        index_negatives = np.where(y_polytope == -1)[0]  # index for Negative Labels

        for consensus_i in range(self.n_consensus):
            weight_sample = np.ones((len(y_polytope), n_clusters)) / n_clusters
            ## depending on the weight initialization strategy, random hyperplanes were initialized with maximum diversity to constitute the convex polytope
            weight_positive_samples = self.init_weight(X, y_polytope, index_positives, index_negatives, n_clusters, initialization_type=self.initialization_type)
            weight_sample[index_positives] = weight_positive_samples  ## only replace the sample weight for positive samples

            ## cluster assignment is based on this svm scores across different SVM/hyperplanes
            svm_scores = np.zeros(weight_sample.shape)

            for iter in range(self.n_iterations):
                for cluster in range(n_clusters):
                    cluster_weight = np.ascontiguousarray(weight_sample[:, cluster])
                    SVM_coefficient, SVM_intercept = self.launch_svc(X, y_polytope, cluster_weight)
                    ## Apply the data again the trained model to get the final SVM scores
                    svm_scores[:, cluster] = (np.matmul(SVM_coefficient, X.transpose()) + SVM_intercept).transpose().squeeze()

                cluster_index = np.argmax(svm_scores[index_positives], axis=1)

                ## decide the convergence of the polytope based on the toleration
                weight_sample_hold = weight_sample.copy()
                # after each iteration, first set the weight of patient rows to be 0
                weight_sample[index_positives, :] = epsilon/n_clusters
                # then set the positives samples weight to be 1 for the assigned hyperplane
                for n in range(len(index_positives)):
                    weight_sample[index_positives[n], cluster_index[n]] = 1-epsilon

                ## check the loss comparted to the tolorence for stopping criteria
                loss = np.linalg.norm(np.subtract(weight_sample, weight_sample_hold), ord='fro')
                if loss < self.tolerance:
                    break
            ## update the cluster index for the consensus clustering
            censensus_assignment[:, consensus_i] = cluster_index + 1

        ## do censensus clustering
        final_predict = consensus_clustering(censensus_assignment.astype(int), n_clusters)
        ## after deciding the final convex polytope, we refit the training data once to save the best model
        weight_sample_final = np.ones((len(y_polytope), n_clusters)) / n_clusters
        ## change the weight of positivess to be 1, negatives to be 1/_clusters
        # then set the positives' weight to be 1 for the assigned hyperplane
        for n in range(len(index_positives)):
            weight_sample_final[index_positives[n], :] *= 0
            weight_sample_final[index_positives[n], final_predict[n]] = 1

        ## create the final polytope by applying all weighted subjects
        for cluster_i in range(n_clusters):
            cluster_sample_final = np.ascontiguousarray(weight_sample_final[:, cluster_i])
            SVM_coefficient, SVM_intercept  = self.launch_svc(X, y_polytope, cluster_sample_final)
            self.coefficients[idx_outside_polytope][cluster_i] = SVM_coefficient
            self.intercepts[idx_outside_polytope][cluster_i] = SVM_intercept

    def optimize_dual(self, X, y, idx_outside_polytope, epsilon=0.001):
        n_clusters = self.n_clusters_per_label[idx_outside_polytope]
        ## put the label idx_center_polytope at the center of the polytope by setting it to positive labels
        y_polytope = np.copy(y)
        y_polytope[
            y_polytope != idx_outside_polytope] = -1  ## if label is inside of the polytope, the distance is negative and the label is not divided into
        y_polytope[
            y_polytope == idx_outside_polytope] = 1  ## if label is outside of the polytope, the distance is positive and the label is clustered

        censensus_assignment = np.zeros(
            (len(y_polytope[y_polytope == 1]), self.n_consensus))  ## only consider the positive index
        cluster_index = censensus_assignment[0, :]

        index_positives = np.where(y_polytope == 1)[0]  # index for Positive Labels
        index_negatives = np.where(y_polytope == -1)[0]  # index for Negative Labels

        weight_sample = np.ones((len(y_polytope), n_clusters)) / n_clusters
        ## depending on the weight initialization strategy, random hyperplanes were initialized with maximum diversity to constitute the convex polytope
        weight_positive_samples = self.init_weight(X, y_polytope, index_positives, index_negatives, n_clusters, initialization_type=self.initialization_type)
        weight_sample[index_positives] = weight_positive_samples  ## only replace the sample weight for positive samples

        ## cluster assignment is based on this svm scores across different SVM/hyperplanes
        svm_scores = np.zeros(weight_sample.shape)

        for iter in range(self.n_iterations):
            for cluster in range(n_clusters):
                cluster_weight = np.ascontiguousarray(weight_sample[:, cluster])
                SVM_coefficient, SVM_intercept = self.launch_svc(X, y_polytope, cluster_weight)
                ## Apply the data again the trained model to get the final SVM scores
                svm_scores[:, cluster] = (
                            np.matmul(SVM_coefficient, X.transpose()) + SVM_intercept).transpose().squeeze()

            cluster_index = np.argmax(svm_scores[index_positives], axis=1)

            ## decide the convergence of the polytope based on the toleration
            weight_sample_hold = weight_sample.copy()
            # after each iteration, first set the weight of patient rows to be 0
            weight_sample[index_positives, :] = epsilon / n_clusters
            # then set the positives samples weight to be 1 for the assigned hyperplane
            for n in range(len(index_positives)):
                weight_sample[index_positives[n], cluster_index[n]] = 1 - epsilon

            ## check the loss comparted to the tolorence for stopping criteria
            loss = np.linalg.norm(np.subtract(weight_sample, weight_sample_hold), ord='fro')
            if loss < self.tolerance:
                break
        ## update the cluster index for the consensus clustering
        censensus_assignment[:, consensus_i] = cluster_index + 1



    def init_weight(self, X, y, index_positives, index_negatives, n_clusters, initialization_type="DPP") :
        if initialization_type == "DPP":  ##
            num_subject = y.shape[0]
            W = np.zeros((num_subject, X.shape[1]))
            for j in range(num_subject):
                ipt = np.random.randint(len(index_positives))
                icn = np.random.randint(len(index_negatives))
                W[j, :] = X[index_positives[ipt], :] - X[index_negatives[icn], :]

            KW = np.matmul(W, W.transpose())
            KW = np.divide(KW, np.sqrt(np.multiply(np.diag(KW)[:, np.newaxis], np.diag(KW)[:, np.newaxis].transpose())))
            evalue, evector = np.linalg.eig(KW)
            Widx = sample_dpp(np.real(evalue), np.real(evector), n_clusters)
            prob = np.zeros((len(index_positives), n_clusters))  # only consider the PTs

            for i in range(n_clusters):
                prob[:, i] = np.matmul(
                    np.multiply(X[index_positives, :], np.divide(1, np.linalg.norm(X[index_positives, :], axis=1))[:, np.newaxis]),
                    W[Widx[i], :].transpose())

            l = np.minimum(prob - 1, 0)
            d = prob - 1
            S = proportional_assign(l, d)
        elif initialization_type == "GMM":
            X_positives = X[index_positives, :]
            random_index_choice = np.random.randint(len(X_positives), size=len(X_positives)//3)
            X_subset = X_positives[random_index_choice, :]

            GMM = GaussianMixture(n_components=n_clusters, max_iter=2).fit(X_subset)
            S = GMM.predict_proba(X_positives)
        return S

    def launch_svc(self, X, y, sample_weight) :
        SVC_clsf = SVC(kernel='linear', C=self.C)
        ## fit the different SVM/hyperplanesDPP
        SVC_clsf.fit(X, y, sample_weight=sample_weight)

        SVM_coefficient = SVC_clsf.coef_
        SVM_intercept = SVC_clsf.intercept_

        return SVM_coefficient, SVM_intercept


