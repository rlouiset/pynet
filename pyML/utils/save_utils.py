import numpy as np

HYBRID_METHODS = ['HYDRA']


def fullfill_predictions(dict_of_predictions, ML_method_name, ML_method, fold, data_manager, features_df, features_names,
                         dict_of_representations):
    ''' fullfill the dictionary containing predictions '''
    fold_predictions_dict = {'important_features': get_most_important_features(ML_method, features_names), 'dict_of_metadata': None}

    # get TEST/TRAIN/VAL indices and representation
    test_indices = data_manager.dataset['test'].indices
    train_indices = data_manager.dataset['train'][fold].indices
    val_indices = data_manager.dataset['validation'][fold].indices

    fold_predictions_dict['dict_of_metadata'] = {'TRAIN': features_df.iloc[train_indices].to_json(),
                                                 'VAL': features_df.iloc[val_indices].to_json(),
                                                 'TEST': features_df.iloc[test_indices].to_json()}

    # get CLASSIFICATION predictions
    classification_predictions = {}
    for PHASE in ['TRAIN', 'VAL', 'TEST']:
        X_rep_phase = dict_of_representations[PHASE]
        classification_predictions[PHASE] = ML_method.predict(X_rep_phase).tolist()
    fold_predictions_dict['classification_predictions'] = classification_predictions

    # get CLUSTERING predictions
    if ML_method_name in HYBRID_METHODS:
        clustering_predictions = {}
        for PHASE in ['TRAIN', 'VAL', 'TEST']:
            X_rep_phase = dict_of_representations[PHASE]
            clusters_pred = ML_method.predict_clusters(X_rep_phase)
            clusters_pred = {key: value.tolist() for (key, value) in clusters_pred.items()}
            clustering_predictions[PHASE] = clusters_pred
        fold_predictions_dict['clustering_predictions'] = clustering_predictions

    dict_of_predictions[str(fold)] = fold_predictions_dict
    return dict_of_predictions


def get_most_important_features(ML_method, features_names):
    important_features = {label:[] for label in range(ML_method.n_labels)}

    for label in range(ML_method.n_labels):
        for cluster in range(ML_method.n_clusters_per_label[label]):
            coeff = ML_method.coefficients[label][cluster][0]
            idx = np.argsort(coeff)[:5]
            important_features[label].extend(list(np.array(features_names)[idx]))

    for label in range(ML_method.n_labels):
        important_features[label] = list(np.unique(important_features[label]))

    return important_features
