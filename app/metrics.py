import sklearn
import numpy as np

def weighted_mae(y_true, y_pred, class_weights):
    """
    Compute the weighted mean absolute error
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets.
    class_weights : dict
        It maps class indices (integers) to a weight (float) value
    """
    sample_weight = np.ones_like(y_true, dtype=np.float32)
    for cl, w in class_weights.items():
        sample_weight[y_true == cl] = w
    return sklearn.metrics.mean_absolute_error(y_true, y_pred, sample_weight)

def balanced_accuracy_score(y_true, y_pred):
    """
    Compute the balanced accuracy
    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets.
    """
    y_true = np.round(y_true).astype('int32')
    y_pred = np.round(y_pred).astype('int32')

    return sklearn.metrics.balanced_accuracy_score(y_true, y_pred)


def confusion_matrix(y_true, y_pred, labels):
    """
    Compute the confusion matrix
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets.
    labels : array-like of shape n_classes
        List of labels to index the matrix
    """
    y_true = np.round(y_true).astype('int32')
    y_pred = np.round(y_pred).astype('int32')

    conf_mat = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    normalized_conf_mat = sklearn.preprocessing.normalize(conf_mat, axis=1, norm='l1')

    return conf_mat, normalized_conf_mat

