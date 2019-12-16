
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

import metrics

def _draw_conf_matrix(conf_mat, classes):
    ax = plt.subplot()
    sns.heatmap(conf_mat, annot=True, ax = ax, cmap='Greens')

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)
    plt.show()


def print_metrics(mode, y_true, y_pred, classes, class_weights, draw_conf_matrix=False):
    print('\n%s' % mode)
    print('MAE: %4f' % mean_absolute_error(y_true, y_pred))
    print('Weighted MAE: %.4f' % metrics.weighted_mae(y_true=y_true, y_pred=y_pred, class_weights=class_weights))
    print('Balanced accuracy score: %.4f' % metrics.balanced_accuracy_score(y_true, y_pred))
    if draw_conf_matrix:
        conf_mat, normalized_conf_mat = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=classes)
        print('Confusion matrix:\n', conf_mat)
        _draw_conf_matrix(normalized_conf_mat, classes)