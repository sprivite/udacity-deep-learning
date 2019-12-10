import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastai.vision import *
import sys

from sklearn.metrics import roc_curve, auc, confusion_matrix

def plot_roc_auc(y_true, y_pred):
    """
    This function plots the ROC curves and provides the scores.
    """

    # initialize dictionaries and array
    fpr = dict()
    tpr = dict()
    roc_auc = np.zeros(3)

    # prepare for figure
    plt.figure()
    colors = ['aqua', 'cornflowerblue']

    # for both classification tasks (categories 1 and 2)
    for i in range(2):
        # obtain ROC curve
        fpr[i], tpr[i], _ = roc_curve(y_true[:,i], y_pred[:,i])
        # obtain ROC AUC
        roc_auc[i] = auc(fpr[i], tpr[i])
        # plot ROC curve
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label='ROC curve for task {d} (area = {f:.2f})'.format(d=i+1, f=roc_auc[i]))
    # get score for category 3
    roc_auc[2] = np.average(roc_auc[:2])

    # format figure
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    plt.show()

    # print scores
    for i in range(3):
        print('Category {d} Score: {f:.3f}'. format(d=i+1, f=roc_auc[i]))


def predict_all(learner, imagelist='sample_predictions.csv', output=None, silent=False):
    '''
    Routine to predict on test set with FastAI learner.
    '''
    labels_task1 = []         # task 1: melanoma vs non-melanoma
    labels_task2 = []
    predictions_task1 = []    # task 2: melanoctyic vs keratinocytic
    predictions_task2 = []

    if output is not None:
        output = open(output, 'w')
        output.write('Id,task_1,task_2\n')

    il = ImageList.from_csv('.', imagelist)
    for j, test_img in enumerate(il.items):

        img = il.open(test_img)

        p_melanoma = learner.predict(img)[-1][0]
        predictions_task1.append(p_melanoma)

        p_keratosis = learner.predict(img)[-1][-1]
        predictions_task2.append(p_keratosis)

        if 'melanoma' in test_img:
            labels_task1.append(1) # benign
        else:
            labels_task1.append(0) # malignent

        if 'seborrheic_keratosis' in test_img:
            labels_task2.append(1) # keratinoctyic
        else:
            labels_task2.append(0) # melanocytic

        if not silent:
            print(j, end=' ')

        if output is not None:
            output.write(f'{test_img},{p_melanoma},{p_keratosis}\n')

    output.close()

    return np.array([labels_task1, labels_task2]).T, np.array([predictions_task1, predictions_task2]).T
