import numpy as np
import os
from math import sqrt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, accuracy_score


def evaluate_DTI(total_loss, pred_labels, predictions, labels):
    LOSS = np.mean(total_loss)
    pred_labels = np.array(pred_labels)
    predictions = np.array(predictions)
    labels = np.array(labels)

    Acc = accuracy_score(labels, pred_labels)
    Pre = precision_score(labels, pred_labels)
    Recall = recall_score(labels, pred_labels)
    AUC = roc_auc_score(labels, predictions)
    tpr2, fpr2, s2 = precision_recall_curve(labels, predictions)
    AUPR = auc(fpr2, tpr2)

    Acc = float(format(Acc, '.4f'))
    Pre = float(format(Pre, '.4f'))
    Recall = float(format(Recall, '.4f'))
    AUC = float(format(AUC, '.4f'))
    AUPR = float(format(AUPR, '.4f'))
    LOSS = float(format(LOSS, '.4f'))
    return [Acc, Pre, Recall, AUC, AUPR, LOSS]


def evaluate_CPA(total_loss, pred, labels):
    pred = np.array(pred)
    labels = np.array(labels)

    LOSS = np.mean(total_loss)
    LOSS = float(format(LOSS, '.4f'))
    labels = labels.reshape(-1)
    pred = pred.reshape(-1)
    rmse = sqrt(((labels - pred)**2).mean(axis=0))
    pearson = np.corrcoef(labels, pred)[0, 1]
    return [round(rmse, 4), round(pearson, 4), LOSS]


def save_result_DTI(dataset, item, res, epoch):
    print("%s>>> Acc: %f Pre: %f Recall: %f AUC: %f AUPR: %f LOSS:%f" % (item, res[0], res[1], res[2], res[3], res[4], res[5]))

    dir_output = "./result/{}/".format(dataset)
    os.makedirs(dir_output, exist_ok=True)
    file = "./result/{}/{}.txt".format(dataset, item)
    with open(file, 'a') as f:
        res = [epoch] + res
        f.write('\t'.join(map(str, res)) + '\n')
        f.close()


def save_result_CPA(dataset, item, res, epoch):
    print("%s>>> RMSE: %f PCC: %f LOSS:%f" % (item, res[0], res[1], res[2]))

    dir_output = "./result/{}/".format(dataset)
    os.makedirs(dir_output, exist_ok=True)
    file = "./result/{}/{}.txt".format(dataset, item)
    with open(file, 'a') as f:
        res = [epoch] + res
        f.write('\t'.join(map(str, res)) + '\n')
        f.close()

def save(res1, res2):
    file1 = "./result1.txt"
    with open(file1, 'a') as f1:
        f1.write('\t'.join(map(str, res1)) + '\n')
        f1.close()
    file2 = "./result2.txt"
    with open(file2, 'a') as f2:
        f2.write('\t'.join(map(str, res2)) + '\n')
        f2.close()