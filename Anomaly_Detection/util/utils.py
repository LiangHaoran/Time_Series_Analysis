"#_*_ coding:utf-8 _*_"
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from sklearn.metrics import confusion_matrix
import random
from math import isinf
import math
from tqdm import tqdm
from PIL import Image
import warnings
import time
import argparse
warnings.filterwarnings('ignore')


def normalized(max_value, min_value, data, a=-1, b=1):
    """
    Normalize the time series to (-1, 1)
    :param max_value:
    :param min_value:
    :param data:
    :param a:
    :param b:
    :return:
    """
    k = (b - a) / (max_value - min_value)
    m, n = data.shape
    out = np.ones([m, n])
    for i in range(m):
        for j in range(n):
            out[i, j] = a + k*(data[i, j] - min_value)
    return out


def data2feature(data):
    """
    Calculate signature matrices
    """
    m, n = data.shape
    feat = np.ones([data.shape[1], data.shape[1]])
    feat = np.dot(data.T, data)/m
    # find max value and min value
    max_item = max(max(row) for row in feat)
    min_item = min(min(row) for row in feat)
    # normalize
    feat = normalized(max_value=max_item, min_value=min_item, data=feat, a=-1, b=1)
    return feat


def sampling(data):
    """
    Sampling from original time series
    :param data:
    :return:
    """
    feat = []
    for i in range(data.shape[0]):
        if i%2 != 0:
            feat.append(data[i, :])
    feat = np.array(feat)
    return feat


def weight_time(series, nn=0.9, model='p'):
    """
    Forgetting mechanism
    :param series: multi-time series
    :param nn:
    :param model: p: power; l: linear; s: without forgetting mechanism
    :return:
    """
    series = np.transpose(series)
    lens = series.shape[0]
    if model == 'p':
        weights = np.power(nn, np.arange(lens)[::-1])
    elif model == 'l':
        weights = nn * (np.arange(lens))
    elif model == 's':
        weights = np.ones(lens)
    weights = np.array(weights).reshape(-1, 1)
    weights = np.repeat(weights, series.shape[1], axis=1)
    series = np.multiply(series, weights)
    return np.transpose(series)


def get_metrics(series_path, window_1=10, window_2=30, window_3=60, dimension=10, fm='p'):
    """
    Transform multi-dimensional time series into signature matrices
    :param series_path:
    :param window_1:
    :param window_2:
    :param window_3:
    :param dimension:
    :param fm: forgetting mechanism model: p, l, s
    :return:
    """
    time_series = np.array(pd.read_csv(series_path, header=None))
    print('time series:', time_series.shape)

    """Sample channel，window length = window_1"""
    sample_window_1 = []
    for i in tqdm(range((window_1 - 1), (time_series.shape[0]))):
        tem_se = time_series[i - (window_1 - 1):i + 1, :]
        tem = sampling(data=tem_se)
        sample_window_1.append(tem)
    sample_window_1 = np.array(sample_window_1)

    """Correlation channel，window length = window_1"""
    out_window_1 = []
    for i in tqdm(range((window_1 - 1), (time_series.shape[0]))):
        tem_se = time_series[i - (window_1 - 1):i + 1, :]
        # forgetting mechanism
        tem_se = weight_time(series=tem_se, nn=1.05, model=fm)
        tem = data2feature(data=tem_se)
        out_window_1.append(tem)
    out_window_1 = np.array(out_window_1)

    """Correlation channel，window length = window_2"""
    out_window_2 = []
    for i in tqdm(range((window_2 - 1), (time_series.shape[0]))):
        tem_se = time_series[i - (window_2 - 1):i + 1, :]
        # forgetting mechanism
        tem_se = weight_time(series=tem_se, nn=1.05, model=fm)
        tem = data2feature(data=tem_se)
        out_window_2.append(tem)
    out_window_2 = np.array(out_window_2)

    """Correlation channel，window length = window_3"""
    out_window_3 = []
    for i in tqdm(range((window_3 - 1), (time_series.shape[0]))):
        tem_se = time_series[i - (window_3 - 1):i + 1, :]
        # forgetting mechanism
        tem_se = weight_time(series=tem_se, nn=1.05, model=fm)
        tem = data2feature(data=tem_se)
        out_window_3.append(tem)
    out_window_3 = np.array(out_window_3)

    """Combine multiple signature matrices"""
    max_window = np.max((window_1, window_2, window_3))
    channel_1 = out_window_1[(max_window - window_1):, :, :]
    channel_2 = out_window_2[(max_window - window_2):, :, :]
    channel_3 = out_window_3[(max_window - window_3):, :, :]
    channel_4 = sample_window_1[(max_window - window_1):, :, :]

    channel_1 = channel_1.reshape(-1, 1, dimension, dimension)
    channel_2 = channel_2.reshape(-1, 1, dimension, dimension)
    channel_3 = channel_3.reshape(-1, 1, dimension, dimension)
    channel_4 = channel_4.reshape(-1, 1, dimension, dimension)

    channel_1 = channel_1.transpose(1, 0, 2, 3)
    channel_2 = channel_2.transpose(1, 0, 2, 3)
    channel_3 = channel_3.transpose(1, 0, 2, 3)
    channel_4 = channel_4.transpose(1, 0, 2, 3)

    combine_channel = np.vstack((channel_1, channel_2, channel_3, channel_4))
    combine_channel = combine_channel.transpose(1, 2, 3, 0)
    return combine_channel


def load_data(train_path, test_path, label_path, window_1, window_2, window_3, dimension, data_type, fm):
    """
    load data
    :param train_path:
    :param test_path:
    :param label_path:
    :param window_1:
    :param window_2:
    :param window_3:
    :param dimension:
    :param data_type:
    :return:
    """
    if data_type == 'train':
        # load train data
        train_data = get_metrics(series_path=train_path, window_1=window_1, window_2=window_2, window_3=window_3, dimension=dimension, fm=fm)
        print('train data shape:', train_data.shape)
        return train_data
    else:
        # load test data
        test_data = get_metrics(series_path=test_path, window_1=window_1, window_2=window_2, window_3=window_3, dimension=dimension, fm=fm)
        label = np.array(pd.read_csv(label_path, header=None))
        print('test data shape:', test_data.shape)
        return test_data, label


def convert_01(data, threshold):
    """
    Convert the detection score to 0 or 1
    :param data:
    :param threshold:
    :return:
    """
    out = []
    data = np.array(data).reshape(-1, 1)
    for i in range(data.shape[0]):
        if data[i] >= threshold:
            out.append(1)
        else:
            out.append(0)
    return np.array(out)


def cal_fpr_tpr(detec, label):
    """
    Calculate tpr, fpr, tp, fp, fn, tn
    :param detec:
    :param label:
    :return:
    """
    tp, fp, fn, tn = 0, 0, 0, 0
    label = np.array(label).reshape(-1, 1)
    # compute confusion matrix
    cm = confusion_matrix(label, detec)

    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    tpr = float(tp) / float(tp+fn)
    fpr = float(fp) / float(fp+tn)
    return tpr, fpr, tp, fp, fn, tn


def eval(pred, true, verbose=0):
    """
    Calculate precision, recall and f1
    :param pred: 0-1
    :param true: 0-1
    :param verbose:
    :return:
    """
    pred = np.array(pred)
    true = np.array(true)
    ###
    cm = confusion_matrix(true, pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    mcc = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if verbose == 1:
        print('INFO: TP:', TP)
        print('INFO: FP:', FP)
        print('INFO: FN:', FN)
        print('INFO: TN:', TN)
        print('INFO: MCC:', mcc)
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    f1 = 2*precision*recall/(precision+recall)
    if verbose == 1:
        print('INFO: precision:', precision)
        print('INFO: recall:', recall)
        print('INFO: f1:', f1)
        print('INFO: tpr:', TPR)
        print('INFO: fpr:', FPR)
    return precision, recall, f1


def search_threshold(pred, label, methods="matrix", verbose=0):
    """
    search the best threshold based on the confusion matrix
    :param pred:
    :param label:
    :return:
    """
    # find max and min
    pred = np.array(pred)
    max_value, min_value = max(pred), min(pred)
    thr_range = np.arange(min_value, max_value, (max_value - min_value) / 1000)
    if methods == "matrix":
        maxx = 1000000
        thr_opt = 0
        pred_result = []
        # Search for detection thresholds
        for i in tqdm(thr_range):
            # Convert the test score to 0 or 1 based on the threshold
            search_detection = convert_01(pred, i)
            _, _, tp, fp, fn, tn = cal_fpr_tpr(search_detection, label)
            matrix = (fp + fn) / (tp + tn)
            if matrix < maxx:
                maxx = matrix
                thr_opt = i
                pred_result = search_detection
                mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                fm = tp / np.sqrt((tp+fp)*(tp+fn))
        # Calculated performance index
        # Print information when verbose=1
        print('INFO: Optimal threshold based on confusion matrix:', thr_opt)
        precision, recall, f1 = eval(pred_result, label, verbose=verbose)
        if verbose == 1:
            print('INFO: MCC:', mcc)
            print('INFO: FM:', fm)
            print('***************************')

    else:
        # Select the optimal threshold based on the ROC curve
        roc_detection_resutl = []
        min_dis = 100000
        thr_opt = 0
        for i in tqdm(thr_range):
            search_detection = convert_01(pred, i)
            tpr_tem, fpr_tem, tp, fp, fn, tn = cal_fpr_tpr(search_detection, label)
            # Calculate distance
            tem_dis = (fpr_tem - 0) ** 2 + (tpr_tem - 1) ** 2
            if tem_dis < min_dis:
                min_dis = tem_dis
                thr_opt = i
                roc_detection_resutl = search_detection
                mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                fm = tp / np.sqrt((tp + fp) * (tp + fn))
        print('INFO: Optimal threshold based on ROC curve:', thr_opt)
        precision, recall, f1 = eval(roc_detection_resutl, label, verbose=verbose)
        if verbose == 1:
            print('INFO: MCC:', mcc)
            print('INFO: FM:', fm)
            print('***************************')
    return precision, recall, f1, mcc, fm, thr_opt


