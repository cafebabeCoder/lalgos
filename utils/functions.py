#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2021/08/1
# @Author : lorineluo 
# @File    : file_utils.py 

import numpy as np 
from sklearn.metrics import precision_recall_fscore_support, f1_score

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square

# 一维, 多分类
def f1measure(_real_label, _pred_label, _labels):
    total_acc, total_count = 0, 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        _real_label, _pred_label, labels=_labels, zero_division=0)
    micro_f1 = f1_score(_real_label, _pred_label, labels=_labels, average='micro', zero_division=0)
    weighted_f1 = f1_score(_real_label, _pred_label, labels=_labels, average='weighted', zero_division=0)
    avg_pre = precision.mean()
    avg_rec = recall.mean()
    avg_f1 = f1.mean()
    total_acc += sum([_real_label[i] == _pred_label[i] for i in range(len(_real_label))])
    total_count += len(_real_label)
    acc = total_acc / total_count
    metrics = {
            'acc': acc, 
            'avg_pre':avg_pre,  #/micro_f1
            'avg_recall': avg_rec, 
            'avg_f1':avg_f1, 
            'w_f1':weighted_f1}

    return metrics
