#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
各种采样方式
"""
import pandas as pd
import random
import math
import numpy as np

appid_path = "/apdcephfs/private_lorineluo/python_data_analysis/data/transformer/sample/appid"
ori_file = '/apdcephfs/private_lorineluo/python_data_analysis/data/transformer/sample/20201202/AppuinTransSampleBST_20201112'
result_file = '/apdcephfs/private_lorineluo/python_data_analysis/data/transformer/sample/20201202/aa'

# w2v的负采样方式
def neg_sample(cnt, total_count=971904000, param=0.0000001):
    p = cnt * 1.0/ (971904000)
    return (math.sqrt(p / param) + 1) * (param/ p)

# 计算词频
def word_count():
    _CSV_COLUMN_NAMES = ['today_seq']
    data = pd.read_csv(appid_path, names=_CSV_COLUMN_NAMES)
    data = data[data['today_seq']!=""]
    c = data.groupby(['today_seq'], as_index=False)['today_seq'].agg({'cnt':'count'})
    c = c[c.cnt > 1000]
    freq = dict(zip(c['today_seq'], c['cnt']))
    return freq

def main():
    count = 0
    with open(result_file, 'w') as w:
        with open(ori_file, 'r') as r:
            for line in r:
                count += 1
                if len(line.split(",")) != 18 :
                    print(len(line.split(",")))
                    continue
                aid = line.split(",")[1].split("|")[0]
                freq = word_count()
                cnt = freq.get(aid, 100)
                p = random.random()
                neg_p = neg_sample(cnt)
                if(p < neg_p):
                    w.write(line)
                if count % 200000 == 0:
                    print(count)

if __name__ == '__main__':
    main()