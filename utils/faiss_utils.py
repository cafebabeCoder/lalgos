#!/data/anaconda2/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 14:39
# @Author  : lorineluo
# @File    : faiss_recall_All.py

import argparse
import faiss
import numpy as np
import pandas as pd

PATHS='D:\workPlace\gitCode\python-lorineluo-dev\src\main\python\youtube\data\yoo'

parser = argparse.ArgumentParser(description="dnnr find cmsids")
parser.add_argument("--user", help="user input path", default=PATHS + "\\user_vector")
parser.add_argument("--item", help="item input path", default=PATHS + "\\item_vector")
parser.add_argument("--output", help="output path", default="./result")
parser.add_argument("--dimension", type=int, help="vec dimension", default=128)
parser.add_argument("--topK", type=int, help="k nearest", default=100)
parser.add_argument("--nlist", type=int, help="just IVF", default=10)
parser.add_argument("--nprobe", type=int, help="just IVF", default=10)
parser.add_argument("--reg", type=int, help="1=normalization, 0=not", default=0)
parser.add_argument("--index", help="train or test", choices=["FlatIP", "FlatL2", "FlatIPGPU", "IVFFlat", "IVFFlatGPU"], default="FlatIPGPU")

args = parser.parse_args()
np.set_printoptions(precision=2)

DEMENSION = args.dimension
TOPK = args.topK
NLIST = args.nlist
NPROBE = args.nprobe

#IndexFlatIP 默认cosine搜索 暴搜
class FlatIP:
    def __init__(self, vec):
        self.embeddings = vec
        self.index = faiss.IndexFlatIP(DEMENSION)
        self.index.train(self.embeddings)
        self.index.add(self.embeddings)

#d2距离，暴搜
class FlatL2:
    def __init__(self, vec):
        self.embeddings = vec
        self.index = faiss.IndexFlatL2(DEMENSION)
        print(self.index.is_trained)
        self.index.add(vec)  # add vectors to the index

#GPU flatIp暴搜,  效果最好
class FlatIPGPU:
    def __init__(self, vec):
        self.embeddings = vec
        self.index = faiss.GpuIndexFlat(faiss.StandardGpuResources(),DEMENSION, faiss.METRIC_INNER_PRODUCT)
        self.index.train(self.embeddings)
        self.index.add(self.embeddings)

# kmeans 搜，nlist聚多少类中心， nprobe 搜多少类中心, nlist+nprobe要配合调试
class IVFFlat:
    def __init__(self, vec):
        self.embeddings = vec
        self.quantizer = faiss.IndexFlatIP(DEMENSION)
        self.index = faiss.IndexIVFFlat(self.quantizer, DEMENSION, NLIST, faiss.METRIC_INNER_PRODUCT)
        self.index.train(self.embeddings)
        self.index.nprobe = NPROBE
        self.index.add(self.embeddings)

#加GPU，nprobe 搜多少类中心 nlist聚多少类
class IVFFlatGPU:
    def __init__(self, vec):
        self.embeddings = vec
        self.res = faiss.StandardGpuResources()
        self.index = faiss.GpuIndexIVFFlat(self.res, DEMENSION, NLIST, faiss.METRIC_INNER_PRODUCT)
        self.index.train(self.embeddings)
        self.index.nprobe = NPROBE
        self.index.add(self.embeddings)

# 向量取模
def modOfvec(v):
    return np.sqrt(sum(v*v))


def readDataPandas(inputpath):
    data = pd.read_csv(inputpath, sep='\t', header=None,
                       names=['id', 'vector'], keep_default_na=False)
    print("data shape : %d, %d" % data.shape)
    data_all = data.drop_duplicates(subset=['id'], keep='first').copy()  #data.loc[~data.vid.duplicated(), :]

    data_all.loc[:, 'vector'] = data_all.loc[:, 'vector'].apply(
        lambda x: np.fromstring(x, sep=','))
    data_all.loc[:, "vl"] = data_all['vector'].apply(lambda x: len(x))
    # 向量归一化
    if args.reg == 1:
        data_all['mod'] = data_all['vector'].apply(lambda x: modOfvec(x))
        data_all['vector'] = data_all['vector'] / data_all['mod']
    data_all = data_all[data_all.vl == DEMENSION]

    vecs = np.stack(data_all['vector'], axis=0).astype('float32')
    vids = data_all['id'].values
    return vids, vecs

#"FlatIP", "FlatL2", "FlatIPGPU", "IVFFlat", "IVFFlatGPU"
def getFaissIndex(vec):
    if args.index == "FlatIP":
        return FlatIP(item_vec)
    elif args.index == "FlatL2":
        return FlatL2(item_vec)
    elif args.index == "IVFFlat":
        return IVFFlat(item_vec)
    elif args.index == "IVFFlatGPU":
        return IVFFlatGPU(item_vec)
    else:
        return FlatIPGPU(item_vec)


if __name__ == '__main__':
    print("==========")
    print(args)
    (user, user_vec) = readDataPandas(args.user)
    (vid, item_vec) = readDataPandas(args.item)
    print("user shape : %d, %d" % user_vec.shape)
    print("item shape : %d, %d" % item_vec.shape)

    p = getFaissIndex(item_vec)
    D, I = p.index.search(user_vec, TOPK)  # actual search

    # save
    itemDic = dict(zip(range(len(vid)), vid))
    userDic = zip(user, I)
    # scoreDic = zip(data.cmsid, D)

    i = 0
    with open(args.output, "w+") as wf:
        for userid, indexs in userDic:
            recommVids = []
            j = 0
            for idx in indexs:
                if idx > 0 & idx < len(vid):
                    recommVids.append(itemDic[idx] + '|' + str(np.round(D[i][j], 4)))
                j = j + 1
            i = i + 1
            wf.write(userid + ":" + ",".join(recommVids) + "\n")