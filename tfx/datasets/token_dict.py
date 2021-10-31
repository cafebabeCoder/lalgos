#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
appid -> token_id
"""

def main():
    input_file = "/apdcephfs/private_lorineluo/python_data_analysis/data/transformer/common/recomm_appuininfo"
    output_file = "/apdcephfs/private_lorineluo/python_data_analysis/data/transformer/common/AcctRecommendTransformer.dict"   # 对应tfcc的模型名
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as wf:
        token_id = 0
        wf.write("<PAD> %d\n" % token_id)
        token_id = token_id + 1
        wf.write("<UNK> %d\n" % token_id)
        token_id = token_id + 1

        lines = f.readlines()
        for line in lines:
            l = line.strip().split(',')
            wf.write("%s %d\n" % (l[-1], token_id))
            token_id = token_id + 1

        wf.write("<START> %d\n" % token_id)
        token_id = token_id + 1
        wf.write("<END> %d\n" % token_id)

if __name__ == '__main__':
    main()