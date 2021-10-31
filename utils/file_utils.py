#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2021/08/1
# @Author : lorineluo 
# @File    : file_utils.py 

import os
from os import path
from datetime import datetime, timedelta
import time
from functools import wraps

def ensure_dir(dir_path): 
    print(dir_path)
    if not os.path.exists(dir_path): 
        os.makedirs(dir_path) 


def get_file_list(_dict_path, _start_time, _days):
    """
    样本路径下包含若干文件夹，以YYYYMMDD命名
    给定样本路径，训练开始时间(YYYYMMDD)，训练多少天；返回训练样本list
    
    Args:
        _dict_path: 样本路径
        _start_time: 开始时间
        _days: 训练多少天
    """
    _file_list = []
    for i in range(_days):
        dpath = os.path.join(_dict_path, get_day(_start_time, i))
        print("data path: {0}".format(dpath))
        if os.path.exists(dpath):
            files = os.listdir(dpath)
            for file in files:
                file_path = os.path.join(dpath, file)
                if os.path.isfile(file_path):
                    _file_list.append(file_path)

    return _file_list

def get_day(_date_str, i):
    """
    返回_data_str这天前i天的时间 
    
    Args:
        _data_str: 时间基线
        i: 返回i天前的时间，eg.  get_day("20200805", 3) 返回"20200802"
    """
    start_time = datetime.strptime(_date_str, '%Y%m%d')
    cur_time = start_time - timedelta(days=i)
    cur_day = cur_time.strftime('%Y%m%d')
    return cur_day


class Print_log(object):
    '''
    打印日志
    '''
    def __init__(self, logfile='/tmp/out.log', is_write_file=True, split=" ", mode="a"):
        self.logfile = logfile
        self.is_write_file = is_write_file
        self.split = split
        self.mode = mode
 
    def __call__(self, *args):
        # 打开logfile并写入
        t = time.strftime(str("%Y-%m-%d %H:%M:%S"), time.localtime())
        s = '[%s]' % t
        log_str = s + ' ' + str(self.split.join(args))
        print(log_str)
        if self.is_write_file:
            with open(self.logfile, self.mode) as opened_file:
                # 现在将日志打到指定的文件
                opened_file.write(log_str + "\n")