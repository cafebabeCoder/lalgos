# -*- coding: utf-8 -*-
# @Time   : 2021/3/7
# @Author : lorineluo 

import logging
import os
import datetime
import colorlog
from colorama import init
from .file_utils import ensure_dir
from .date_utils import get_local_time

def init_logger(log_state, log_path):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Example:
        >>> logger = logging.getLogger(config)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    LOGROOT = log_path 
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)

    logfilename = '{}-{}.log'.format("log", get_local_time())

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "[%a %d %b] %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d-%b %H:%M:%S [%s]"
    sformatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', sdatefmt)
    if log_state is None or log_state.lower() == 'info':
        level = logging.INFO
    elif log_state.lower() == 'debug':
        level = logging.DEBUG
    elif log_state.lower() == 'error':
        level = logging.ERROR
    elif log_state.lower() == 'warning':
        level = logging.WARNING
    elif log_state.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[fh, sh])

# logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'

if __name__ == '__main__':
    init_logger(log_state='info', log_path="/tmp/logs")
    log = logging.getLogger()
    log.info('test')
    log.warning('test')