from datetime import datetime, timedelta

# ts序列转成 week和hour序列
def tf_date_trans(ts_array):    
    transed_week = []
    transed_hour = []
    for ts in ts_array:
        ts = ts.decode()
        if(ts > '6000000000' or ts < '1000000000'):
            transed_week.append(0)
            transed_hour.append(0)
        else:
            week, hour = get_week_hour(ts)
            transed_week.append(week)
            transed_hour.append(hour)
    return transed_week, transed_hour 

# ts序列转成 week和hour序列
def date_trans(ts_array):    
    transed_week = []
    transed_hour = []
    for ts in ts_array:
        if(ts > '6000000000' or ts < '1000000000'):
            transed_week.append(0)
            transed_hour.append(0)
        else:
            week, hour = get_week_hour(ts)
            transed_week.append(week)
            transed_hour.append(hour)
    return transed_week, transed_hour 

# 因为要padding, 每天0点按24点处理， 每周从周1-周7
def get_week_hour(ts):
    try:
        t = datetime.fromtimestamp(int(ts))
        if t.hour == 0:
            hour = 24
        else:
            hour = t.hour

        return t.weekday() + 1, hour
    except:
        print("timestamp trans error!",ts)
        return 0, 0

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

def get_local_time():
    cur = datetime.now()
    cur = cur.strftime('%b-%d-%Y')
    return cur