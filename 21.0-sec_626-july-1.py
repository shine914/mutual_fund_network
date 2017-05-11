# -*- coding: utf-8 -*-

#avoid chinese mess
import xlrd
import glob,os
import openpyxl
import networkx as nx

import pandas as pd
import numpy as np
import glob,os
import math
from math import log10

import collections
from collections import Counter
import matplotlib.pyplot as plt

from my_dataframes import drop_duplicate, drop_emptycell, stock_status

date = '702'
day = '2'
ystd = '1'

'''是否跌停'''
os.chdir('E:/financial_network/data/status_wind_612_710')
# status = pd.read_excel('20150626.xlsx', 'Wind资讯'.decode('utf-8'))
status = pd.read_excel('20150'+date+'.xlsx', 'Wind资讯'.decode('utf-8'))
status = pd.DataFrame(status)
status.columns = ['node', 'stock_name', 'limit_ori', 'sus_reason', 'sus_days', 'sus_ori']
# status['limit'] = status.apply(lambda df: stock_status(df, 'limit_ori', 'sus_ori'), axis=1)
status = status[['node', 'limit_ori']]
print 'status has', len(status)
print status.columns
print '-------and-----------------'

'''end of day收盘价,最低价'''
os.chdir('E:/financial_network/data/2956_sec')
last = pd.read_csv('day_last_price_reuters_2956.csv')
last = pd.DataFrame(last)
print last.columns

last_today = last[['node',day+'-Jul-15']]
last_today.columns = ['node',day+'-Jul-15_last']

last_ystd = last[['node', ystd+'-Jul-15']]
last_ystd.columns = ['node',ystd+'-Jul-15_last']
print 'last has files', len(last)

low = pd.read_csv('day_last_price_reuters_2956.csv')
low = pd.DataFrame(low)

low = low[['node', day+'-Jul-15']]

low.columns = ['node', day+'-Jul-15_low']
print 'low has files', len(last)

'''今日sec价首次达到最低价的时间'''
# os.chdir('E:/financial_network/data/original/thomson_reuters/sec_626')
os.chdir('E:/financial_network/data/original/thomson_reuters/sec_'+date)
FileList1 = glob.glob('*.csv')
print 'second data has', len(FileList1)
t_df = pd.DataFrame()
for workingfile in FileList1:
    #print workingfile
    filename = workingfile.split('07-')[1]
    filename = filename.split('.c')[0]
    print filename, 'is the filename'

    df = pd.read_csv(workingfile)
    df = pd.DataFrame(df)
    df = df[['#RIC','Time[L]','Low']] #每秒最低价

    # its_low = low[low['node'] == filename]['26-Jun-15_low']
    its_low = low[low['node'] == filename][day+'-Jul-15_low']
    if not its_low.empty:
        for i, value in enumerate(its_low): ##extract value from array
            its_low = value
            # print its_low
        for i in range(0, len(df)):
            if df.iloc[i, 2] == its_low:
                its_time = pd.DataFrame(df.iloc[i,:])
                its_time = its_time.transpose()
                # print its_time
                its_time = its_time[['#RIC','Time[L]']]
                its_time.columns =['node', 'first_low_price_time']
                t_df = pd.concat([t_df, its_time])
                #print t_df
                break

os.chdir('E:/financial_network/data/2956_sec')
# t_df.to_csv('2956_sec_626_first_to_low.csv')
t_df.to_csv('2956_sec_'+ date +'_first_to_low.csv')
result = pd.merge(status, t_df, how ='outer', on = 'node')
result = pd.merge(result, low, how ='outer', on = 'node')
result = pd.merge(result, last_today, how ='outer', on = 'node')
result = pd.merge(result, last_ystd, how ='outer', on = 'node')
# result.to_csv('2956_sec_626.csv')
result.to_csv('2956_sec_'+date+'.csv')
###这段代码跑了快2小时