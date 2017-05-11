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
from scipy.stats import rankdata
import collections
from collections import Counter
import matplotlib.pyplot as plt
from my_plots import hist_plt
from my_dataframes import drop_duplicate, drop_emptycell, stock_status
import datetime
from datetime import timedelta

import seaborn as sns

'''strip time那里要記得調成当天的时间'''

delta = 10
date = '626'
day = '26'
ystd = '25'


'''end of day收盘价,最低价'''
os.chdir('/Users/shine/work_hard/financial_network/data/2956_sec')
last = pd.read_csv('day_last_price_reuters_2956.csv')
last = pd.DataFrame(last)


'''------all node decline-------'''
os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
V = nx.read_graphml('2015-1_95.graphml')

d = list(V.nodes())  # 一定要list一下，不然格式不对
print d

os.chdir('/Users/shine/work_hard/financial_network/data/original/thomson_reuters/sec_626')
FileList = glob.glob('*.csv')
# print '-------before',len(FileList)
# print 'successors',len(d)

filelist_new = []
for workingfile in FileList:
    # print workingfile
    filename = workingfile.split('06-')[1]
    filename = filename.split('.c')[0]
    # print filename, 'is the filename'
    if filename in d:
        filelist_new.append(workingfile)


ave_decline = pd.DataFrame()
for workingfile in filelist_new:
    filename = workingfile.split('06-')[1]
    filename = filename.split('.c')[0]

    os.chdir('/Users/shine/work_hard/financial_network/data/original/thomson_reuters/sec_626')
    df = pd.read_csv(workingfile)
    df = pd.DataFrame(df)
    # print df.columns

    df = df[np.isfinite(df['Low'])]  # 删除这一列中的NAN

    df['time'] = pd.to_datetime(df['Time[L]'])
    df.index = df['time']

    suc_ystd = last[last['node'] == filename]
    suc_ystd = suc_ystd[ystd + '-Jun-15']
    suc_ystd = suc_ystd.iloc[0]

    fre = pd.DataFrame()
    for i in range(0, int(330 / delta)):
        if (i*delta < 120) or (i*delta >= 210):  #把中午11:30-13:00的休市时间排除
            start = datetime.datetime(2017,1,26,9,30,00) + i* datetime.timedelta(minutes = delta) ##这个时间要是当天的时间
            end = datetime.datetime(2017,1,26,9,30,00) + (i+1) * datetime.timedelta(minutes = delta)
            a = df[(df.index > start) & (df.index <= end)]
            if (np.isfinite(suc_ystd)) and (len(a['Low'])!=0) :#确保这两个有数，也就是判断这只股票这一天有数，且昨天收盘价不是0或昨天有收盘价
                a = (min(a['Low'])-suc_ystd)/suc_ystd
                # print a
                fre1 = pd.DataFrame({'time':[end],
                                     filename: [a]
                                     })
                fre1.index = fre1['time']
                fre = fre.append(fre1)
    if len(fre) != 0: ##判断这只股票这一天有数
        fre.drop('time', axis=1, inplace=True)#去掉time那一列
        # os.chdir('/Users/shine/work_hard/financial_network/data/2956_sec/successor_decline_10min')
        # fre.to_csv('11.csv')
        ave_decline = pd.concat([ave_decline, fre], axis=1, join='outer')
        print '------',filename

os.chdir('/Users/shine/work_hard/financial_network/data/2956_sec/successor_decline_10min')
ave_decline.to_csv(date+'10min_all_node_decline.csv')


