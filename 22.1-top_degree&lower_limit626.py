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


delta = 10
date = '626'
day = '26'
ystd = '25'

node = '601318.SH'
filename = date + node + 'lower limit ratio'

'''end of day收盘价,最低价'''
os.chdir('E:/financial_network/data/2956_sec')
last = pd.read_csv('day_last_price_reuters_2956.csv')
last = pd.DataFrame(last)

'''none node return within a day'''
one_ystd = last[last['node'] == node]
one_ystd = one_ystd[ystd+'-Jun-15']
one_ystd = one_ystd.iloc[0]

os.chdir('E:/financial_network/data/original/thomson_reuters/sec_'+date)
FileList1 = glob.glob('*-'+node+'.csv')
for workingfile in FileList1: ##这里其实就一个file
    one_df = pd.read_csv(workingfile)
    one_df = pd.DataFrame(one_df)

one_df = one_df[['#RIC', 'Time[L]', 'Low']]  # 每秒最低价
one_df = one_df[np.isfinite(one_df['Low'])] #删除这一列中的NAN
# one_df['Low']= one_df['Low'].replace(np.nan, np.inf)
one_df['time'] = pd.to_datetime(one_df['Time[L]'])
one_df.index = one_df['time']
# print one_df
one = pd.DataFrame()

for i in range(0, int(330/delta)):
    if (i*delta < 120) or (i*delta >= 210):  #把中午11:30-13:00的休市时间排除
        start = datetime.datetime(2017,1,19,9,30,00) + i* datetime.timedelta(minutes = delta) ##这个时间要是当天的时间
        end = datetime.datetime(2017,1,19,9,30,00) + (i+1) * datetime.timedelta(minutes = delta)
        a = one_df[(one_df.index > start) & (one_df.index <= end)]
        # print i,'time interval',a
        # a = (min(a['Low']) - one_ystd)/one_ystd
        a = min(a['Low'])
        # print a
        one1 = pd.DataFrame({'time': [end],
                            'low': [a]
                           })
        one = one.append(one1)


'''lower limit ratio within a day'''
os.chdir('E:/financial_network/data/threshold/stock_alpha/0.95_2015-1')
df1 = pd.read_excel('node_degree_weight-back_up.xlsx', 'Sheet1')
df1 = pd.DataFrame(df1)
df1.columns = ['node', 'in_degree', 'out_degree', 'weight_target_sum','weight_source_sum']
os.chdir('E:/financial_network/data/2956_sec')
df2 = pd.read_csv('2956_sec_'+date+'.csv')
df2 = pd.DataFrame(df2)

df = pd.merge(df1, df2, how ='outer', on = 'node')
df = df[df['limit_ori'] == -1]
df['first_low_rank'] = rankdata(df['first_low_price_time'], method = 'min')

rng = pd.to_datetime(df['first_low_price_time'])
ts = pd.Series(np.ones(len(rng)), index=rng)

fre = pd.DataFrame()
for i in range(0, int(330/delta)):
    if (i*delta < 120) or (i*delta >= 210):  #把中午11:30-13:00的休市时间排除
        start = datetime.datetime(2017,1,19,9,30,00) + i* datetime.timedelta(minutes = delta) ##这个时间要是当天的时间
        end = datetime.datetime(2017,1,19,9,30,00) + (i+1) * datetime.timedelta(minutes = delta)
        a = ts[(ts.index > start) & (ts.index <= end)]
        fre1 = pd.DataFrame({'time': [end],
                            'frequency': [len(a)]
                           })
        fre = fre.append(fre1)

result = pd.merge(one, fre, how ='outer', on = 'time')

result.to_csv('time_fre_and_one_stock'+date+'.csv')

df = pd.read_csv('time_fre_and_one_stock'+date+'.csv')##为了能用strptime把年份日期去掉，必须重新读入文件
df = pd.DataFrame(df)
df['time'] = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in df['time']]
# df1 = df[(df['time'] <= datetime.datetime(2017,1,19,11,30,00))]
# df2 = df[(df['time'] >= datetime.datetime(2017,1,19,13,00,00))]
df['time'] = [d.time() for d in df['time']]



'''plot'''

fig, ax1 = plt.subplots()
ax1.plot_date(df['time'], df['frequency'],linestyle='-', xdate=True, ydate=False, color = 'b')
datemin = datetime.datetime.strptime('2015-1-10 9:30:00', "%Y-%m-%d %H:%M:%S").time()
datemax = datetime.datetime.strptime('2015-1-10 15:00:00', "%Y-%m-%d %H:%M:%S").time()
a0 = datetime.datetime.strptime('2015-1-10 10:00:00', "%Y-%m-%d %H:%M:%S").time()
a1 = datetime.datetime.strptime('2015-1-10 10:30:00', "%Y-%m-%d %H:%M:%S").time()
a2 = datetime.datetime.strptime('2015-1-10 11:00:00', "%Y-%m-%d %H:%M:%S").time()
a3 = datetime.datetime.strptime('2015-1-10 11:30:00', "%Y-%m-%d %H:%M:%S").time()
a4 = datetime.datetime.strptime('2015-1-10 13:00:00', "%Y-%m-%d %H:%M:%S").time()
a5 = datetime.datetime.strptime('2015-1-10 13:30:00', "%Y-%m-%d %H:%M:%S").time()
a6 = datetime.datetime.strptime('2015-1-10 14:00:00', "%Y-%m-%d %H:%M:%S").time()
a7 = datetime.datetime.strptime('2015-1-10 14:30:00', "%Y-%m-%d %H:%M:%S").time()
labels = [datemin, a0, a1, a2, a3, a4, a5, a6, a7,datemax]
ax1.set_xlim(datemin, datemax)
ax1.xaxis.set_ticks(labels) #to get a tick every 15 minutes
ax1.set_ylabel('time interval lower limit number', color='b')
ax1.tick_params('time interval lower limit number', colors='b')
ax2 = ax1.twinx()
ax2.plot_date(df['time'], df['low'],linestyle='--', xdate=True, ydate=False, color = 'r')
ax2.set_ylabel('lowest return till now', color='r')
ax2.tick_params('lowest return till now', colors='r')
# plt.legend(loc='best')
plt.title(filename)
plt.xlabel('time')
fig.tight_layout()
fig.autofmt_xdate() #automatically rotates dates appropriately for you figure.
plt.savefig(filename + "_fre.png")
plt.clf()