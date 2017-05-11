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
import matplotlib.dates as mdates
from my_plots import hist_plt
from my_dataframes import drop_duplicate, drop_emptycell, stock_status
import datetime


delta = 30
date = '708'
filename = '708_%r_min'%delta

os.chdir('E:/financial_network/data/threshold/stock_alpha/0.95_2015-1')
df1 = pd.read_excel('node_degree_weight-back_up.xlsx', 'Sheet1')
df1 = pd.DataFrame(df1)
df1.columns = ['node', 'in_degree', 'out_degree', 'weight_target_sum','weight_source_sum']
print df1.columns
# print df1[['node']]

os.chdir('E:/financial_network/data/2956_sec')
df2 = pd.read_csv('2956_sec_'+date+'.csv')
df2 = pd.DataFrame(df2)
print df2.columns
# print df2[['node']]

df = pd.merge(df1, df2, how ='outer', on = 'node')
df = df[df['limit_ori'] == -1]
# df = df.sort(['first_low_price_time'], ascending=[1]).to_string(index=False)
df['first_low_rank'] = rankdata(df['first_low_price_time'], method = 'min')
# print df['first_low_rank']
df.to_csv('sec_'+date+'_lower.csv')

# x = pd.date_range("9:25:00", "15:00:10", freq="5min").time
# print x
#
rng = pd.to_datetime(df['first_low_price_time'])
print rng[:5]
ts = pd.Series(np.ones(len(rng)), index=rng)
print ts

fre = pd.DataFrame()
for i in range(0, int(330/delta+1)):
    start = datetime.datetime(2017,1,16,9,30,00) + i* datetime.timedelta(minutes = delta) ##这个时间要是当天的时间
    end = datetime.datetime(2017,1,16,9,30,00) + (i+1) * datetime.timedelta(minutes = delta)
    a = ts[(ts.index > start) & (ts.index <= end)]
    fre1 = pd.DataFrame({'time': [end],
                        'frequency': [len(a)]
                        })
    fre = fre.append(fre1)
#
fre.to_csv('time_fre_'+date+'.csv')

df = pd.read_csv('time_fre_'+date+'.csv')##为了能用strptime把年份日期去掉，必须重新读入文件
df = pd.DataFrame(df)
df['time'] = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in df['time']]
df['time'] = [d.time() for d in df['time']]
print df['time']

fig, ax = plt.subplots()
plt.plot_date(df['time'], df['frequency'],linestyle='-', xdate=True, ydate=False)
plt.xlabel('time')
plt.ylabel('number_of_lower_limit')
datemin = datetime.datetime.strptime('2015-1-10 9:30:00', "%Y-%m-%d %H:%M:%S").time()
datemax = datetime.datetime.strptime('2015-1-10 15:00:00', "%Y-%m-%d %H:%M:%S").time()
ax.set_xlim(datemin, datemax)
a0 = datetime.datetime.strptime('2015-1-10 10:00:00', "%Y-%m-%d %H:%M:%S").time()
a1 = datetime.datetime.strptime('2015-1-10 10:30:00', "%Y-%m-%d %H:%M:%S").time()
a2 = datetime.datetime.strptime('2015-1-10 11:00:00', "%Y-%m-%d %H:%M:%S").time()
a3 = datetime.datetime.strptime('2015-1-10 11:30:00', "%Y-%m-%d %H:%M:%S").time()
a4 = datetime.datetime.strptime('2015-1-10 13:00:00', "%Y-%m-%d %H:%M:%S").time()
a5 = datetime.datetime.strptime('2015-1-10 13:30:00', "%Y-%m-%d %H:%M:%S").time()
a6 = datetime.datetime.strptime('2015-1-10 14:00:00', "%Y-%m-%d %H:%M:%S").time()
a7 = datetime.datetime.strptime('2015-1-10 14:30:00', "%Y-%m-%d %H:%M:%S").time()
labels = [datemin, a0, a1, a2, a3, a4, a5, a6, a7,datemax]
ax.xaxis.set_ticks(labels) #to get a tick every 15 minutes
plt.title(filename)
fig.autofmt_xdate() #automatically rotates dates appropriately for you figure.
# plt.show()
plt.savefig(filename + "_fre.png")
plt.clf()