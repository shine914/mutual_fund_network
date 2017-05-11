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

delta = 1
date = '626'
day = '26'
ystd = '25'
node = '601166.SH'
# node = '601318.SH'
f_name = date + '_'+ node + '_ %d min' %delta

'''end of day收盘价,最低价'''
os.chdir('/Users/shine/work_hard/financial_network/data/2956_sec')
last = pd.read_csv('day_last_price_reuters_2956.csv')
last = pd.DataFrame(last)


''''-------大出度个股--------'''

'''none node return within a day'''
one_ystd = last[last['node'] == node]
one_ystd = one_ystd[ystd+'-Jun-15']
one_ystd = one_ystd.iloc[0]

os.chdir('/Users/shine/work_hard/financial_network/data/original/thomson_reuters/sec_'+date)
FileList1 = glob.glob('*-'+node+'.csv')
for workingfile in FileList1: ##这里其实就一个file
    one_df = pd.read_csv(workingfile)
    one_df = pd.DataFrame(one_df)

one_df = one_df[['#RIC', 'Time[L]', 'Low']]  # 每秒最低价
one_df = one_df[np.isfinite(one_df['Low'])] #删除这一列中的NAN
# one_df['Low']= one_df['Low'].replace(np.nan, np.inf)
one_df['time'] = pd.to_datetime(one_df['Time[L]'])
one_df.index = one_df['time']
print one_df
one = pd.DataFrame()

for i in range(0, int(330/delta)):
    if (i*delta < 120) or (i*delta >= 210):  #把中午11:30-13:00的休市时间排除
        start = datetime.datetime(2017,1,26,9,30,00) + i* datetime.timedelta(minutes = delta) ##这个时间要是当天的时间
        end = datetime.datetime(2017,1,26,9,30,00) + (i+1) * datetime.timedelta(minutes = delta)
        a = one_df[(one_df.index > start) & (one_df.index <= end)]
        # print i,'time interval',a
        a = (min(a['Low']) - one_ystd)/one_ystd
        #a = min(a['Low'])
        # print a
        one1 = pd.DataFrame({'time': [end],
                            'low': [a]
                           })
        one = one.append(one1)


'''------successors decline-------'''
os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
V = nx.read_graphml('2015-1_95.graphml')

d = list(V.successors(node))  # 一定要list一下，不然格式不对

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
print '-------after',len(filelist_new)

suc_ave_decline = pd.DataFrame()
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
        suc_ave_decline = pd.concat([suc_ave_decline, fre], axis=1, join='outer')
        print '------',filename

os.chdir('/Users/shine/work_hard/financial_network/data/2956_sec/successor_decline_10min')
suc_ave_decline.to_csv(f_name+'_successor_decline.csv')

'''大出度股票下跌幅度+其他股票的下跌幅度，to csv'''
os.chdir('/Users/shine/work_hard/financial_network/data/2956_sec/successor_decline_10min')
suc_ave_decline = pd.read_csv(f_name+'_successor_decline.csv')
suc_ave_decline = pd.DataFrame(suc_ave_decline)
ave = suc_ave_decline.mean(axis = 1)
ave = pd.DataFrame(ave)
ave.columns = ['successor_decline']

ave.reset_index(drop=True, inplace=True)
one.reset_index(drop=True, inplace=True)

ave_one = pd.concat([ave, one], axis=1)
ave_one.to_csv(f_name+'_ave_one.csv')

'''plot'''
os.chdir('/Users/shine/work_hard/financial_network/data/2956_sec/successor_decline_10min')
ave_one = pd.read_csv(f_name+'_ave_one.csv')
df = pd.DataFrame(ave_one)

df['time'] = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in df['time']]
df1 = df[(df['time'] <= datetime.datetime(2017,1,26,11,30,00))]
df2 = df[(df['time'] >= datetime.datetime(2017,1,26,13,00,00))]
df1['time'] = [d.time() for d in df1['time']]
df2['time'] = [d.time() for d in df2['time']]

# max_fre = max(df['frequency']) #为了画图时坐标一致
# max_price = int(max(df['low']) * 1.05)
# min_price = int(min(df['low']) / 1.05)


fig = plt.figure(1)
fig.set_size_inches(18.5, 10.5)

ax1 = fig.add_subplot(121)   ##先画上午的
ax1.plot_date(df1['time'], df1['low'],linestyle='-', xdate=True, ydate=False, color = 'b')
datemin = datetime.datetime.strptime('2015-1-10 9:30:00', "%Y-%m-%d %H:%M:%S").time()
datemax = datetime.datetime.strptime('2015-1-10 15:00:00', "%Y-%m-%d %H:%M:%S").time()
a0 = datetime.datetime.strptime('2015-1-10 10:00:00', "%Y-%m-%d %H:%M:%S").time()
a1 = datetime.datetime.strptime('2015-1-10 10:30:00', "%Y-%m-%d %H:%M:%S").time()
a2 = datetime.datetime.strptime('2015-1-10 11:00:00', "%Y-%m-%d %H:%M:%S").time()
a3 = datetime.datetime.strptime('2015-1-10 11:30:00', "%Y-%m-%d %H:%M:%S").time()
labels = [datemin, a0, a1, a2, a3]
ax1.set_xlim(datemin, a3)
ax1.xaxis.set_ticks(labels) #to get a tick every 15 minutes
ax1.set_ylim(-0.11, 0.05)  #让两张图的坐标轴一致
ax1.set_ylabel(node+'_return', color='b')
ax1.tick_params(node+'_return', colors='b')

ax2 = ax1.twinx()
ax2.plot_date(df1['time'], df1['successor_decline'],linestyle='--', xdate=True, ydate=False, color = 'r')
ax2.set_ylim(-0.11, 0.05)  #让两张图的坐标轴一致
plt.xlabel('time')
fig.autofmt_xdate() #automatically rotates dates appropriately for you figure.


ax1 = plt.subplot(122)  #下午的
ax1.plot_date(df2['time'], df2['low'],linestyle='-', xdate=True, ydate=False, color = 'b')
datemax = datetime.datetime.strptime('2015-1-10 15:00:00', "%Y-%m-%d %H:%M:%S").time()
a4 = datetime.datetime.strptime('2015-1-10 13:00:00', "%Y-%m-%d %H:%M:%S").time()
a5 = datetime.datetime.strptime('2015-1-10 13:30:00', "%Y-%m-%d %H:%M:%S").time()
a6 = datetime.datetime.strptime('2015-1-10 14:00:00', "%Y-%m-%d %H:%M:%S").time()
a7 = datetime.datetime.strptime('2015-1-10 14:30:00', "%Y-%m-%d %H:%M:%S").time()
labels = [datemin, a0, a1, a2, a3, a4, a5, a6, a7,datemax]
ax1.set_xlim(a4, datemax)
ax1.set_ylim(-0.11, 0.05)  #让两张图的坐标轴一致
ax1.xaxis.set_ticks(labels) #to get a tick every 15 minutes

ax2 = ax1.twinx()
ax2.plot_date(df2['time'], df2['successor_decline'],linestyle='--', xdate=True, ydate=False, color = 'r')
ax2.set_ylabel('successors_average_return', color='r')
ax2.tick_params('successors_average_return', colors='r')
ax2.set_ylim(-0.11, 0.05)  #让两张图的坐标轴一致
plt.xlabel('time')
fig.autofmt_xdate() #automatically rotates dates appropriately for you figure.


plt.savefig(f_name + "_time.png")
plt.clf()

plt.figure()
plt.scatter(df['low'], df['successor_decline'])
plt.xlabel(node +'_return')
plt.ylabel(node+'_successors_average_return')
plt.savefig(f_name +"_successor_decline.png")
plt.clf()
