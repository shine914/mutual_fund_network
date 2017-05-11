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

f_name = date + '_ %d min' %delta

'''end of day收盘价,最低价'''
os.chdir('/Users/shine/work_hard/financial_network/data/2956_sec')
last = pd.read_csv('day_last_price_reuters_2956.csv')
last = pd.DataFrame(last)

'''出度信息，分为5组'''
os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')
df = pd.read_excel('node_degree_weight-back_up.xlsx', 'Sheet1')
df = pd.DataFrame(df)

q1 = 451 #30%
q2 = 946 #60%
q3 = 1490 #90%

k0 = df[df.out_degree == 0]['nodes']

k1 = df[df.out_degree <= q1]
k1 = k1[k1.out_degree > 0]['nodes']

k2 = df[df.out_degree <= q2]
k2 = k2[k2.out_degree > q1]['nodes']

k3 = df[df.out_degree <= q3]
k3 = k3[k3.out_degree > q2]['nodes']

k4 = df[df.out_degree >= q3]['nodes']


'''大出度股票下跌幅度+其他股票的下跌幅度，to csv'''
os.chdir('/Users/shine/work_hard/financial_network/data/2956_sec/successor_decline_10min')
ave_decline = pd.read_csv(date+'10min_all_node_decline.csv')
df = pd.DataFrame(ave_decline)

g1 = df.loc[:, lambda df: list(k0)]
g2 = df.loc[:, lambda df: list(k1)]
g3 = df.loc[:, lambda df: list(k2)]
g4 = df.loc[:, lambda df: list(k3)]
g5 = df.loc[:, lambda df: list(k4)]


g1 = pd.DataFrame(g1.mean(axis = 1))
g1.columns = ['g1_successor_decline']
# print g1
g2 = pd.DataFrame(g2.mean(axis = 1))
g2.columns = ['g2_successor_decline']

g3 = pd.DataFrame(g3.mean(axis = 1))
g3.columns = ['g3_successor_decline']

g4 = pd.DataFrame(g4.mean(axis = 1))
g4.columns = ['g4_successor_decline']

g5 = pd.DataFrame(g5.mean(axis = 1))
g5.columns = ['g5_successor_decline']

g = pd.concat([g1,g2,g3,g4,g5], axis = 1)
g['time'] = df['time']
g.to_csv(f_name+'_ave_k_curve.csv')


'''plot'''
os.chdir('/Users/shine/work_hard/financial_network/data/2956_sec/successor_decline_10min')
g = pd.read_csv(f_name+'_ave_k_curve.csv')
df = pd.DataFrame(g)

df['time'] = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in df['time']]
df1 = df[(df['time'] <= datetime.datetime(2017,1,26,11,30,00))]
df2 = df[(df['time'] >= datetime.datetime(2017,1,26,13,00,00))]
df1['time'] = [d.time() for d in df1['time']]
df2['time'] = [d.time() for d in df2['time']]

print df1


fig = plt.figure(1)
fig.set_size_inches(18.5, 10.5)

ax1 = fig.add_subplot(121)   ##先画上午的
datemin = datetime.datetime.strptime('2015-1-10 9:30:00', "%Y-%m-%d %H:%M:%S").time()
a0 = datetime.datetime.strptime('2015-1-10 10:00:00', "%Y-%m-%d %H:%M:%S").time()
a1 = datetime.datetime.strptime('2015-1-10 10:30:00', "%Y-%m-%d %H:%M:%S").time()
a2 = datetime.datetime.strptime('2015-1-10 11:00:00', "%Y-%m-%d %H:%M:%S").time()
a3 = datetime.datetime.strptime('2015-1-10 11:30:00', "%Y-%m-%d %H:%M:%S").time()
labels = [datemin, a0, a1, a2, a3]
ax1.set_xlim(datemin, a3)
ax1.xaxis.set_ticks(labels) #to get a tick every 15 minutes
ax1.set_ylim(-0.11, -0.02)  #让两张图的坐标轴一致
ax1.set_ylabel('return', color='b')
ax1.tick_params('return', colors='k')

ax1.plot_date(df1['time'], df1['g1_successor_decline'],linestyle='-', xdate=True, ydate=False, color = 'b')
ax1.plot_date(df1['time'], df1['g2_successor_decline'],linestyle='--', xdate=True, ydate=False, color = 'g')
ax1.plot_date(df1['time'], df1['g3_successor_decline'],linestyle='--', xdate=True, ydate=False, color = 'y')
ax1.plot_date(df1['time'], df1['g4_successor_decline'],linestyle='--', xdate=True, ydate=False, color = 'c')
ax1.plot_date(df1['time'], df1['g5_successor_decline'],linestyle='--', xdate=True, ydate=False, color = 'k')

plt.xlabel('time')
fig.autofmt_xdate() #automatically rotates dates appropriately for you figure.


ax1 = plt.subplot(122)  #下午的
datemax = datetime.datetime.strptime('2015-1-10 15:00:00', "%Y-%m-%d %H:%M:%S").time()
a4 = datetime.datetime.strptime('2015-1-10 13:00:00', "%Y-%m-%d %H:%M:%S").time()
a5 = datetime.datetime.strptime('2015-1-10 13:30:00', "%Y-%m-%d %H:%M:%S").time()
a6 = datetime.datetime.strptime('2015-1-10 14:00:00', "%Y-%m-%d %H:%M:%S").time()
a7 = datetime.datetime.strptime('2015-1-10 14:30:00', "%Y-%m-%d %H:%M:%S").time()
labels = [a4, a5, a6, a7,datemax]
ax1.set_xlim(a4, datemax)
ax1.set_ylim(-0.11, -0.02)  #让两张图的坐标轴一致
ax1.xaxis.set_ticks(labels) #to get a tick every 15 minutes

ax1.plot_date(df2['time'], df2['g1_successor_decline'],linestyle='-', xdate=True, ydate=False, color = 'b')
ax1.plot_date(df2['time'], df2['g2_successor_decline'],linestyle='--', xdate=True, ydate=False, color = 'g')
ax1.plot_date(df2['time'], df2['g3_successor_decline'],linestyle='--', xdate=True, ydate=False, color = 'y')
ax1.plot_date(df2['time'], df2['g4_successor_decline'],linestyle='--', xdate=True, ydate=False, color = 'c')
ax1.plot_date(df2['time'], df2['g5_successor_decline'],linestyle='--', xdate=True, ydate=False, color = 'k')

L=plt.legend(loc='upper right')
L.get_texts()[0].set_text('out degree = 0')
L.get_texts()[1].set_text('0 < out degree <= 30 quantile')
L.get_texts()[2].set_text('30 quantitle < out degree <= 60 quantile')
L.get_texts()[3].set_text('60 quantitle < out degree <= 90 quantile')
L.get_texts()[4].set_text('90 quantitle < out degree')

plt.xlabel('time')
fig.autofmt_xdate() #automatically rotates dates appropriately for you figure.

plt.savefig(f_name + "_time.png")
plt.clf()
