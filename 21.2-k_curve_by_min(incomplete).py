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

import datetime

delta = 1
date = '619'
filename = '619_%r_min'%delta

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


os.chdir('E:/financial_network/data/threshold/stock_alpha/0.95_2015-1')
df = pd.read_excel('node_degree_weight-back_up.xlsx', 'Sheet1')
df = pd.DataFrame(df)
df = df[df.out_degree != 0]
a = df['out_degree']
df_1 = pd.DataFrame()
for threshold in np.arange(0,1,0.01):
    beta = a.quantile(q = threshold )
    df_11 = pd.DataFrame({'percentile': [threshold],
                          'out_degree': [beta]
                          })
#    df_1 = df_1.append(df_11)
#df_1.to_excel('out_degree_percentile.xlsx', sheet_name='Sheet1')

#-----
lower = []
os.chdir('E:/financial_network/data/status_wind_612_710')
FileList = glob.glob('*.xlsx')  # for workingfile in filelist


print '-------and-----------------'

os.chdir('E:/financial_network/data/threshold/stock_alpha/0.95_2015-1')
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


df_1 = pd.DataFrame()
for i in range(0,18):
    day = '%d'%i
    a = lower[i]
    a1 = len([w for w in list(k0) if w in a])
    a2 = len([w for w in list(k1) if w in a])
    a3 = len([w for w in list(k2) if w in a])
    a4 = len([w for w in list(k3) if w in a])
    a5 = len([w for w in list(k4) if w in a])
    #df_11 = pd.DataFrame({'date': [day],
	#					'k0': [a1],
	#					'k1': [a2],
		##				'k2': [a3],
			#			'k3': [a4],
           #             'k4':[a5]
			#			})
    df_11 = pd.DataFrame({'date': [day],
                          'k0': [float(a1)/len(list(k0))],
                          'k1': [float(a2)/len(list(k1))],
                          'k2': [float(a3)/len(list(k2))],
                          'k3': [float(a4)/len(list(k3))],
                          'k4': [float(a5)/len(list(k4))]
                          })
    df_1 = df_1.append(df_11)

#df_1.to_excel('k-curve_lower_ratio_by_date.xlsx', sheet_name='Sheet1')
df_1.to_excel('k-curve_lower_each_group_ratio_by_date.xlsx', sheet_name='Sheet1')