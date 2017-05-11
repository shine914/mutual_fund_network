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
print FileList
print "FileList length is", len(FileList)  # 共20个文件
for workingfile in FileList:
    print workingfile, 'is working now'
    filename = workingfile.split('.')[0]
    print 'filename is', filename
    os.chdir('E:/financial_network/data/status_wind_612_710')
    status = pd.ExcelFile(workingfile)
    status = pd.read_excel(status, 'Wind资讯'.decode('utf-8'))
    status = pd.DataFrame(status)
    status.columns = ['source', 'stock_name', 'limit_ori', 'sus_reason', 'sus_days', 'sus_ori']
    status['limit'] = status.apply(lambda df: stock_status(df, 'limit_ori', 'sus_ori'), axis=1)
    df = status[['source', 'stock_name', 'limit']]

    df_lower = df[df.limit == 'lower_limit']
    lower.append(df_lower['source'].tolist())

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