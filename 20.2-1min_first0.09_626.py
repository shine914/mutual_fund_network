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

'''昨日收盘价'''
os.chdir('E:/financial_network/data/2709_return')
df2 = pd.read_csv('day_last_price_reuters_2709.csv')


'''今日分钟价'''
os.chdir('E:/financial_network/data/original/thomson_reuters/2709_intraday_1min_open_close')
FileList1 = glob.glob('111-2015-06'+'*.csv')
#print 'there are', len(FileList1),'files'


t_df = pd.DataFrame()
for workingfile in FileList1:
    #print workingfile
    filename = workingfile.split('06-')[1]
    filename = filename.split('.c')[0]
    print filename, 'is the filename'

    os.chdir('E:/financial_network/data/original/thomson_reuters/2709_intraday_1min_open_close')
    df1 = pd.read_csv(workingfile)
    df1 = pd.DataFrame(df1)

    today = df1[df1['Date[L]'] == '26-JUN-2015']
    today = today.loc[:,['#RIC','Time[L]','Last']]
    #print 'toaday 1min ',len(today)
    today['return'] = np.nan

    yesterday = df2[df2['source'] == filename]['25-Jun-15']
    yesterday = yesterday.iloc[0]
    #print 'yesterday',yesterday

    for i in range(0, len(today)):
        # print today.iloc[i, 5]
        today.iloc[i, 3] = (today.iloc[i, 2]-yesterday)/yesterday
        if today.iloc[i, 3]<=-0.09:
            df = pd.DataFrame(today.iloc[i,:])
            df = df.transpose()
            # print df
            t_df = pd.concat([t_df, df])
            break



    # print 'return',today
    # t_df = pd.concat([t_df, today])
    # print t_df
#
os.chdir('E:/financial_network/data')
t_df.to_csv('626_2709-first0.09.csv')

# '''合并6月7月'''
# os.chdir('E:/financial_network/data')
# df1 = pd.read_csv('returns_reuters_2709-july.csv')
# df2 = pd.read_csv('returns_reuters_2709-jun.csv')
#
# df = pd.merge(df1, df2, on='node')
# df.to_csv('today_returns_reuters_2709.csv')




