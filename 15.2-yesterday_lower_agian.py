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

df_1 = pd.DataFrame()
for i in range(1,18):
    day = '%d'%i
    a1 = len([w for w in lower[i-1] if w in lower[i]])
    df_11 = pd.DataFrame({'date': [day],
                          'lower_today':[len(lower[i])],
                          'is_lower_yesterday': [a1]
                          })
    df_1 = df_1.append(df_11)

os.chdir('E:/financial_network/data/threshold/stock_alpha/0.95_2015-1')
df_1.to_excel('yesterday_lower_again.xlsx', sheet_name='Sheet1')
