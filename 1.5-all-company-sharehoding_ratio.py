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


def f(group):
    return pd.DataFrame({u'持股占流通股比': group[u'持股占流通股比(%)'].sum()})


"""
每只股票中的基金公司投资占比
"""
# os.chdir('E:/financial_network/data/original/dropna_duplicates')
#
# df = pd.read_excel('2015-1.xlsx', 'Sheet1')
# df = pd.DataFrame(df)
#
# df = df[[u'股票代码', u'持股占流通股比(%)']]
# grouped = df.groupby([u'股票代码'], as_index=False)
# df = grouped.aggregate(np.sum)
# # print df
# #
# # grouped_stock = df.groupby([u'股票代码'])
# # ratio = grouped_stock.apply(f)
# # df = pd.concat([df, ratio], axis=1)
#
# os.chdir('E:/financial_network/data')
# df.to_excel('all_company_shareholding_ratio.xlsx', sheet_name='Sheet1')
#
# s = df[u'持股占流通股比(%)'].describe()
# print s
print'-----------'
os.chdir('/Users/shine/work_hard/financial_network/data')
df = pd.read_excel('all_company_shareholding_ratio.xlsx', sheet_name='Sheet1')
x = df[u'持股占流通股比(%)']
plt.hist(x, facecolor='green', alpha=0.75)
plt.xlabel('shareholding ratio')
# plt.ylabel('density')
plt.savefig('shareholding ratio.png')
plt.clf()
