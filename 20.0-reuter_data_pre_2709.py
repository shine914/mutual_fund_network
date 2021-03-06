
# -*- coding: utf-8 -*-

#avoid chinese mess
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import glob,os
import xlrd
import networkx as nx
import pandas as pd
import numpy as np
from my_network import remove_edges, read_edgelist, weight, all_graph_info, bipartite_graph,stock_network,graph_structure_info
from my_plots import density, hist_plt,hist_log_bin
import matplotlib.pyplot as plt
from my_dataframes import drop_duplicate, drop_emptycell, stock_status,summary


# os.chdir('E:/financial_network/data/original/thomson_reuters/2956_end_of_day')

# os.chdir('E:/financial_network/data/original/2709end_of_day_open_close')
#FileList = glob.glob('111-2015-06'+'*.csv')
# FileList = glob.glob('114-2015-06'+'*.csv')
# FileList = glob.glob('115-2015-07'+'*.csv')
# print 'there are', len(FileList),'files'

# print FileList
#
# t_df = pd.DataFrame()
# for workingfile in FileList:
#     print workingfile
#     # filename = workingfile.split('07-')[1]
#     filename = workingfile.split('06-')[1]
#     filename = filename.split('.c')[0]
#     print filename, 'is the filename'
#
#     df = pd.read_csv(workingfile)  # 保证了只有一个header，sheet名称是wind资讯
#     df = pd.DataFrame(df)
#
#     df['return'] = (df['Last']-df['Open'])/df['Open']
#     #print df
#
#     df = df.loc[:,['Date[L]','return']]
#     df = df.transpose()
#     df = df.rename(index={'return': filename})
#     df.columns = df.iloc[0]
#     df = df.reindex(df.index.drop('Date[L]'))
#     #print df
#     t_df = t_df.append(df)
#     # print t_df



# # os.chdir('E:/financial_network/data')
# # t_df.to_csv('returns_reuters_2709-july.csv')
#
# os.chdir('E:/financial_network/data/2956_end_of_day')
# t_df.to_csv('2956_end_of_day-june.csv')
# # t_df.to_csv('2956_end_of_day-july.csv')

# '''合并6月7月'''
# os.chdir('E:/financial_network/data')
# df1 = pd.read_csv('returns_reuters_2709-july.csv')
# df2 = pd.read_csv('returns_reuters_2709-jun.csv')
#
# df = pd.merge(df1, df2, on='node')
# df.to_csv('today_returns_reuters_2709.csv')

'''合并6月7月'''
os.chdir('E:/financial_network/data/2956_end_of_day')
df1 = pd.read_csv('2956_end_of_day-july.csv')
df2 = pd.read_csv('2956_end_of_day-june.csv')

df = pd.merge(df1, df2, on='node')
df.to_csv('today_returns_reuters_2709.csv')
