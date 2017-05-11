
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


os.chdir('E:/financial_network/data/original/2709end_of_day_open_close')

FileList1 = glob.glob('111-2015-06'+'*.csv')
FileList2 = glob.glob('111-2015-07'+'*.csv')
print 'there are', len(FileList1),'files'
print 'there are', len(FileList2),'files'


t_df = pd.DataFrame()
for workingfile in FileList1:
    print workingfile
    filename = workingfile.split('06-')[1]
    filename = filename.split('.c')[0]
    print filename, 'is the filename'

    df1 = pd.read_csv(workingfile)
    df1 = pd.DataFrame(df1)

    workingfile2 = '111-2015-07-'+filename+'.csv'
    print workingfile2
    df2 = pd.read_csv(workingfile2)
    df2 = pd.DataFrame(df2)

    df = df1.append(df2)
    #print df
    df['return'] = np.nan
    for i in range(1, len(df)):
        df.iloc[i, 6] = (df.iloc[i, 5]-df.iloc[i-1, 5])/df.iloc[i-1, 5]
    #print df

    df = df.loc[:,['Date[L]','return']]
    df = df.transpose()
    # print df
    df = df.rename(index={'return': filename})
    df.columns = df.iloc[0]
    df = df.reindex(df.index.drop('Date[L]'))
    # print df
    t_df =  pd.concat([t_df, df])
    # print t_df

os.chdir('E:/financial_network/data')
t_df.to_csv('returns_reuters_2709.csv')

# '''合并6月7月'''
# os.chdir('E:/financial_network/data')
# df1 = pd.read_csv('returns_reuters_2709-july.csv')
# df2 = pd.read_csv('returns_reuters_2709-jun.csv')
#
# df = pd.merge(df1, df2, on='node')
# df.to_csv('today_returns_reuters_2709.csv')
