
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


# os.chdir('E:/financial_network/data/original/thomson_reuters/2709end_of_day')
os.chdir('E:/financial_network/data/original/thomson_reuters/2956_end_of_day')

# FileList1 = glob.glob('111-2015-06'+'*.csv')
# FileList2 = glob.glob('111-2015-07'+'*.csv')

FileList1 = glob.glob('114-2015-06'+'*.csv')
FileList2 = glob.glob('115-2015-07'+'*.csv')
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

    workingfile2 = '115-2015-07-'+filename+'.csv'
    print workingfile2
    # if workingfile2 in FileList2:
    df2 = pd.read_csv(workingfile2)
    df2 = pd.DataFrame(df2)

    df = df1.append(df2)
    #print df
    # df = df.loc[:, ['Date[L]', 'Last']]
    df = df.loc[:, ['Date[L]', 'Low']]
    df = df.transpose()
    # df = df.rename(index={'Last': filename})
    df = df.rename(index={'Low': filename})
    df.columns = df.iloc[0]
    df = df.reindex(df.index.drop('Date[L]'))

    t_df =  pd.concat([t_df, df])
    # print t_df

# os.chdir('E:/financial_network/data/2709_return')
# t_df.to_csv('day_last_price_reuters_2709.csv')
os.chdir('E:/financial_network/data/2956_end_of_day')
# t_df.to_csv('day_last_price_reuters_2956.csv')
t_df.to_csv('day_low_price_reuters_2956.csv')