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
from my_network import remove_edges, read_edgelist, weight, all_graph_info
from my_plots import density, hist_plt,hist_log_bin
import matplotlib.pyplot as plt
from my_dataframes import drop_duplicate, drop_emptycell, stock_status,summary



os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
V = nx.read_graphml('2015-1_95.graphml')

os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')
t_list = ['601318.SH', '601166.SH', '600036.SH', '600016.SH', '600030.SH','600000.SH','300059.SZ','002183.SZ','000002.SZ']

df = pd.read_excel('node_degree_weight-manual分列.xlsx', 'Sheet1')
df = pd.DataFrame(df)


for node1 in t_list:
    suc_df = pd.DataFrame()
    if node1 in V.nodes():  # 确认跌停的这只股票在网络中
        d = list(V.successors(node1))  # 一定要list一下，不然格式不对
        #print d
        for node2 in d:
            df1 = df[df.node == node2]
            #print df1
            suc_df = suc_df.append(df1)
    summary(node1+'successors_in_degree', suc_df['in_degree'])
    summary(node1+'successors_out_degree', suc_df['out_degree'])
    hist_log_bin(suc_df['in_degree'], node1+'successors_in_degree', 1 , 400, 50)
    hist_log_bin(suc_df['out_degree'], node1+'successors_out_degree', 1 , 2400, 50)



