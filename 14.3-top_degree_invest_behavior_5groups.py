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
import seaborn as sns


os.chdir('E:/financial_network/data/threshold/transformed')
df1 = pd.read_excel('2015-1_transformed.xlsx', 'Sheet1')
df1 = pd.DataFrame(df1)
df1 = drop_duplicate(df1)
df1 = drop_emptycell(df1)


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

top_stocks = ['601318.SH','601166.SH','600000.SH','000002.SZ']

# df_1 = pd.DataFrame()
# company = list(df1[u'管理公司'].unique()) #考察全部公司

for j in range(0,4):
    df_1 = pd.DataFrame()
    a = df1[df1[u'股票代码'] == top_stocks[j]]
    company = list(a[u'管理公司'])

    for i in company:
        a = df1[df1[u'管理公司'] == i]
        a = list(a[u'股票代码'])
        a1 = len([w for w in list(k0) if w in a])
        a2 = len([w for w in list(k1) if w in a])
        a3 = len([w for w in list(k2) if w in a])
        a4 = len([w for w in list(k3) if w in a])
        a5 = len([w for w in list(k4) if w in a])

        df_11 = pd.DataFrame({'company': [i],
                              'out_degree=0': [float(a1) / len(list(k0))],
                              'k1': [float(a2) / len(list(k1))],
                              'k2': [float(a3) / len(list(k2))],
                              'k3': [float(a4) / len(list(k3))],
                              'k4': [float(a5) / len(list(k4))]
                              })
        df_1 = df_1.append(df_11)

    os.chdir('E:/financial_network/data/threshold/stock_alpha/0.95_2015-1/investment_behavior')
    df_1.to_excel(top_stocks[j]+'_4group_number_ratio.xlsx', sheet_name='Sheet1')
    print '--------------'
