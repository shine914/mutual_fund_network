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




'''网络中的点的出入度信息，分为5组'''
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

'''biparite网络的原始数据'''
os.chdir('E:/financial_network/data/threshold/transformed')
df1 = pd.read_excel('2015-1_transformed.xlsx', 'Sheet1')
df1 = pd.DataFrame(df1)
df1 = drop_duplicate(df1)
df1 = drop_emptycell(df1)

df_1 = pd.DataFrame()
company = list(df1[u'管理公司'].unique())

def f(df, list):
    dd = df[df[u'股票代码'].isin(list)]
    dd = dd[u'持股市值(万元)']
    return dd


for i in company:
    b = df1[df1[u'管理公司'] == i]
    s = np.sum(b[u'持股市值(万元)']) #该公司的投资总额
    a = list(b[u'股票代码'])
    a1 = [w for w in list(k0) if w in a]
    a2 = [w for w in list(k1) if w in a]
    a3 = [w for w in list(k2) if w in a]
    a4 = [w for w in list(k3) if w in a]
    a5 = [w for w in list(k4) if w in a]

    df_11 = pd.DataFrame({'company': [i],
                          'out_degree=0': [np.sum(f(b, a1))/s],
                          'k1': [np.sum(f(b, a2))/s],
                          'k2': [np.sum(f(b, a3))/s],
                          'k3': [np.sum(f(b, a4))/s],
                          'k4': [np.sum(f(b, a5))/s]
                          })

    df_1 = df_1.append(df_11)

os.chdir('E:/financial_network/data/threshold/stock_alpha/0.95_2015-1/investment_behavior')
df_1.to_excel('all_company__each_4group_value_ratio.xlsx', sheet_name='Sheet1')
print '--------------'
