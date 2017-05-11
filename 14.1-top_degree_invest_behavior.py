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
df_3 = pd.DataFrame()
for j in range(0,4):
    a = df1[df1[u'股票代码'] == top_stocks[j]]
    company = list(a[u'管理公司'])
    company_num = len(company) #多少个公司投资了股票j
    total_invest = np.sum(a[u'持股市值(万元)'])#这些公司总共在股票j上投资了多少钱

    total_else = 0
    num_else = 0
    for i in company:
        b = df1[df1[u'管理公司'] == i ]
        total_else1 = np.sum(b[u'持股市值(万元)'])
        total_else = total_else + total_else1
        num_else1 = len(b)-1
        num_else = num_else +num_else1
    df_31 = pd.DataFrame({'stock': [top_stocks[j]],
                          'invest_company_num': [company_num],
                          'how_much_total': [total_invest],
                          'how_much_those_company_invest_on_else':[total_else],
                          'how_many_on_else':[num_else]
                          })
    df_3 = df_3.append(df_31)
os.chdir('E:/financial_network/data/threshold/stock_alpha/0.95_2015-1/investment_behavior')
df_3.to_excel('companys.xlsx', 'Sheet1')
# bb = pd.DataFrame()
# cc = pd.DataFrame()
# for i in company:
#     print i
#     b = df1[df1[u'管理公司'] == i ]
#     print len(b)
    # b = b.merge(df2, on=[u'股票代码'], how='left')
    # print len(b)
    # print b
    # b = b.sort_values(by='out_degree')
    # bb = bb.append(b)
    # c = b[u'占股票投资市值比']
    # cc = pd.concat([cc, c], axis=1)
# bb.to_excel('000002_company_invest_ratio_network-info.xlsx','Sheet1')
# cc.to_excel('000002_company_invest_ratio.xlsx','Sheet1')

