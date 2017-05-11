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
from my_plots import density
import matplotlib.pyplot as plt



os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
V = nx.read_graphml('2015-1_95.graphml')

os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')
file = glob.glob('node_degree_weight-manual分列.xlsx') #for workingfile in filelist
df = pd.ExcelFile(file[0])
df = pd.read_excel(df, 'Sheet1')
df = pd.DataFrame(df)

z = df['weight_source_sum']

info = pd.DataFrame()
for threshold in np.arange(0,1,0.01):
    beta = z.quantile(q=threshold)
    sub_list = []
    for i, node in enumerate(df['node']):
        if z[i]>=beta:
            sub_list.append(node)
    Z = V.subgraph(sub_list)
    info = all_graph_info(Z, '0', threshold, beta, info)
    print '--------'

info.to_excel('top_weight_source_sum_subgraph_info.xlsx', sheet_name='Sheet1')