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

df = pd.DataFrame()
for threshold in np.arange(0,2301,1):
    sub_list = []
    for node, d in V.out_degree():
        if d >= threshold:
            sub_list.append(node)
    Z = V.subgraph(sub_list)
    df = all_graph_info(Z, '0', threshold, 0, df)
    print '--------'

df.to_excel('top_out_degree_subgraph_info.xlsx', sheet_name='Sheet1')