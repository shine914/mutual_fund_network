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
from my_network import remove_edges, read_edgelist, weight, weight_target_sum, weight_source_sum
from my_plots import density

os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
V = nx.read_graphml('2015-1_95.graphml')
print "stock network nodes and edges are ",nx.number_of_nodes(V), nx.number_of_edges(V)

weight_source_sum = weight_source_sum(V)
weight_target_sum = weight_target_sum(V)


node_list = V.nodes()
df = pd.DataFrame()
for node in node_list:
    df1 = pd.DataFrame(data={'node': [node],
                             'out_degree': [V.out_degree(node)],
                             'in_degree': [V.in_degree(node)],
                             'weight_target_sum': [weight_target_sum[weight_target_sum.source == node]['weight'].tolist()],
                             'weight_source_sum':[weight_source_sum[weight_source_sum.source ==node]['weight'].tolist()] #有空值，所以用list，用value会出错
                             })
    print df1
    df = df.append(df1)


os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')
df.to_excel('node_degree_weight.xlsx', sheet_name='Sheet1')  ##所有股票的出入度、出入强度信息；可以在excel中选择排序




