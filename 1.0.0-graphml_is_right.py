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




os.chdir('/Users/shine/work_hard/financial_network/data/threshold/transformed')
file = glob.glob('2015-1*.xlsx') #for workingfile in filelist
print file,'is working now'
df = pd.ExcelFile(file[0])
df = pd.read_excel(df, 'Sheet1')
df = pd.DataFrame(df)

G, top_nodes = bipartite_graph(df)
V = stock_network(G,top_nodes)
V = remove_edges(V, 0)
print '----'

os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')
#t_list = ['601318.SH', '601166.SH', '600036.SH', '600016.SH', '600030.SH','600000.SH','300059.SZ','002183.SZ','000002.SZ']

w = weight(V)
w = pd.DataFrame(w)
beta = w.quantile(q = 0.95)
print beta

V = remove_edges(V, beta)

print 'nodes_num', nx.number_of_nodes(V)
print 'edges_num', nx.number_of_edges(V)


os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
V = nx.read_graphml('2015-1_95.graphml')
print 'nodes_num', nx.number_of_nodes(V)
print 'edges_num', nx.number_of_edges(V)


