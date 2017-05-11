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
from my_network import remove_edges, read_edgelist, weight, all_graph_info,bipartite_graph,stock_network
from my_plots import density
import matplotlib.pyplot as plt


os.chdir('/Users/shine/work_hard/financial_network/data/threshold/transformed')
file = glob.glob('2015-1*.xlsx') #for workingfile in filelist
print file,'is working now'
df = pd.ExcelFile(file[0])
df = pd.read_excel(df, 'Sheet1')
df = pd.DataFrame(df)


G, top_nodes = bipartite_graph(df)
V = stock_network(G,top_nodes)
V = remove_edges(V, 0)

w = weight(V)
w = pd.DataFrame(w)
beta = w.quantile(q = 0.1)
alpha = w.quantile(q = 0.9)

print 'before remove any edges', nx.number_of_edges(V)
for u,v,data in V.edges(data=True):
	#if data['weight'] < float(alpha):
    if data['weight'] > float(beta):
	    V.remove_edge(u,v)
print 'after remove edges', nx.number_of_edges(V), nx.number_of_nodes(V)
remove = [node for node,degree in V.degree() if degree == 0]
V.remove_nodes_from(remove)
print 'after remove nodes with degree = 0', nx.number_of_nodes(V)


df = pd.DataFrame() #不是上面的df了
for nodes1,nodes2,data in V.edges(data=True):
    w = 0
    p = 0
    w_df = []
    if nodes1 != nodes2:
        if len(list(nx.common_neighbors(G, nodes1, nodes2))) != 0:
            for nbr in nx.common_neighbors(G, nodes1, nodes2):
                p = p + 1
                w1 = G[nodes1][nbr]['weight']
                #print w1
                w = w + w1
                w_df.append(w1)
            #print w_df
            df1 = pd.DataFrame({'node1':[nodes1],
                                'node2':[nodes2],
                                'weight': [w],
                                'num_of_nbrs': [p],
                                'ave_nbr_weight': [float(w)/p],
                                'var_nbr_weight': [np.var(w_df)]
                                })

            df = df.append(df1)
            print '--------'

os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
df.to_csv('bipar_nbr_weight0.1.csv')