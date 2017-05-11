# -*- coding: utf-8 -*-

#avoid chinese mess
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import glob,os

import networkx as nx
import pandas as pd
import numpy as np

from my_network import remove_edges, weight, read_edgelist

w_threshold = 0.95

os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
V = read_edgelist('2015-1.edgelist')

w = weight(V)
w = pd.DataFrame(w)
beta = w.quantile(q = w_threshold)
print beta

V = remove_edges(V, beta)
print "after remove edges, stock network nodes and edges are ",nx.number_of_nodes(V), nx.number_of_edges(V)
#nx.write_graphml(V,  "2015-1_95.graphml")

os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')

####
out_degree =np.asarray([d for n,d in V.out_degree()]) # degree sequence
in_degree =np.asarray([d for n,d in V.in_degree()]) # degree sequence
#density('2015-1_95_out_degree', out_degree)
#density('2015-1_95_in_degree', in_degree)
#####
threshold = 95
print len(out_degree)
out_degree_th = np.percentile(out_degree, threshold)
print out_degree_th
out_list = [n for n,d in V.out_degree() if d >= out_degree_th]
print len(out_list)

from my_plots import hist_log_bin

hist_log_bin(out_degree, 'out_degree', 1, 2301, 50) #度为0的情况，没法在log_bin图中显示
hist_log_bin(in_degree, 'in_degree', 1, 400, 50)

#in_degree_th = in_degree.quantile(q = threshold)

#top_out_degree = [i for i in out_degree if i >= out_degree_th]
#top_in_degree = [i for i in in_degree if i >= in_degree_th]
