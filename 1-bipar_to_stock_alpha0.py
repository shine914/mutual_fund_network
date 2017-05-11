
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
	
os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
#nx.write_weighted_edgelist(V,"2015-1.edgelist")
nx.write_graphml(V,"2015-1.graphml")

#graph_basic_info = pd.DataFrame()
#weight_info = pd.DataFrame()

#graph_basic_info = graph_structure_info(V, '2015-1', graph_basic_info, '0')
#graph_basic_info.to_excel('graph_basic_info-by_year.xlsx', sheet_name='Sheet1')

w = weight(V)
#weight_info = weight_info_by_year(V, '2015-1', weight_info)
#weight_info.to_excel('weight_info-by_year.xlsx', sheet_name='Sheet1')

out_degree = [d for n, d in V.out_degree()]  # degree sequence

in_degree = [d for n, d in V.in_degree()]  # degree sequence


hist_log_bin(in_degree, 'in_degree', 1, 2710, 50)
hist_log_bin(out_degree, 'out_degree', 1, 2710, 50)

hist_log_bin(w, 'weight', 1, 3000000, 50)


print '-------end-----------------'