# -*- coding: utf-8 -*-

#avoid chinese mess
import sys 
reload(sys)
sys.setdefaultencoding('utf8') 

import glob,os

import xlrd

import networkx as nx
from networkx.algorithms import bipartite

import numpy as np
import pandas as pd


import pandas as pd
import numpy as np
import glob,os
import math
from math import log10

import collections
from collections import Counter
import matplotlib.pyplot as plt
import scipy
from scipy import stats, integrate #draw density line
import seaborn as sns


"""
read the edgelist
"""
def read_edgelist(workingfile):
	G = nx.MultiDiGraph()
	G = nx.read_weighted_edgelist(workingfile,create_using=nx.MultiDiGraph())
	print "reading G is done"
 	return G

"""
compute the graph basic information, and add the graph infro to a dataframe.
"""
def graph_structure_info(G, filename, w_threshold, beta, df):
	degree_out  = pd.DataFrame([d for n,d in G.out_degree()]) # 这里定义了out_degree
	degree_in = pd.DataFrame([d for n,d in G.in_degree()]) # 这里定义了out_degree
	df1 = pd.DataFrame(data={'year': [filename],
							'threshold':[w_threshold],
							#'weight_threhold':[beta],
							#'nodes_num': [nx.number_of_nodes(G)],
							#'edges_num': [nx.number_of_edges(G)],
							#'density':[nx.density(G)],
							#'number_weakly_connected_components':[nx.number_weakly_connected_components(G)],
							#'number_strongly_connected_components':[nx.number_strongly_connected_components(G)],
							#'size_largest_strongly_connected_components':[len(max(nx.strongly_connected_components(G), key=len))],
							#'size_largest_weakly_connected_components':[len(max(nx.weakly_connected_components(G), key=len))],
							#'weights_sum':[sum(list((nx.get_edge_attributes(G, 'weight')).values()))],
							#'ave_degree_out':[degree_out.mean()], #入度均值=出度均值
							'ave_degree_in':[degree_in.mean()]
							})
							#'ave_clustering_coeffient':[nx.average_clustering(G)],
							#'ave_shortest_path_length':[nx.average_shortest_path_length(G)]
	#print df1
	df = df.append(df1)
	print "this year graph_basic_info is done"
	return df

"""
remove certain edges
"""
def remove_edges(G, w_threshold):
	for u,v,data in G.edges(data=True):
		if data['weight'] <= float(w_threshold):
			G.remove_edge(u,v)
	print "remove_edges is done"
	return G
"""
weight sequence
"""	
def weight(G):
	w = nx.get_edge_attributes(G, 'weight')
	w = list(w.values())
	return w

"""
run
"""	
	
os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
V = read_edgelist('2015-1.edgelist')
w = weight(V)
w = pd.DataFrame(w)

graph_basic_info = pd.DataFrame()
filename = '2015-1'

for w_threshold in np.arange(0,1,0.01):
	beta = w.quantile(q = w_threshold)
	print beta
	G = remove_edges(V, beta)
	print "after remove edges, stock network nodes and edges are ",nx.number_of_nodes(G), nx.number_of_edges(G)
	remove = [node for node,degree in V.degree() if degree == 0]
	G.remove_nodes_from(remove)
	graph_basic_info = graph_structure_info(G, filename, w_threshold, beta, graph_basic_info)

#graph_basic_info.to_excel('pick_threshold.xlsx', sheet_name='Sheet1')
graph_basic_info.to_excel('pick_threshold-del_degree0_0.01.xlsx', sheet_name='Sheet1')
print "graph_basic_info is done"

