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
remove certain edges
"""
def remove_edges(G, w_threshold):
	for u,v,data in G.edges(data=True):
		if data['weight'] <= float(w_threshold):
			G.remove_edge(u,v)
	print "remove_edges is done"
	return G



"""
compute the graph basic information, and add the graph infro to a dataframe.
"""
def graph_structure_info(G, filename, df):
	df1 = pd.DataFrame(data={'date': [filename],
							'nodes_num': [nx.number_of_nodes(G)],
							'edges_num': [nx.number_of_edges(G)],
							'density':[nx.density(G)]
							})
	#print df1
	df = df.append(df1)
	print "this year graph_basic_info is done"
	return df
	

"""
weight sequence
"""	
def weight(G):
	w = nx.get_edge_attributes(G, 'weight')
	w = list(w.values())
	return w

"""
read the edgelist
"""
def read_edgelist(workingfile):
	G = nx.MultiDiGraph()
	G = nx.read_weighted_edgelist(workingfile,create_using=nx.MultiDiGraph())
	print "reading G is done"
 	return G

"""
run
"""	
filename = '2015-1'
#alpha = '1925'	
#alpha = '0'	

w_threshold = 0.95

"""
原始网络
"""	
os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
V = read_edgelist('2015-1.edgelist')

w = weight(V)
w = pd.DataFrame(w)
beta = w.quantile(q = w_threshold )
print beta

V = remove_edges(V, beta)
print "after remove edges, stock network nodes and edges are ",nx.number_of_nodes(V), nx.number_of_edges(V)
#remove = [node for node,degree in V.degree() if degree == 0]
#V.remove_nodes_from(remove)
#print "after remove nodes with no degree, stock network nodes and edges are ",nx.number_of_nodes(V), nx.number_of_edges(V)

os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')
degree_out  = sorted([d for n,d in V.out_degree()], reverse=True) # 这里定义了out_degree
degree_out  = pd.DataFrame(degree_out)
degree_out.columns = [beta]
degree_in = sorted([d for n,d in V.in_degree()], reverse=True) # 这里定义了out_degree
degree_in  = pd.DataFrame(degree_in)
degree_in.columns = [beta]
out_summary = degree_out.describe()
in_summary = degree_in.describe()
out_summary.to_excel('degree_out_summary.xlsx', sheet_name='Sheet1')
in_summary .to_excel('degree_in_summary.xlsx', sheet_name='Sheet1')


graph_basic_info = pd.DataFrame()
graph_basic_info = graph_structure_info(V, filename, graph_basic_info)
graph_basic_info.to_excel('graph_basic_info-by_year.xlsx', sheet_name='Sheet1')
print "graph_basic_info-by_year is done"


print '-------end-----------------'

