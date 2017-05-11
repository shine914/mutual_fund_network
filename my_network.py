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
remove certain edges
"""
def remove_edges(G, w_threshold):
	for u,v,data in G.edges(data=True):
		if data['weight'] <= float(w_threshold):
			G.remove_edge(u,v)
	print "remove_edges is done"
	return G


"""
create bipartite graph
"""
def bipartite_graph(df):
    G = nx.Graph()  # stock-institute

    top_nodes = df[u'股票代码']
    # print 'original top nodes', top_nodes
    top_nodes = top_nodes.tolist()
    top_nodes_unique = list(set(top_nodes))

    G.add_nodes_from(df[u'股票代码'], bipartite=0)
    G.add_nodes_from(df[u'管理公司'], bipartite=1)

    G.add_weighted_edges_from([(row[u'股票代码'], row[u'管理公司'], row[u'持股市值(万元)']) for idx, row in df.iterrows()],
                              weight="weight")
    return G, top_nodes_unique

"""
compute the graph basic information, and add the graph infro to a dataframe.
"""
def bipar_graph_structure_info(G, filename, df, alpha):
	df1 = pd.DataFrame(data={'year': [filename],
							'nodes_num': [nx.number_of_nodes(G)],
							'edges_num': [nx.number_of_edges(G)],
							alpha: [alpha]
							})
	#print df1
	df = df.append(df1)
	print "this year graph_basic_info is done"
	return df

"""
projection ls
"""
def stock_network(G, top_nodes):
    V = nx.MultiDiGraph()  # top_nodes network, stocks network

    for nodes1 in top_nodes:
        for nodes2 in top_nodes:
            w = 0
            if nodes1 != nodes2:
                for nbr in nx.common_neighbors(G, nodes1, nodes2):
                    # print G[nodes1][nbr]['weight']
                    w = w + G[nodes1][nbr]['weight']
                V.add_edge(nodes1, nodes2, weight=w)

    return V

"""
compute the graph basic information, and add the graph infro to a dataframe.
"""
def graph_structure_info(G, filename, df, alpha):
	df1 = pd.DataFrame(data={alpha: [alpha],
							'year': [filename],
							'nodes_num': [nx.number_of_nodes(G)],
							'edges_num': [nx.number_of_edges(G)],
							'density':[nx.density(G)]
							})
	#print df1
	df = df.append(df1)
	print "this year graph_basic_info is done"
	return df


"""
degree sequence
"""
def degree_sequence(G):
	d = sorted([d for n,d in G.degree()], reverse=True) # degree sequence
	degree_sequence = np.asarray(d) #turn list into arrary
	return degree_sequence

"""
weight sequence
"""
def weight(G):
	w = nx.get_edge_attributes(G, 'weight')
	w = list(w.values())
	return w

"""
sorted by weight_source_sum
"""
def weight_source_sum(G):
	w = sorted(G.edges(data=True), key=lambda (source,target,data): data['weight'], reverse=True)
	w = pd.DataFrame(w)
	ww = pd.DataFrame(w.loc[:,2].values.tolist())
	w1 = pd.concat([w.loc[:,0], ww], axis=1)
	w1.columns = ['source','weight']
	weight_source_sum = w1.groupby(['source']).sum()
	weight_source_sum['source'] = weight_source_sum.index
	print 'weight_source_sum function is done'
	return weight_source_sum ## sorted weights dataframe

"""
sorted by weight_target_sum
"""
def weight_target_sum(G):
	w = sorted(G.edges(data=True), key=lambda (source,target,data): data['weight'], reverse=True)
	w = pd.DataFrame(w)
	ww = pd.DataFrame(w.loc[:,2].values.tolist())
	w1 = pd.concat([w.loc[:,1], ww], axis=1)
	w1.columns = ['source','weight'] #实际上是target,但为了后面合并，这里用source
	weight_target_sum = w1.groupby(['source']).sum()
	weight_target_sum['source'] = weight_target_sum.index
	print 'weight_target_sum function is done'
	return weight_target_sum


def all_graph_info(G, filename, w_threshold, beta, df):
	degree_out  = pd.DataFrame([d for n,d in G.out_degree()]) # 这里定义了out_degree
	degree_in = pd.DataFrame([d for n,d in G.in_degree()]) # 这里定义了out_degree
	df1 = pd.DataFrame(data={'year': [filename],
							'threshold':[w_threshold],
							'weight_threhold':[beta],
							'nodes_num': [nx.number_of_nodes(G)],
							'edges_num': [nx.number_of_edges(G)],
							'density':[nx.density(G)],
							'number_weakly_connected_components':[nx.number_weakly_connected_components(G)],
							'number_strongly_connected_components':[nx.number_strongly_connected_components(G)],
							'size_largest_strongly_connected_components':[len(max(nx.strongly_connected_components(G), key=len))],
							'size_largest_weakly_connected_components':[len(max(nx.weakly_connected_components(G), key=len))],
							'weights_sum':[sum(list((nx.get_edge_attributes(G, 'weight')).values()))],
							#'ave_degree_out':[degree_out.mean()], #入度均值=出度均值
							'ave_degree_in':[degree_in.mean()]
							})
							#'ave_clustering_coeffient':[nx.average_clustering(G)],
							#'ave_shortest_path_length':[nx.average_shortest_path_length(G)]
	#print df1
	df = df.append(df1)
	print "this year graph_basic_info is done"
	return df
