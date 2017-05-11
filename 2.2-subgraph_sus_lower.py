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
delete rows that contain n NaN values
"""
def drop_emptycell(df):
	print 'original data row number', df.shape[0] #Count_Row
	df.replace('', np.nan, inplace=True)#删除原始数据中的空白单元格，不是NaN
	df.dropna(axis=0, inplace=True)
	print 'after delete missing data, row number is', df.shape[0] #Count_Row
	return df
	
def drop_duplicate(df): #after drop empty cells
	df = df.drop_duplicates()  #df.drop_duplicates(subset[])
	print 'after delete duplicate data, row number is', df.shape[0] #Count_Row
	#print df
	return df		


"""
label stock status 
"""
def stock_status(df, col_name1, col_name2):
	if  df[col_name1] == 1: 
		return 'upper_limit'
	if  df[col_name1] == -1: 
		return 'lower_limit'
	if  df[col_name1] == 0:
		if df[col_name2] == u'停牌一天':
			return 'suspension'
		else:
			return 'other'


"""
read the edgelist
"""
def read_edgelist(workingfile):
	G = nx.MultiDiGraph()
	G = nx.read_weighted_edgelist(workingfile,create_using=nx.MultiDiGraph())
	print "reading G is done"
 	return G

"""
weight sequence
"""	
def weight(G):
	w = nx.get_edge_attributes(G, 'weight')
	w = list(w.values())
	return w
"""
compute the graph basic information, and add the graph infro to a dataframe.
"""
def graph_structure_info(G, filename, df):
	#degree_out  = pd.DataFrame([d for n,d in G.out_degree()]) # 这里定义了out_degree
	degree_in = pd.DataFrame([d for n,d in G.in_degree()]) # 这里定义了out_degree
	df1 = pd.DataFrame(data={'date': [filename],
							'nodes_num': [G.number_of_nodes()],
							'edges_num': [G.number_of_edges()],
							'density':[nx.density(G)],
							'number_weakly_connected_components':[nx.number_weakly_connected_components(G)],
							'number_strongly_connected_components':[nx.number_strongly_connected_components(G)],
							#'size_largest_strongly_connected_components':[len(max(nx.strongly_connected_components(G), key=len))],
							#'size_largest_weakly_connected_components':[len(max(nx.weakly_connected_components(G), key=len))],
							'weights_sum':[sum(list((nx.get_edge_attributes(G, 'weight')).values()))],
							#'ave_degree_out':[degree_out.mean()], #入度均值=出度均值
							'ave_degree':[degree_in.mean()]
							})
							#'ave_clustering_coeffient':[nx.average_clustering(G)],
							#'ave_shortest_path_length':[nx.average_shortest_path_length(G)]
	#print df1
	df = df.append(df1)
	print "this year graph_basic_info is done"
	return df




os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
V = read_edgelist("2015-1_95.edgelist")

summary_lower_sus = pd.DataFrame()
graph_basic_info = pd.DataFrame()
"""
子网络
"""	
os.chdir('/Users/shine/work_hard/financial_network/data/status_wind_612_710')
FileList = glob.glob('*.xlsx') #for workingfile in filelist
print FileList
print "FileList length is", len(FileList) #共20个文件
for workingfile in FileList:	
	print workingfile,'is working now'
	filename  = workingfile.split('.')[0]
	print 'filename is', filename
	os.chdir('/Users/shine/work_hard/financial_network/data/status_wind_612_710')
	status = pd.ExcelFile(workingfile) 
	status = pd.read_excel(status, 'Wind资讯'.decode('utf-8'))
	status = pd.DataFrame(status)
	status.columns = ['source', 'stock_name', 'limit_ori', 'sus_reason', 'sus_days', 'sus_ori']
	status['limit'] = status.apply (lambda df: stock_status(df, 'limit_ori', 'sus_ori'), axis=1)
	#print len(status) ##经过drop_emptycell和drop_duplicate验证，status中有很多missing data, 如果只提取分析用的几列数可能会改善
	df = status[['source', 'stock_name','limit']]
	df = drop_duplicate(df)
	df = drop_emptycell(df)
	#print status.iloc[1:20,:]
	#print '------------'

	df_lower = df[df.limit == 'lower_limit']
	lower =  df_lower['source'].tolist()
	#print 'lower length', len(lower)
	#print lower
	df_sus = df[df.limit == 'suspension']
	sus =  df_sus['source'].tolist()
	df_lower_sus = df[df.limit == 'lower_limit'].append(df[df.limit == 'suspension'])
	sus_lower =  df_lower_sus['source'].tolist()


	#summary_lower_sus1 = pd.DataFrame(data = {'date': [filename],
	#										'lower_limit_num': [len(lower)],
	#										'suspension_num': [len(sus)],
	#										'sus_lower_num': [len(sus_lower)]
	#										})
	
	#summary_lower_sus = summary_lower_sus.append(summary_lower_sus1)
	
	sus_lower_within = []
	for node1 in sus_lower:
		if node1 in V.nodes():#确认跌停的这只股票在网络中
			sus_lower_within.append(node1)
	
	print 'lenght of today lower is ',len(sus_lower)
	print 'lenght of today lower in the network is ',len(sus_lower_within)
	

	#Z = V.subgraph(lower_within)

	Z = V.subgraph(sus_lower_within)
	print 'nodes_num', len(Z)
	
	#os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1/subgraph/lower')
	os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1/subgraph/sus_lower')
	graph_basic_info = graph_structure_info(Z, filename, graph_basic_info)
	#nx.write_graphml(Z,filename+"_graph_sus_lower.graphml") 
	nx.write_graphml(Z,filename+"_sus_lower.graphml") 

#os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1/subgraph/lower')
os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1/subgraph/sus_lower')
graph_basic_info.to_excel('aplha95_graph_basic_info-by_date.xlsx', sheet_name='Sheet1')
print "graph_basic_info is done"


#os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1/subgraph')
#summary_lower_sus.to_excel('status_summary.xlsx', sheet_name='Sheet1')
