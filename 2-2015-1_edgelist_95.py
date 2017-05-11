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
run考虑到7月9日、10日的跌停股票=1或0，这两天考察停牌股票，要单独运行（先把这两天的文件挪到另一个文件夹）
"""	
	
w_threshold = 0.95

os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
V = read_edgelist('2015-1.edgelist')

w = weight(V)
w = pd.DataFrame(w)
beta = w.quantile(q = w_threshold)
print beta

V = remove_edges(V, beta)
print "after remove edges, stock network nodes and edges are ",nx.number_of_nodes(V), nx.number_of_edges(V)
	
nx.write_weighted_edgelist(V,"2015-1_95.edgelist")

#验证了nx.write_weighted_edgelist会删去度为0的点
#V = read_edgelist('2015-1_95.edgelist')
#print nx.number_of_nodes(V), nx.number_of_edges(V)