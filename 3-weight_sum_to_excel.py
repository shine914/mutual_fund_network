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
weight sequence
"""	
def weight(G):
	w = nx.get_edge_attributes(G, 'weight')
	w = list(w.values())
	return w

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
sorted by weight_source_sum
"""
def weight_source_sum(G, filename):
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
def weight_target_sum(G, filename):
	w = sorted(G.edges(data=True), key=lambda (source,target,data): data['weight'], reverse=True)
	w = pd.DataFrame(w)
	ww = pd.DataFrame(w.loc[:,2].values.tolist())
	w1 = pd.concat([w.loc[:,1], ww], axis=1)
	w1.columns = ['source','weight'] #实际上是target,但为了后面合并，这里用source
	weight_target_sum = w1.groupby(['source']).sum()
	weight_target_sum['source'] = weight_target_sum.index
	print 'weight_target_sum function is done'
	return weight_target_sum 

"""
nodes-degrees
"""
def nodes_degree(G, filename):
	w = sorted(G.edges(data=True), key=lambda (source,target,data): data['weight'], reverse=True)
	w = pd.DataFrame(w)
	ww = pd.DataFrame(w.loc[:,2].values.tolist())
	w1 = pd.concat([w.loc[:,0], ww], axis=1)
	w1.columns = ['source','weight']
	count = Counter(w1['source'])
	print 'counter is done'
	node_degree = pd.DataFrame(count, index=[0])#use scalar values and pass an index:eg({'A': a, 'B': b}, index=[0])
	node_degree = node_degree.transpose()
	node_degree['source'] = node_degree.index
	node_degree.columns = ['weight','source']#这里的weight就是节点出度；根据数据框情况发现是度数在前，股票代码在后
	print 'nodes_degree function is done'
	return node_degree  


	
"""
weight summary 
"""	
def weight_info_by_year(G, filename, df):
	w = nx.get_edge_attributes(G, 'weight')
	weights_info1 = list(w.values())
	weights_info1 = pd.DataFrame(weights_info1)
	weights_info1.columns = [filename]
	df = pd.concat([df, weights_info1], axis=1)
	print 'this year weights summary is done'
	return df
	
		
"""
degree summary 
"""
def degree_info_by_year(G, filename, df):
	degree_info1  = sorted([d for n,d in G.out_degree()], reverse=True) # 这里定义了out_degree
	degree_info1  = pd.DataFrame(degree_info1)
	degree_info1.columns = [filename]
	df = pd.concat([df, degree_info1], axis=1)
	print "this year degree summary is done"
	return df

"""
Draw scatter with matplotlib.
"""	
def scatter(filename, x, y):
	ax = plt.axes()
	ax.scatter(x,y,color='blue',s=5,edgecolor='none')
	ax.set_title(filename+'_scatter')
	plt.savefig(filename+"_scatter.png")
	plt.clf()	
	
def summary(filename, df):
	summary = df.describe()
	summary = pd.DataFrame(summary)
	summary.to_excel(filename+'_summary.xlsx', sheet_name='Sheet1')
	
	
"""
Draw denstiy with seaborn.
"""
def density(filename, sequence):
	ax = plt.axes()
	sns.kdeplot(sequence, ax = ax)
	ax.set_title(filename+'_density')
	plt.savefig(filename+'_density.png')
	plt.clf()

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
run
"""	

filename = '2015-1'
w_threshold = float(0.95)
print 'w_threshold is', w_threshold
alpha = '95'
print 'alpha is', alpha

os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
V = nx.read_graphml('2015-1_95.graphml')
	
os.chdir('/Users/shine/work_hard/financial_network/data/threshold')
decline = pd.ExcelFile('6.12-7.9_decline.xlsx')
decline = decline.parse("Sheet1")
print 'decline file is working, length is', len(decline)
decline = drop_duplicate(decline)


os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')	
weight_source_sum = weight_source_sum(V, filename)
weight_target_sum = weight_target_sum(V, filename)
nodes_degree = nodes_degree(V, filename)	
weight_source_target_sum = weight_source_sum.merge(weight_target_sum,on='source')

weight_source_sum.to_excel(filename+alpha+'_weight_source_sum.xlsx', sheet_name='Sheet1')
weight_target_sum.to_excel(filename+alpha+'_weight_target_sum.xlsx', sheet_name='Sheet1')
weight_source_target_sum.to_excel(filename+'_weight_source_target_sum.xlsx', sheet_name='Sheet1')

# nodes_degree.to_excel(filename+alpha+'_nodes_degree.xlsx', sheet_name='Sheet1')
#
# merged_nodes_degree = nodes_degree.merge(decline,on='source')
# merged_nodes_degree = drop_duplicate(merged_nodes_degree)
# merged_nodes_degree = drop_emptycell(merged_nodes_degree)
# x1 = merged_nodes_degree['weight']#这里的weight就是out_degree有多少
# y1 = merged_nodes_degree['总计'.decode('utf-8')]
# density(filename+alpha+'_nodes_degree', x1)
# summary(filename+alpha+'_nodes_degree', x1)
# scatter(filename+alpha+'_nodes_degree', x1, y1)
#
# merged_weight_source_sum = weight_source_sum.merge(decline,on='source')
# merged_weight_source_sum = drop_duplicate(merged_weight_source_sum)
# merged_weight_source_sum = drop_emptycell(merged_weight_source_sum)
# x2 = merged_weight_source_sum['weight']
# y2 = merged_weight_source_sum['总计'.decode('utf-8')]
# density(filename+alpha+'_weight_source_sum', x2)
# summary(filename+alpha+'_weight_source_sum', x2)
# scatter(filename+alpha+'_weight_source_sum', x2, y2)
#
# merged_weight_target_sum = weight_target_sum.merge(decline,on='source')
# merged_weight_target_sum = drop_duplicate(merged_weight_target_sum)
# merged_weight_target_sum = drop_emptycell(merged_weight_target_sum)
# x3 = merged_weight_target_sum['weight']
# y3 = merged_weight_target_sum['总计'.decode('utf-8')]
# density(filename+alpha+'_weight_target_sum', x3)
# summary(filename+alpha+'_weight_target_sum', x3)
# scatter(filename+alpha+'_weight_target_sum', x3, y3)
#
# degree_info = pd.DataFrame()
# weight_info = pd.DataFrame()
# weight_info = weight_info_by_year(V, filename, weight_info)
# degree_info = degree_info_by_year(V, filename, degree_info)
# degree_info.to_excel(alpha+'degree_info-by_year.xlsx', sheet_name='Sheet1')
# weight_info.to_excel(alpha+'weight_info-by_year.xlsx', sheet_name='Sheet1')
# print "degree info and weight info are done"
#

print '-------end-----------------'