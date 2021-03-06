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
degree sequence 
"""	
def degree_sequence(G):
	d = sorted([d for n,d in G.degree()], reverse=True) # degree sequence
	degree_sequence = np.asarray(d) #turn list into arrary
	return degree_sequence
	
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
	return summary
	#summary.to_excel(filename+'_summary.xlsx', sheet_name='Sheet1')
	


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

lower = []
sus = []
sus_lower = []

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
	#print status.iloc[1:20,:]
	#print '------------'
	df = drop_duplicate(df)
	df = drop_emptycell(df)
	#df_lower = df[df.limit == 'lower_limit']
	#lower.append(df_lower['source'].tolist())
	#df_sus = df[df.limit == 'suspension']
	#sus.append(df_sus['source'].tolist())
	df_lower_sus = df[df.limit == 'lower_limit'].append(df[df.limit == 'suspension'])
	sus_lower.append(df_lower_sus['source'].tolist())
	
print '-------and-----------------'

df = pd.DataFrame()
for delt_t in range(0,20):
#for delt_t in range(0,1):
	sum_rate_today = 0
	for i in range(0, 20 - delt_t):
		rate_today = 0
		sum_rate_node1 = 0
		p = 0
		q = 0
		for node1 in sus_lower[i]:
			#print node1
			if node1 in V.nodes():#确认跌停的这只股票在网络中
				#print node1 in V.nodes()
				d = list(V.successors(node1))#一定要list一下，不然格式不对
				if d:#确定跌停的这只股票有predecessors
					q = q + 1
					#print 'predecessors of',node1, 'are', d
					N = len(d) #影响node1的点是nodes1的前面的点
					#print 'number of predecessors of',node1, 'is', N
					k = 0
					for node2 in list(d):
						if node2 in sus_lower[i+delt_t]:
							k = k+1
					rate_node1 = float(k)/N
					print 'rate of',node1, 'is', rate_node1
					sum_rate_node1 = sum_rate_node1 + rate_node1
		if q != 0:
			rate_today = float(sum_rate_node1)/q
		else:
			rate_today = 0
		print 'rate of day',i,'is', rate_today
		sum_rate_today = sum_rate_today + rate_today
	rate_delt = sum_rate_today/(20-delt_t)
	print 'rate of delt',delt_t,'is', rate_delt
	df1 = pd.DataFrame(data={'delt_t': [delt_t],
							'out_sus_lower_ave_rate': [rate_delt]
							})
	print df1
	df = df.append(df1)
	
os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')
df.to_excel('successors_sus_lower_rate_ave.xlsx', sheet_name='Sheet1')
		
	

