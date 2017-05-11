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
	
os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
V = read_edgelist("2015-1_95.edgelist")

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

	df_lower = df[df.limit == 'lower_limit']
	lower.append(df_lower['source'].tolist())
	df_sus = df[df.limit == 'suspension']
	sus.append(df_sus['source'].tolist())
	df_lower_sus = df[df.limit == 'lower_limit'].append(df[df.limit == 'suspension'])
	sus_lower.append(df_lower_sus['source'].tolist())
	
print '-------and-----------------'


df2 = pd.DataFrame(np.zeros((18,18)))
df1 = pd.DataFrame(np.zeros((18,18)))
for delt_t in range(0,18):
	sum_rate_today = 0
	for i in range(0, 18 - delt_t):
		rate_today = 0
		sum_rate_node1 = 0
		sum_rate_has_path = 0
		p = 0
		#print 'today lower limit is', len(lower[i])
		for node1 in lower[i]:
			if node1 in V.nodes():#确认跌停的这只股票在网络中
				p = p + 1  #第i天有多少只跌停股票在网络中
				q = 0
				t = 0
				ws = 0
				for node2 in lower[i+delt_t]:
					if node2 in V.nodes():#确认跌停的这只股票在网络中
						q = q + 1 #在node1时有多少个node2与之成对，即node2在网络中的个数，q/len(lower[i+delt_t])就是这一天跌停的股票在网络中的比例
						if nx.has_path(V, node1, node2):
							t = t + 1 #在node1与node2成对中有多少个是连通的
							#k = nx.dijkstra_path_length(V, node1, node2)
							#ws = ws + k		
				#rate_node1 = float(ws)/t #node1到所有能到达的node2的平均长度
				rate_has_path = float(t)/q #node1与node2集合中能连通的比例
				#print 'rate of',node1, 'is', rate_node1
				#sum_rate_node1 = sum_rate_node1 + rate_node1 #所有node1到达能到达的node2的平均长度之和
				sum_rate_has_path =sum_rate_has_path + rate_has_path #所有node1到达能到达的node2集合中能连通的比例
		if p != 0:
			#path_today = float(sum_rate_node1)/p #在网络中p只股票到达delta_t后那天跌停的股票的路径长度（能到达的）
			has_path_today = sum_rate_has_path/p #在网络中p只股票到达delta_t后那天跌停的股票能连通的比例
		else:
			#path_today = 0
			has_path_today = 0
		#print 'rate of day',i,'is', path_today
		#df1.iloc[delt_t, i] = path_today
		df2.iloc[delt_t, i] = has_path_today

	
os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')
#df1.to_excel('path_length_lower.xlsx', sheet_name='Sheet1')
df2.to_excel('has_path_rate_lower.xlsx', sheet_name='Sheet1')
		
	

