# -*- coding: utf-8 -*-

#avoid chinese mess
import sys 
reload(sys)
sys.setdefaultencoding('utf8')
import xlrd
import glob,os
import openpyxl
import networkx as nx

import pandas as pd
import numpy as np
import glob,os
import math
from math import log10

import collections
from collections import Counter
import matplotlib.pyplot as plt



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
	
os.chdir('E:/financial_network/data/threshold/stock_alpha')
V = nx.read_graphml('2015-1_95.graphml')

lower = []
sus = []
sus_lower = []
os.chdir('E:/financial_network/data/status_wind_612_710')
FileList = glob.glob('*.xlsx') #for workingfile in filelist
print FileList
print "FileList length is", len(FileList) #共20个文件
for workingfile in FileList:	
	print workingfile,'is working now'
	filename  = workingfile.split('.')[0]
	print 'filename is', filename
	os.chdir('E:/financial_network/data/status_wind_612_710')
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

df1 = pd.DataFrame(np.zeros((18,20)))
df2 = pd.DataFrame(np.zeros((18,20)))
df3 = pd.DataFrame(np.zeros((18,20)))
for delt_t in range(0,2):
	sum_rate_today = 0
	for i in range(0, 18 - delt_t):
		col_name = '%d'%i 
		rate_today = 0
		sum_rate_node1 = 0
		p = 0
		q = 0
		print 'lenght of today lower is ', len(lower[i])
		print 'lenght of today+delt_t lower is ', len(lower[i+delt_t])
		for node1 in lower[i]:
			if node1 in V.nodes():#确认跌停的这只股票在网络中
				p = p+1
				#print node1 in V.nodes()
				d = list(V.predecessors(node1))#一定要list一下，不然格式不对
				if d:#确定跌停的这只股票有predecessors
					q = q + 1
					#print 'predecessors of',node1, 'are', d
					N = len(d) #影响node1的点是nodes1的前面的点
					#print 'number of predecessors of',node1, 'is', N
					k = 0
					for node2 in list(d):
						if node2 in lower[i+delt_t]:
							k = k+1
					rate_node1 = float(k)/N
					#print 'rate of',node1, 'is', rate_node1
					sum_rate_node1 = sum_rate_node1 + rate_node1
		print p,'stocks lower limit today in this network'
		print q,'stocks lower limit today in this network and has predecessors'
		if q != 0:
			rate_today = float(sum_rate_node1)/q
		else:
			rate_today = 0
		#print 'rate of day',i,'is', rate_today
		df1.iloc[delt_t, i] = rate_today
		df2.iloc[delt_t, i] = float(q)/len(lower[i])
		df3.iloc[delt_t, i] = float(p)/len(lower[i])
		
	
os.chdir('E:/financial_network/data/threshold/stock_alpha/0.95_2015-1')
df1.to_excel('predecessors_lower_rate-1.xlsx', sheet_name='Sheet1')
df2.to_excel('lower_has_predecessors-1.xlsx', sheet_name='Sheet1')
#df3.to_excel('lower_in_network.xlsx', sheet_name='Sheet1')
		
	

