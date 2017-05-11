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
Draw hist with plt
"""
def hist_plt(x, bin,filename, filename2): #filename2是横轴的变量名称，比如出强度;filename是日期
	#plt.hist(x, bins = bin, normed=True,facecolor='green', alpha=0.75)	#normed=1则是频率；log = True,
	#plt.xscale('log')
	#plt.yscale('log')
	plt.hist(x, bins=bin, facecolor='green', alpha=0.75)
	plt.xlabel(filename2)
	plt.ylabel('Frequency')
	plt.savefig(filename+'_'+filename2+'_density_hist.png')
	plt.clf()


def summary(filename, df):
	summary = df.describe()
	summary = pd.DataFrame(summary)
	summary.to_excel(filename+'_summary.xlsx', sheet_name='Sheet1')

"""
run
"""		
os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
V = nx.read_graphml('2015-1_95.graphml')

out_degree =[d for n,d in V.out_degree()] # degree sequence

in_degree =[d for n,d in V.in_degree()] # degree sequence

filename = '2015-195'
#bin = np.linspace(0, 2301, num=50)
bin = np.linspace(-0.5, 9.5, num=10, endpoint=False)#看出度小于10的是什么情况
hist_plt(in_degree, bin, filename, 'in_degree')
#hist_plt(out_degree, bin, filename, 'out_degree')

#
# os.chdir('/Users/shine/work_hard/financial_network/data/status_wind_612_710')
# FileList = glob.glob('*.xlsx') #for workingfile in filelist
# print FileList
# print "FileList length is", len(FileList) #共20个文件
#
#
# labels = list() #boxplot的横轴，即交易日
#
#
# df_sus_box = pd.DataFrame()
# df_lower_sus_box = pd.DataFrame()
#
# lower_out_degree =  pd.DataFrame()
# lower_in_degree =  pd.DataFrame()
# for workingfile in FileList:
# 	print workingfile,'is working now'
# 	filename  = workingfile.split('0150')[1]
# 	filename  = filename .split('.')[0]#只保留日期，不然plot出来的图都重叠了
# 	print 'filename is', filename
# 	os.chdir('/Users/shine/work_hard/financial_network/data/status_wind_612_710')
# 	status = pd.ExcelFile(workingfile)
# 	status = pd.read_excel(status, 'Wind资讯'.decode('utf-8'))
# 	status = pd.DataFrame(status)
# 	status.columns = ['source', 'stock_name', 'limit_ori', 'sus_reason', 'sus_days', 'sus_ori']
# 	status['limit'] = status.apply (lambda df: stock_status(df, 'limit_ori', 'sus_ori'), axis=1)
# 	#print len(status) ##经过drop_emptycell和drop_duplicate验证，status中有很多missing data, 如果只提取分析用的几列数可能会改善
# 	status = status[['source', 'stock_name','limit']]
#
# 	df = drop_emptycell(status)
# 	df = drop_duplicate(df)
#
# 	df_lower = df[df.limit == 'lower_limit']
# 	#print df_lower.iloc[1:20,:]
# 	#df_sus = df[df.limit == 'suspension']
# 	#print df_sus.iloc[1:20,:]
# 	#df_lower_sus = df[df.limit == 'lower_limit'].append(df[df.limit == 'suspension'])
# 	#print df_lower_sus.iloc[1:20,:]
#
# 	labels.extend([filename])#用于boxplot
# 	print 'labels are', labels
#
# 	df_lower = list(df_lower['source'])
#
# 	os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')
#
# 	#out_degree =[d for n,d in V.out_degree(df_lower)] # degree sequence
#
# 	#in_degree =[d for n,d in V.in_degree(df_lower)] # degree sequence
#
# 	bin = np.linspace(0,2301, num = 50)
# 	hist_plt(in_degree, bin,filename, 'in_degree')
# 	hist_plt(out_degree, bin,filename, 'out_degree')
#
