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
run
"""		

os.chdir('/Users/shine/work_hard/financial_network/data/status_wind_612_710')
FileList = glob.glob('*.xlsx') #for workingfile in filelist
print FileList
print "FileList length is", len(FileList) #共20个文件

df_len = pd.DataFrame()
summary_lower = pd.DataFrame()
summary_sus = pd.DataFrame()
summary_lower_sus = pd.DataFrame()
labels = list() #boxplot的横轴，即交易日
length = pd.DataFrame()
	

for workingfile in FileList:	
	print workingfile,'is working now'
	filename  = workingfile.split('0150')[1]
	filename  = filename .split('.')[0]#只保留日期，不然plot出来的图都重叠了
	print 'filename is', filename
	os.chdir('/Users/shine/work_hard/financial_network/data/status_wind_612_710')
	status = pd.ExcelFile(workingfile) 
	status = pd.read_excel(status, 'Wind资讯'.decode('utf-8'))
	status = pd.DataFrame(status)
	status.columns = ['source', 'stock_name', 'limit_ori', 'sus_reason', 'sus_days', 'sus_ori']
	status['limit'] = status.apply (lambda df: stock_status(df, 'limit_ori', 'sus_ori'), axis=1)
	#print len(status) ##经过drop_emptycell和drop_duplicate验证，status中有很多missing data, 如果只提取分析用的几列数可能会改善
	status = status[['source', 'stock_name','limit']]
	#print '------------'

	df = drop_emptycell(status)
	df = drop_duplicate(df)
	
	df_lower = df[df.limit == 'lower_limit']
	#print df_lower.iloc[1:20,:]
	df_sus = df[df.limit == 'suspension']
	#print df_sus.iloc[1:20,:]
	df_lower_sus = df[df.limit == 'lower_limit'].append(df[df.limit == 'suspension'])
	#print df_lower_sus.iloc[1:20,:]
	
	length1 = pd.DataFrame(data={'date': [filename],
							'total': [len(df)],
							'lower': [len(df_lower)],
							'sus': [len(df_sus)],
							'lower_sus': [len(df_lower_sus)]
							})
	length = length.append(length1)
							
length.to_excel('status_ratio.xlsx', sheet_name='Sheet1')