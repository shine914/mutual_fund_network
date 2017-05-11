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

def box_viz(df):
    ax = sns.boxplot(df)
    plt.xticks(rotation=60)
    plt.show()

"""
run
"""		

filename1 = '2015-1'
#alpha = '1925'	
#alpha = '0'	
alpha = '95'	
filename2 = '_weight_source_target_sum-manual'
#filename2 = '_weight_source_sum'
#filename2 = '_weight_target_sum'
#filename2 = '_nodes_degree'

os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')	
#df = pd.ExcelFile('2015-2_weight_source_target_sum-manual.xlsx')#需要manual出weight一列，即weight_x+weight_y的均值
weight = pd.ExcelFile(filename1+alpha+filename2+'.xlsx') #nodes_degree也暂称为weight
weight = weight.parse("Sheet1")	


os.chdir('/Users/shine/work_hard/financial_network/data/status_wind_612_710')
FileList = glob.glob('*.xlsx') #for workingfile in filelist
print FileList
print "FileList length is", len(FileList) #共20个文件

df_len = pd.DataFrame()
summary_lower = pd.DataFrame()
summary_sus = pd.DataFrame()
summary_lower_sus = pd.DataFrame()
labels = list() #boxplot的横轴，即交易日

df_lower_box = pd.DataFrame()
df_sus_box = pd.DataFrame()
df_lower_sus_box = pd.DataFrame()
	

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
	#print status.iloc[1:20,:]
	#print '------------'

	os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')	
	df = status.merge(weight,on='source')
	df = drop_emptycell(df)
	df = drop_duplicate(df)
	#print df.iloc[1:20,:]
	df=df.rename(columns = {'weight':filename})

	df_lower = df[df.limit == 'lower_limit']
	#print df_lower.iloc[1:20,:]
	df_sus = df[df.limit == 'suspension']
	#print df_sus.iloc[1:20,:]
	df_lower_sus = df[df.limit == 'lower_limit'].append(df[df.limit == 'suspension'])
	#print df_lower_sus.iloc[1:20,:]
	
	#df_len1 = pd.DataFrame(data={'date': [filename],
	#						'lower_limit_num': [len(df_lower)],
	#						'suspension_num': [len(df_sus)],
	#						'lower_sus_num': [len(df_lower_sus)]
	#						})
	#print df_len1
	#df_len = df_len.append(df_len1)
	
	#summary_lower1 = pd.DataFrame(df_lower[filename].describe())
	#summary_sus1 = pd.DataFrame(df_sus[filename].describe())
	#summary_lower_sus1 = pd.DataFrame(df_lower_sus[filename].describe())

	#summary_lower = pd.concat([summary_lower, summary_lower1], axis=1)
	#summary_sus = pd.concat([summary_sus, summary_sus1], axis=1)
	#summary_lower_sus = pd.concat([summary_lower_sus, summary_lower_sus1], axis=1)
	
	labels.extend([filename])#用于boxplot
	print 'labels are', labels
	
	df_lower_box = pd.concat([df_lower_box, df_lower[filename]], axis=1)
	print len(df_lower_box)
	df_sus_box = pd.concat([df_sus_box, df_sus[filename]], axis=1)
	print len(df_sus_box )
	df_lower_sus_box = pd.concat([df_lower_sus_box, df_lower_sus[filename]], axis=1)
	print len(df_lower_sus_box)



"""
boxplot跌停股票
"""		
sns.set_style('whitegrid')
plt.figure()
ax = sns.boxplot(df_lower_box)##最终。。。sns包救了我。。。感谢天感谢地，死磕三小时有了结果2016年11月6日，一个独自在实验室奋战到深夜的女博士,,,
#plt.xticks(rotation=60)
#df_lower_box.boxplot(fontsize=9, showfliers=True) #df_lower_box每一列是一个交易日的数据，不跌停的股票是NAN；所以先把有值的取出来http://stackoverflow.com/questions/23144071/python-boxplot-out-of-columns-of-different-lengths
#I use df.dropna to drop the rows in each column with missing values. However, this is resizing the dataframe to the lowest common denominator of column length, and messing up the plotting.
#The right way to do it, saving from reinventing the wheel, would be to use the .boxplot() in pandas, where the nan handled correctly:
#plt.xticks(range(1,21),labels,fontsize=9)
#means = [df_lower_box.iloc[:,i].mean(axis = 0) for i in range(0, 20)] #算出每一天的均值；pandas的boxplot中只有四分位点；mat的boxplot中有showmeans,但是无法处理有NaN的情况;这里不懂为什么非得axis=0才对...测试结果如此
#print means
#print labels
#labels_scatter = pd.DataFrame(labels)
#means = pd.DataFrame(means)
#plt.scatter(labels, means)#仍然与mean错位，即612日上是615的均值，错位了。未找到方法解决。。。
plt.xlabel('date')
plt.ylabel(filename2)
plt.title(filename2+'_lower_boxplot')
plt.savefig(filename2+'_lower_boxplot.png')
plt.clf()

"""
boxplot停牌股票
"""		
plt.figure()
ax = sns.boxplot(df_sus_box)
#plt.xticks(rotation=60)
#df_sus_box.boxplot(fontsize=9, showfliers=True)
#plt.xticks(range(1,21),labels, fontsize=9)
plt.xlabel('date')
plt.ylabel(filename2)
plt.title(filename2+'_sus_boxplot')
plt.savefig(filename2+'_sus_boxplot.png')
plt.clf()

"""
boxplot跌停与停牌股票
"""	
plt.figure()
ax = sns.boxplot(df_lower_sus_box)
#plt.xticks(rotation=60)
#df_lower_sus_box.boxplot(fontsize=9, showfliers=True)
#plt.xticks(range(1,21),labels,fontsize=9)
plt.xlabel('date')
plt.ylabel(filename2)
plt.title(filename2+'_lower_sus_boxplot')
plt.savefig(filename2+'_lower_sus_boxplot.png')
plt.clf()
