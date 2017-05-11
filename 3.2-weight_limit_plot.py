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
Draw scatter with matplotlib.
"""	
def scatter(filename, x, y):
	ax = plt.axes()
	ax.scatter(x,y,color='blue',s=5,edgecolor='none')
	ax.set_title(filename+'_scatter')
	plt.savefig(filename+"_scatter.png")
	plt.clf()	
	

"""
Draw hist with seaborn.
"""
def hist_seaborn(filename, w):
	ax = plt.axes()
	sns.distplot(w, kde=False, ax = ax)
	ax.set_title(filename+'_hist')
	plt.savefig(filename+'_hist.png')
	plt.clf()
	print "weight distribution: histogram  is done"

"""
Draw hist with plt
"""
def hist_plt(x, bin,filename, filename2): #filename2是横轴的变量名称，比如出强度;filename是日期
	plt.hist(x, bins = bin, normed=True,facecolor='green', alpha=0.75)	#normed=1则是频率；log = True,
	#plt.xscale('log')
	#plt.yscale('log')
	plt.xlabel(filename2)
	plt.ylabel('Frequency')
	plt.savefig(filename+'_'+filename2+'_density_hist.png')
	plt.clf()

"""
Draw hist in scatter.
"""
def hist_to_scatter(x, bin, filename, filename2):
    n, bins = np.histogram(x,bins = bin)
    bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
    plt.scatter(bins_mean, n, marker='x', s = 30)
    plt.xlabel(filename2)
    plt.ylabel('Frequency')
    plt.savefig(filename+'_'+filename2+'_frequency_scatter.png')
    plt.clf()


"""
Draw hist with plt-log 
"""
def hist_plt_log(x, bin,filename, filename2): #filename2是横轴的变量名称，比如出强度;filename是日期
	x = np.log(x)
	weights = np.ones_like(x)/float(len(x))
	plt.hist(x, bins = bin, normed= 0,  weights=weights, facecolor='green', alpha=0.75)	#normed=1则是频率,但和weight有冲突，用weight比较稳定；log = True,bins = bin,bins = bin,
	#if you want the sum of all bars to be equal unity, weight each bin by the total number of values:http://stackoverflow.com/questions/5498008/pylab-histdata-normed-1-normalization-seems-to-work-incorrect
	plt.xlabel(filename2)
	plt.ylabel('Frequency')
	plt.savefig(filename+'_'+filename2+'_frequency_log_hist.png')
	plt.clf()

"""
Draw hist in scatter-log
"""
def hist_to_scatter_log(x, bin, filename, filename2):
	x = np.log(x)
	n, bins = np.histogram(x, bins = bin)
	bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
	plt.scatter(bins_mean, n, marker='x', s = 30)
	plt.xlabel(filename2)
	plt.ylabel('Frequency')
	plt.savefig(filename+'_'+filename2+'_frequency_log_scatter.png')
	plt.clf()

	
# plot a scatter plot using the histogram output in matplotlib?http://stackoverflow.com/questions/18325706/how-to-plot-a-scatter-plot-using-the-histogram-output-in-matplotlib
#http://stackoverflow.com/questions/6855710/how-to-have-logarithmic-bins-in-a-python-histogram
#As far as I know the option Log=True in the histogram function only refers to the y-axis.
#use logspace() to create a geometric sequence, and pass it to bins parameter. And set the scale of xaxis to log scale.
#data = np.random.normal(size=10000)
#plt.xscale('log')
#plt.yscale('log')
#pl.hist(data, bins=np.logspace(np.log10(0.1),np.log10(1.0),50))
#pl.show()
#这与前面看到的log-bin的度分布代码是一致的

"""
run考虑到7月9日、10日的跌停股票=1或0，这两天考察停牌股票，要单独运行（先把这两天的文件挪到另一个文件夹）
"""		

filename1 = '2015-1'
#alpha = '1925'	
#alpha = '0'	
alpha = '95'	
filename2 = '_weight_source_target_sum-manual'
#filename2 = '_weight_source_sum'
#filename2 = '_weight_target_sum'
#filename2 = '_nodes_degree'

os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1/whole0.95network')	
#df = pd.ExcelFile('2015-2_weight_source_target_sum-manual.xlsx')#需要manual出weight一列，即weight_x+weight_y的均值
weight = pd.ExcelFile(filename1+alpha+filename2+'.xlsx') #nodes_degree也暂称为weight
weight = weight.parse("Sheet1")	


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
	status = status[['source', 'stock_name','limit']]
	#print status.iloc[1:20,:]
	#print '------------'

	os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1/whole0.95network')		
	df = status.merge(weight,on='source')
	df = drop_emptycell(df)
	df = drop_duplicate(df)
	#print df.iloc[1:20,:]
	#df=df.rename(columns = {'weight':filename})

	df_lower = df[df.limit == 'lower_limit'] #该日的跌停股票的weight
	#print df_lower.iloc[1:20,:]
	df_sus = df[df.limit == 'suspension']
	#print df_sus.iloc[1:20,:]
	df_sus_lower = df[df.limit == 'lower_limit'].append(df[df.limit == 'suspension'])
	#print df_lower_sus.iloc[1:20,:]
	
	#print df_lower['weight']
	#hist(filename+'_'+filename2+'_'+'lower', df_lower['weight']) #use seaborn

	#bin_count = 50#for weights
	
	os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1/whole0.95network/lower')	
	#os.chdir('/Users/shine/work_hard/financial_network/data/threshold/edgelist_alppha_2015-2/alpha0.75/sus_lower')	
	#os.chdir('/Users/shine/work_hard/financial_network/data/threshold/edgelist_alppha_2015-2/alpha0.75/sus')	
	#bin = np.linspace(3, 8, num = 50)
	bin = np.linspace(9, 21, num = 50) #for weights
	hist_plt_log(df_lower['weight'], bin,filename, filename2)
	bin = np.linspace(0, 500000000, num = 50) #for weights
	#bin = np.linspace(0, 2400, num = 50)
	hist_to_scatter(df_lower['weight'], bin, filename, filename2)


