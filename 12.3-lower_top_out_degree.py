# -*- coding: utf-8 -*-

#avoid chinese mess
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import glob,os
import xlrd
import networkx as nx
import pandas as pd
import numpy as np
from my_network import remove_edges, read_edgelist, weight, all_graph_info
from my_plots import density, hist_plt,hist_log_bin
import matplotlib.pyplot as plt
from my_dataframes import drop_duplicate, drop_emptycell, stock_status


lower = []
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


os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')
df = pd.read_excel('node_degree_weight-manual分列.xlsx', 'Sheet1')
df = pd.DataFrame(df)


# for i in range(0,18):
# 	day = '%d'%i
# 	lower_df = pd.DataFrame()
# 	for node2 in lower[i]:
# 		df1 = df[df.node == node2]
# 		lower_df = lower_df.append(df1)
# 	lower_df.to_excel(day + 'lower_node_degree_weight.xlsx', sheet_name='Sheet1')

df_1 = pd.DataFrame()

for i in range(0,18):
 	day = '%d'%i
	lower_df = pd.DataFrame()
 	for node2 in lower[i]:
 		df1 = df[df.node == node2]
 		lower_df = lower_df.append(df1)
	df_11 = pd.DataFrame({'date': [day],
						'in_degree_ave': [np.mean(lower_df['in_degree'])],
						'out_degree_ave': [np.mean(lower_df['out_degree'])],
						'weight_target_sum': [np.mean(lower_df['weight_target_sum'])],
						'weight_source_sum': [np.mean(lower_df['weight_source_sum'])]
						})
	df_1 = df_1.append(df_11)

df_1.to_excel('lower_node_degree_weight_ave.xlsx', sheet_name='Sheet1')