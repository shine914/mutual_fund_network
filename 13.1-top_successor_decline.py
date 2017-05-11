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
from my_network import remove_edges, read_edgelist, weight, all_graph_info, bipartite_graph,stock_network,graph_structure_info
from my_plots import density, hist_plt,hist_log_bin
import matplotlib.pyplot as plt
from my_dataframes import drop_duplicate, drop_emptycell, stock_status,summary


lower = []
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

os.chdir('E:/financial_network/data/threshold/stock_alpha')
V = nx.read_graphml('2015-1_95.graphml')


os.chdir('E:/financial_network/data/2709_return')
decline = pd.read_csv('returns_reuters_2709.csv')

#t_list = ['601318.SH', '601166.SH', '600036.SH', '600016.SH', '600030.SH','600000.SH','300059.SZ','000002.SZ']
#t_list = ['601318.SH', '601166.SH', '600036.SH', '600016.SH', '600030.SH','600000.SH','000002.SZ','300059.SZ','002183.SZ']
t_list = ['601318.SH', '601166.SH', '600000.SH','000002.SZ']
#t_list = ['601318.SH', '601166.SH', '600036.SH', '600016.SH', '600030.SH','600000.SH','300059.SZ','000002.SZ',
 #         '002183.SZ','601169.SH','601988.SH','000651.SZ','000001.SZ','601398.SH','601688.SH','600519.SH','600446.SH',
  #        '601818.SH','600271.SH','300253.SZ','600050.SH']


df = pd.DataFrame()
for node1 in t_list:
    declining = decline[decline.source == node1]
    print declining
    if len(declining) !=0:
        for i in range(1, 19):
            if node1 in V.nodes():  # 确认跌停的这只股票在网络中
                d = list(V.successors(node1))  # 一定要list一下，不然格式不对
                N = len(d)
                k = 0
                for node2 in list(d):
                    if node2 in lower[i-1]:
                        k = k + 1
                node1_suc_rate = float(k) / N
                df1 = pd.DataFrame({'date': [i],
                                    'stock': [node1],
                                    'successor_lower_rate': [node1_suc_rate],
                                    'decline': [declining.iloc[0,i]]
                                    })
                df = df.append(df1)


os.chdir('E:/financial_network/data/threshold/stock_alpha')
df.to_excel('top10_decline_successor_rate.xlsx', sheet_name='Sheet1')
