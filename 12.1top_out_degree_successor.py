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

def zhexian(x, filename):
    plt.figure()
    x.plot()
    plt.title(filename)
    plt.savefig(filename + ".png")
    plt.clf()

os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
V = nx.read_graphml('2015-1_95.graphml')


#t_list = ['601318.SH', '601166.SH', '600036.SH', '600016.SH', '600030.SH','600000.SH','300059.SZ','002183.SZ','000002.SZ']
t_list = ['000732.SZ']
decline = pd.read_excel('6.12-7.8_decline-manual.xlsx', sheet_name='Sheet1')
decline = pd.DataFrame(decline)
decline = drop_emptycell(decline)
decline = drop_duplicate(decline)


for i in t_list:
    decline1 = decline[decline.source == i]
    if len(decline1) != 0: #判断一下是否为空
        #print decline1
        x = decline1.iloc[:,1:18].transpose()
        node = decline1.iloc[0,0]
        #zhexian(x, node+'_decline')

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


os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
for node1 in t_list:
    df = pd.DataFrame(np.zeros((18, 18)))
    for i in range(0, 18):
        if node1 in V.nodes():  # 确认跌停的这只股票在网络中
            d = list(V.successors(node1))  # 一定要list一下，不然格式不对
            N = len(d)
            k = 0
            for node2 in list(d):
                if node2 in lower[i]:
                    k = k + 1
            df.iloc[0, i] = float(k) / N
    df.to_excel(node1 + '_successors.xlsx', sheet_name='Sheet1')
