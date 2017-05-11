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
from my_network import remove_edges, read_edgelist, weight
from my_plots import density
import matplotlib.pyplot as plt



os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')
file = glob.glob('node_degree_weight-manual分列.xlsx') #for workingfile in filelist
df = pd.ExcelFile(file[0])
df = pd.read_excel(df, 'Sheet1')
df = pd.DataFrame(df)


x = df['out_degree']
y = df['in_degree']
#z = df['weight_source_sum']
z = df['weight_target_sum']
#beta = z.quantile(q = 0.98)
beta = z.quantile(q = 0.995)
print beta

xlab = 'out_degree'
ylab = 'in_degree'
#filename = 'degree_weight_source_sum'
filename = 'degree_weight_target_sum'

fig, ax = plt.subplots()
ax.scatter(x= x, y=y, s=z/3000000, linewidth=0, alpha=0.5) #weight数量太大，除以300000，让单位小一些
#for i, txt in enumerate(df['node']):
    #if z[i] >= beta:
       # ax.annotate(txt, (x[i],y[i]), size = 8)
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(filename + '_buble_plot')
plt.savefig(filename + "_buble_plot.png")
plt.clf()