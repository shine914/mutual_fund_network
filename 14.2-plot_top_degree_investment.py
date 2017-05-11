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
import seaborn as sns


#filename = '601318'
#filename = '600000'
filename = '000002'
#filename = '601166'

os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha/0.95_2015-1')
#df = pd.read_excel('601318_company_invest_ratio_network-info_manual.xlsx', 'Sheet1')
df = pd.read_excel(filename+'_company_invest_ratio_network-info.xlsx', 'Sheet1')
df = pd.DataFrame(df)



plt.figure()
sns.lmplot('out_degree', 'rate', data=df, size=15)
#sns.lmplot('out_degree', 'investment', data=df, size=15)
plt.title(filename)
#plt.savefig(filename + "out_degree-investment.png")
plt.savefig(filename + "out_degree-rate.png")
plt.clf()

