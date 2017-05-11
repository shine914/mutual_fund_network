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




os.chdir('/Users/shine/work_hard/financial_network/data/threshold/stock_alpha')
df = pd.read_csv('bipar_nbr_weight0.9.csv')
#df = pd.read_csv('bipar_nbr_weight0.1.csv')
df = pd.DataFrame(df)

a = df.describe()

a.to_csv('bipar_nbr_weight0.1_summary.csv')


hist_plt(df['weight'], None, 'weight', '')
hist_log_bin(df['weight'], 'weight', 70000, 3000000, 50)
#hist_log_bin(df['weight'], 'weight', 1, 110, 20)

hist_plt(df['ave_nbr_weight'], None, 'ave_nbr_weight', '')
hist_log_bin(df['ave_nbr_weight'], 'ave_nbr_weight', 1500, 652000, 50)
#hist_log_bin(df['ave_nbr_weight'], 'ave_nbr_weight', 1, 110, 20)

hist_plt(df['num_of_nbrs'], None, 'num_of_nbrs', '')
hist_log_bin(df['num_of_nbrs'], 'num_of_nbrs', 1, 65, 10)
#hist_log_bin(df['num_of_nbrs'], 'num_of_nbrs', 1, 15, 5)

hist_plt(df['var_nbr_weight'], None, 'var_nbr_weight', '')
hist_log_bin(df['var_nbr_weight'], 'var_nbr_weight', 1, 170000000000, 50)
#hist_log_bin(df['var_nbr_weight'], 'var_nbr_weight', 1, 2600, 30)


