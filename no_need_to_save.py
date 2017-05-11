# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
#avoid chinese mess
import xlrd
import glob,os
import openpyxl
import networkx as nx

import pandas as pd
import numpy as np
import glob,os
import math
from math import log10
from scipy.stats import rankdata
import collections
from collections import Counter
import matplotlib.pyplot as plt
from my_plots import hist_plt
from my_dataframes import drop_duplicate, drop_emptycell, stock_status
import datetime
from datetime import timedelta

import seaborn as sns



os.chdir('/Users/shine/Desktop/Internet_finance')

# # all = pd.read_excel('2016_3-8-ls.xlsx', sheetname = [0,1,2,3,4,5])
# all = pd.read_excel('2016_9-17_2-ls.xlsx', sheetname = [0,1,2,3,4,5])
#
# new = pd.DataFrame()
# for i, df1 in enumerate(all.itervalues()):
#     # df1['month_index'] = np.ones(len(df1))*i + 1
#     df1['month_index'] = np.ones(len(df1)) * i + 7
#     new = pd.concat([new, df1])
#
# print new
#
# # new.to_excel('2016_3-8-all.xlsx','sheet1')
# new.to_excel('2016_9-17_2-all.xlsx','sheet1')


all1 = pd.read_excel('2016_9-17_2-all-manual.xlsx', 'sheet1')
all2 = pd.read_excel('2016_3-8-all-manual.xlsx', 'sheet1')

all = pd.concat([all1, all2])

all.to_excel('2016_3-2017_2_platform-all.xlsx','sheet1')