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



def drop_emptycell(df):
    print 'original data row number', df.shape[0]  # Count_Row
    df.replace('', np.nan, inplace=True)  # 删除原始数据中的空白单元格，不是NaN
    df.dropna(axis=0, inplace=True)
    print 'after delete missing data, row number is', df.shape[0]  # Count_Row
    return df

def drop_duplicate(df):  # after drop empty cells
    df = df.drop_duplicates()  # df.drop_duplicates(subset[])
    print 'after delete duplicate data, row number is', df.shape[0]  # Count_Row
    # print df
    return df


def summary(filename, df):
    summary = df.describe()
    summary = pd.DataFrame(summary)
    summary.to_excel(filename + '_summary.xlsx', sheet_name='Sheet1')


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
