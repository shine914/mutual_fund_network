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


os.chdir('/Users/shine/Desktop')
#file = glob.glob('test_us.xlsx') #for workingfile in filelist
#file = glob.glob('test_uk.xlsx')
#print file,'is working now'
#file = glob.glob('test_jp.xlsx')
#df = pd.ExcelFile(file[0])
df = pd.read_csv('test_jp.csv')

df = pd.DataFrame(df)

df_new = pd.DataFrame()
print len(df)

for i in np.arange(3, len(df), 1):
    #ted1 = df.iloc[i-1, 1]
    #bond1 = df.iloc[i-1, 2]
    fuop1 = df.iloc[i-1, 1]

    #ted2 = df.iloc[i-2, 1]
    #bond2 = df.iloc[i-2, 2]
    fuop2 = df.iloc[i-2, 1]

    #ted3 = df.iloc[i-3, 1]
    #bond3 = df.iloc[i-3, 2]
    fuop3 = df.iloc[i-3, 1]

    #ted4 = df.iloc[i, 1]
    #bond4 = df.iloc[i, 2]
    fuop4 = df.iloc[i, 1]

    #ercn = df.iloc[i, 4]

    df_new1 = pd.DataFrame({
                        #'ted1': [ted1],
                        #'ted2': [ted2],
                        #'ted3': [ted3],
                        #'ted4': [ted4],
                        #'bond1': [bond1],
                        #'bond2': [bond2],
                        #'bond3': [bond3],
                        #'bond4': [bond4]
                        'fuop1': [fuop1],
                        'fuop2': [fuop2],
                        'fuop3': [fuop3],
                        'fuop4': [fuop4]
                        #'ercn':[ercn]
                    })

    df_new = df_new.append(df_new1)

df_new.to_csv('test_jp_new.csv')
