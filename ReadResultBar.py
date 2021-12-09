# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 21:48:11 2021

@author: Diederik
"""

import itertools
import pandas as pd
Baseline_df = pd.read_json("XXX/paper/LSH/Baseline.json")
BaseNewDict_df = pd.read_json("XXX/paper/LSH/BaseNewDict.json")

BaseNewDictVsAlphaNumeric_df = pd.read_json("XXX/paper/LSH/BaseNewDictVsAlphaNumeric.json")
BaseNewDictVsAlphaNumericModelWords_df = pd.read_json("XXX/paper/LSH/BaseNewDictVsAlphaNumericModelWords.json")
BaseNewDictVsImportantWords_df = pd.read_json("XXX/paper/LSH/BaseNewDictVsImportantWords.json")
BaseNewDictVsModelWords_df = pd.read_json("XXX/paper/LSH/BaseNewDictVsModelWords.json")
BaseVsAlphaNumeric_df = pd.read_json("XXX/paper/LSH/BaseVsAlphaNumeric.json")
BaseVsAlphaNumericAndModelWords_df = pd.read_json("XXX/paper/LSH/BaseVsAlphaNumeric+ModelWords.json")
BaseVsImportantWords_df = pd.read_json("XXX/paper/LSH/BaseVsImportantWords.json")
BaseVsModelWords_df = pd.read_json("XXX/paper/LSH/BaseVsModelWords.json")



barResult_df = pd.DataFrame()
barResult_df["Base"] = Baseline_df["Run 1"]
barResult_df["ND: Base"] = BaseNewDict_df["Run 1"]
barResult_df["AN"] = BaseVsAlphaNumeric_df["Run 1"]
barResult_df["ND: AN"] = BaseNewDictVsAlphaNumeric_df["Run 1"]
barResult_df["MW"] = BaseVsModelWords_df["Run 1"]
barResult_df["ND: MW"] = BaseNewDictVsModelWords_df["Run 1"]
barResult_df["AN&MW"] = BaseVsAlphaNumericAndModelWords_df["Run 1"]
barResult_df["ND: AN&MW"] = BaseNewDictVsAlphaNumericModelWords_df["Run 1"]
barResult_df["IW"] = BaseVsImportantWords_df["Run 1"]
barResult_df["ND: IW"] = BaseNewDictVsImportantWords_df["Run 1"]

barResult_df_T = barResult_df.T
del barResult_df["Hyperparameters"]
#%% Plot
import matplotlib.pyplot as plt
import numpy as np
barResult_df_plot = barResult_df.iloc[:,:-8]
ax = barResult_df_plot.plot.bar(stacked=True)
