# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 21:48:11 2021

@author: Diederik
"""
'''     GREAT RESULTS 
        OptimizedRun3-NoAnFm! Contains other replace dictionary where units are removed!. Did not work with alphanumeric at all
        	          Hyperparameters	                                       
        Run 22	[0, 0, 2, False, False, False, True, 950, 160, 2, False]	
        lsh F1	LSH PQ	LSH PC	len(candidatePairs	len(duplicatesFound)	Fraction	(bands,rows)
        0.3710961421	0.24554294980000002	0.7593984962	1234	303	0.0009363552	[95, 10]


        
        Good results
        OptimizedRun2-AnFm! old dictionary Old dictionary with alpha numeric words!!
        	Hyperparameters	
        Run 44	[1, 1, 2, True, True, True, True, 900, 170, 2, False]	
                  lsh F1	LSH PQ	LSH PC	len(candidatePairs	len(duplicatesFound)	Fraction	(bands,rows)
                 0.35129461160000003	0.24368932040000002	0.6290726817	1030	251	0.0007815606	[150, 6]
        
        Good Results - RemoveD Units
        OptimizedWordRun-NoAnFm! -> Combine with Great Results to find Perfect Results
                	Hyperparameters
            Run 29	[2, 1, 2, False, False, False, True, 970, 190, 2, False]
            	          lsh F1	LSH PQ	LSH PC	len(candidatePairs	len(duplicatesFound)	Fraction	(bands,rows)
            Run 29	0.3717706364	0.24831649830000002	0.7393483709	1188	295	0.0009014505	[97, 10]



'''
import itertools


import pandas as pd
Result1_df = pd.read_json("XXX/paper/Results - 1st/Results1.json")
Result1_1_df = pd.read_json("XXX/paper/Results - 1st/Results1_1.json")

Result2_df = pd.read_json("XXX/paper/Results - 1st/Results2.json")
Result2_1_df = pd.read_json("XXX/paper/Results - 1st/Results2_1.json")

Result3_df = pd.read_json("XXX/paper/Results - 1st/Results3.json")
Result3_1_df = pd.read_json("XXX/paper/Results - 1st/Results3_1.json")

Result4_df = pd.read_json("XXX/paper/Results - 1st/Results4.json")
Result4_1_df = pd.read_json("XXX/paper/Results - 1st/Results4_1.json")



# hyperParameterList = ["minWordLength", "filterHitTH", "filterHitTH2", "modelWordsFeatureMap","onlyAlphaNumeric", 
                        # "hashRandomVectorBool", "candidatePairsByModelID", "sizeSignatureMatrix", 
                        # "numberBands", "repetitions", "herhaling","fastLSH"]

hyperParameterList = ["minWordLength", "filterHitTH", "filterHitTH2", "modelWordsFeatureMap","onlyAlphaNumeric", 
                      "hashRandomVectorBool", "candidatePairsByModelID", "sizeSignatureMatrix", 
                      "numberBands", "repetitions", "fastLSH", "importantWordsBool","clusteringBool", "epsilon", "threshold","herhaling"]

list_df = [Result1_df,Result1_1_df,Result2_df,Result2_1_df,Result3_df,Result3_1_df,Result4_df,Result4_1_df]
totalResult_df = pd.concat(list_df, axis=1)


hyperParameters = totalResult_df["HyperParameterList"]
hyperParameters = hyperParameters.iloc[0][0]
hyperParameters_df = pd.DataFrame(hyperParameters, index=hyperParameterList)
# hyperList = list(itertools.product(*hyperParameters))
del totalResult_df["HyperParameterList"]


def findMaxValue(column, dataframe = totalResult_df):
    T_df = dataframe.T
    maxV = T_df[column].max()
    index = list(T_df[column]).index(maxV)
    string = "Run " + str(index+1)
    return dataframe.iloc[:, [index]]

totalResult_df_transpose = totalResult_df.T
maxF1 = findMaxValue("lsh F1")
maxPQ = findMaxValue("LSH PQ")
maxPC = findMaxValue("LSH PC")
# maxDuplicates = findMaxValue("len(duplicatesFound)")

maxfraction = findMaxValue("Fraction")

#%%Check Bands and Rows
# theRange = []
# for i in range(1,1501):
#     if i == getBandAndRows(i,1500)[0]:
#         print("i: " + str(i) + " result: " + str(getBandAndRows(i,1500)))
#         theRange.append(i)

#%% Plotting Data
n = 900
totalResult_df_transpose["bandFraction"] = totalResult_df_transpose["bands"]/n 
totalResult_df_transpose.plot.scatter(x="Fraction", y="F1", alpha=0.5)
totalResult_df_transpose.plot.scatter(x="Fraction", y="PQ", alpha=0.5)
totalResult_df_transpose.plot.scatter(x="Fraction", y="PC", alpha=0.5)


totalResult_df_transpose.plot.scatter(x="Fraction", y="lsh F1", alpha=0.5)
totalResult_df_transpose.plot.scatter(x="Fraction", y="LSH PQ", alpha=0.5)
totalResult_df_transpose.plot.scatter(x="Fraction", y="LSH PC", alpha=0.5)


#%% Plotting Data 2
import matplotlib.pyplot as plt
def createPlot(y,yString,i):
    plt.figure(i)
    ax1 = plt
    ax1.scatter(totalResult_df_transpose.Fraction, y, s=50, c=totalResult_df_transpose.bandFraction, cmap='jet')
    ax1.grid( linestyle='-', linewidth=1)
    ax1.colorbar().set_label('fraction b/n', rotation=270)
    ax1.ylabel(yString)
    ax1.xlabel("Fraction")
    

createPlot(totalResult_df_transpose["F1"], "F1",1)
createPlot(totalResult_df_transpose["PQ"], "PQ",2)
createPlot(totalResult_df_transpose["PC"], "PC",3)
createPlot(totalResult_df_transpose["lsh F1"], "lSH F1",4)
createPlot(totalResult_df_transpose["LSH PQ"], "LSH PQ",5)
createPlot(totalResult_df_transpose["LSH PC"], "LSH PC",6)


#create scatterplot

# plt.colorbar()
