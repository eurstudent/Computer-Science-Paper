# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 17:50:08 2021

@author: Diederik
"""
# %% Importing Functions That are needed Througout de script
import collections
import itertools
import math
import time
from collections import Counter
import numpy as np
import pandas as pd
import random
from itertools import combinations
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

# Creating callable Functions
# %% Simple Uitility Functions
# Check if all elements in list are equal
def allEqual(iterable):
    iterator = iter(iterable)
    try:
        firstItem = next(iterator)
    except StopIteration:
        return True
        
    for x in iterator:
        if x!=firstItem:
            return False
    return True

# get intersection between two lists
def getIntersection(list1, list2):
    return set([tuple(sorted(ele)) for ele in list1]) & set([tuple(sorted(ele)) for ele in list2])

# get difference between two lists
def getDifference(list1, list2):
    return set(tuple(x) for x in [list(ele) for ele in list1]).symmetric_difference(set(tuple(x) for x in list2))

#Flatten any type of nested list,set,tuple
def flatten(t):
    return [item for sublist in t for item in sublist]

#Remove Duplicate paris from list
def removeDuplicates(lst):
    return list(set([i for i in lst]))

# Change nested lists to a list of tuples
def listOfTuples (listToChange):
    return [tuple(ele) for ele in listToChange]

def isPrime(x):
  for j in range(2,int(x**0.5)+1):
    if x%j==0:
      return False
  return True

def findPrimeNum(num):
  for i in range(num,10000,1):
    if isPrime(i):
      return i

#Convert a list of intergers to a list of ModelID's
def listDuplicatesByModelID(listToCheck):
    listDuplicates = []
    for ls in range(len(listToCheck)):
        listDuplicates.append([total_df_uncleaned['index'][listToCheck[ls][0]], total_df_uncleaned['index'][listToCheck[ls][1]]])
    return listDuplicates

def getListTruePairs(dataframe):
    listOfTruePairs = []
    for i in list(dataframe.index):
        for j in list(dataframe.index):
            if i >= j:
                continue
            else:
                if dataframe['index'][i] == dataframe['index'][j]:
                    pair = tuple([i,j])
                    listOfTruePairs.append(pair)
    return listOfTruePairs

def duplicateIndices(listToCheck):
    unique_entries = set(listToCheck) 
    indices = { value : [ i for i, v in enumerate(listToCheck) if v == value ] for value in unique_entries }
    del indices[""]
    return list(indices.values())

def getCleanList(listToClean):
    listClean = [x.strip(' ') for x in listToClean]
    listClean = set(listClean)
    listClean = list(listClean)
    listClean.remove("")
    return listClean

def GetWebsite(listToGetWebsite):
    listWebsite = []
    for ls in range(len(listToGetWebsite)):
        listWebsite.append(total_df.shop[listToGetWebsite[ls]])
    return listWebsite

def GetBrand(listToGetBrand):
    listBrand = []
    for ls in range(len(listToGetBrand)):
        listBrand.append(total_df.Brand[listToGetBrand[ls]])
        brandList = [x.strip(' ') for x in listBrand]
    return brandList
# %% Loading Data
def getDataFromDict(website, dataframe, rows, index):
    string_df = ['1st instance', '2nd instance', '3rd instance', '4th instance']
    for st in string_df:
        kboolean = dataframe[st].notnull()
        j = 0
        for k in kboolean:
            if k == True:
                string = kboolean.index[j]
                if dataframe[st][string]['shop'] == website:
                    index.append(dataframe[st][string]['modelID'])
                    d1 = {"title": dataframe[st][string]['title']}
                    d2 = dataframe[st][string]['featuresMap']
                    d3 = dict(d1, **d2)
                    dshop = {"shop": dataframe[st][string]['shop']}
                    d4 = dict(dshop, **d3)
                    rows.append(d4)
            j += 1
    return rows, index

# Loading Data
"""
    Loading the data from the JSON file and creating a dataframe with all avaiable data -> total_df
    Dataframe contains, title and featuresmap
"""
startTime = time.time()
# Load data
tv_df = pd.read_json(
    'C:/Users/Diede/Documents/Erasmus/Master Econometrie/Computer Science for Business Analytics/paper/TVs-all-merged.json',
    orient='index')
tv_df.columns = ['1st instance', '2nd instance', '3rd instance', '4th instance']

rows = []
index = []
columns = ["title"]
websiteList = ['bestbuy.com', 'newegg.com', 'amazon.com', 'thenerds.net']
for website in websiteList:
    rows, index = getDataFromDict(website, tv_df, rows, index)

total_df = pd.DataFrame(rows, index=index)
total_df.reset_index(drop=False, inplace=True)
total_df_uncleaned = total_df.copy()

listTruePairs = getListTruePairs(total_df)

del website, rows, index, columns, websiteList
# %% Preprocessing and Cleaning of the data
# Create a list of only alphanumeric type of strings
def alphaNumericList(listToClean):
    cleanedList = []
    for word in listToClean:
        if (any(chr.isalpha() for chr in word) and any(chr.isdigit() for chr in word)) == True:
            cleanedList.append(word)
    return cleanedList

def preProcessData(dataframe, replaceDict):
    dataframe = dataframe.apply(lambda x: x.astype(str).str.lower())
    dataframe = dataframe.replace(replaceDict, regex=True)
    dataframe = dataframe.replace([' XXX','XXX'], value = '', regex = True)
    dataframe.columns = dataframe.columns.str.rstrip(":")
    dataframe = dataframe.groupby(level=0, axis=1).sum()
    return dataframe

def findInchDiag(listToCheck):
    check = process.extract("INCHDIAG",listToCheck,limit=2, scorer=fuzz.partial_token_sort_ratio)
    if len(check) == 1 or len(check) == 0:
        return listToCheck
    if check[0][1] > 0.9:
        if check[0][0][:-4] == check[1][0]:
            listToCheck.remove(check[0][0])
    return listToCheck

#Calculate correct combination for bands and rows
def getBandAndRows(band, lengthInputMatrix):
    matchNotFound = True
    while (matchNotFound):
        r = range(math.ceil(lengthInputMatrix / band), 0, -1)
        for i in range(len(r)):
            if r[i] * band == lengthInputMatrix:
                row = r[i]
                matchNotFound = False
        band = band - 1
    return band + 1, row

# %% Header String Comparison
def headerStringComparison(threshold, combine = False, strCompare = total_df.columns):
    columnStringComparison = []
    for str2Match in strCompare:
        ratioFuzzy = process.extract(str2Match,strCompare,limit=5, scorer=fuzz.WRatio)
        filtered = []
        for compare in ratioFuzzy:
            if compare[1] >= threshold:
                filtered.append(compare)
        columnStringComparison.append(filtered)
    if combine == True:
        total_df = combineColumns(columnStringComparison)
        

def combineColumns(columnStringComparison):
    if len(columnStringComparison[0]) != 1:
        for i in range(1,len(columnStringComparison[0])):
            total_df[columnStringComparison[0][0][0]] = total_df[columnStringComparison[0][0][0]] + [" "] + total_df[columnStringComparison[0][i][0]]
            del total_df[columnStringComparison[0][i][0]]
            total_df[columnStringComparison[0][0][0]].str.strip()
            
# %% ModelID Comparison
def longestStringInList(titleList):
    listWords = ["DIAG","HERTZ","INCH","POUNDS"]
    titleListAlphaNumeric = alphaNumericList(titleList)
    for ls in listWords:
        titleListAlphaNumeric = [x for x in titleListAlphaNumeric if ls not in x]
    if titleListAlphaNumeric:
        longest_string = max(titleListAlphaNumeric, key=len)
        if len(longest_string) <= 5:
            longest_string = ""
    else:
        longest_string = ""
    return longest_string

def cleanCandidatesByModelID(proposedCandidates):
    candidatelist = []
    proposedCandidates = [x for x in proposedCandidates if len(x) != 1]
    for i in range(len(proposedCandidates)):
        possibleCombinations = list(combinations(proposedCandidates[i], 2))
        candidatelist.append(possibleCombinations)
    return flatten(candidatelist)

# %% Creating Model Words

def importantWords():
    importantWords = []
    #Brand
    brandList = getCleanList(total_df.Brand)
    importantWords.append(brandList)
    importantWords.append(brandList)
    modelID = getCleanList(total_df["Extracted Model ID"])
    importantWords.append(modelID)
    importantWords.append(modelID)
    importantWords.append(modelID)
    return importantWords
    
    
def createTokenTitleList(dataFrameTitle, filterHits,minWordLength):
    tokenList = []
    for i in range(len(dataFrameTitle)):
        tokenCandidateList = dataFrameTitle[i]
        tokenList.extend(tokenCandidateList)

    tokenCounts = Counter(tokenList).most_common()
    tokenCounts = list(filter(lambda x: len(x[0]) >= minWordLength, tokenCounts))
    tokenCounts = list(filter(lambda x: x[1] >= filterHits, tokenCounts))
    ModelWordList = [i[0] for i in tokenCounts]
    return ModelWordList
# %% Hash 
# Hash with Random Vector    
def hashIt(a, b, r):
    return (int(a * r + b)) % findPrimeNum(inputMatrix.shape[0])
def hashFactoryRandom(n):
    return lambda x: hashIt(a[n], b[n], x)


# Hash with String
def hashFunctionsString(length):
    def hashFactoryString(ni):
        return (lambda x: hash("hashIt" + str(ni) + str(x)))
    return [hashFactoryString(i) for i in range(length)]

# %% Minhash Functions
def minhash(data, hashfuncs):
    '''
    Returns signature matrix
    '''
    rows, cols, sigrows = len(data), len(data[0]), len(hashfuncs)
    curr_min = 99999

    sigmatrix = []
    for i in range(sigrows):
        sigmatrix.append([curr_min] * cols)

    for r in range(rows):
        hashvalue = [x(r) for x in hashfuncs]
        # if data != 0 and signature > hash value, replace signature with hash value
        for c in range(cols):
            if data[r][c] == 0:
                continue
            for i in range(sigrows):
                if sigmatrix[i][c] > hashvalue[i]:
                    sigmatrix[i][c] = hashvalue[i]
    return sigmatrix

# %% LSH HardCoded
def sameStoreAndBrand(candidatePairs):
    candidateList = []
    for i in range(len(candidatePairs)):
        possibleCombinations = list(combinations(candidatePairs[i], 2))
        for ls in range(len(possibleCombinations)):
            websiteList = GetWebsite(possibleCombinations[ls]) 
            brandList = GetBrand(possibleCombinations[ls]) 
            if allEqual(websiteList):
                continue
            if allEqual(brandList) == False:
                if brandList[0] == "" or brandList[1] == "":
                    candidateList.append(set(possibleCombinations[ls]))
                else:
                    continue
            else:
                candidateList.append(set(possibleCombinations[ls]))
    return candidateList

def cleanCandidatePairs(candidatePairsList):
    candidatePairsList = [set(item) for item in set(frozenset(item) for item in candidatePairsList)]  # Remove Duplicates
    candidatePairsList = sameStoreAndBrand(candidatePairsList)
    candidatePairsList = [set(item) for item in set(frozenset(item) for item in candidatePairsList)]
    return candidatePairsList

# %% Jaccard Similarity
def jaccard(xlist, ylist):
        intersection = np.logical_and(xlist, ylist)
        union = np.logical_or(xlist, ylist)
        sim = intersection.sum() / union.sum()
        return sim  
    
#Filter the similarity dictionary by a chosen threshold
def filterSimilarity(similarity,threshold):
    #similarity1 = {key: val for key, val in similarity.items() if val > threshold}
    return {key: val for key, val in similarity.items() if val > threshold}

# %% HyperParameters
# ModelWord hyperParameters
minWordLength = 1
filterHitTH = 1    # Frequency of words in title to be taken into account of the title modelwords
filterHitTH2 = 2  # Frequency of words in FeaturesMap to be taken into account of the modelwords
modelWordsFeatureMap = False
onlyAlphaNumeric = True
#Hash hyper Parameters
hashRandomVectorBool = True
# LSH hyperParameters
sizeSignatureMatrix = 1000
numberBands = 100  # Number of Bands
repetitions = 1
fastLSH = False
# Similarity hyperParameters
threshold = 0.6  # JS threshold for being an exact pair
# CLustering
clusteringBool = False
epsilon = 0.2
importantWordsBool = True

# %% Pre-cleaning of the data
tValue = time.time()
"""
    Pre cleaning The dataset.
    Convert all letters to lowercase
    units converted to: INCH, HZ, POUNDS, MM, NIT, WATT
    Removing interpunctuation
    Cleaning Dataframe of nans to make it more readible
"""
replaceDict = {
    'inches' : 'XXXINCH',
    '-inch'  : 'XXXINCH',
    'inch'   : 'XXXINCH',
    '"'      : 'XXXINCH',
    'hertz'  : 'XXXHERTZ',
    '-hz'    : 'XXXHERTZ',
    'hz'     : 'XXXHERTZ',
    'lbs'    : 'XXXPOUNDS',
    'lb'     : 'XXXPOUNDS',
    'pounds' : 'XXXPOUNDS',
    'mm'     : 'XXXMM',
    'cd/m2'  : 'XXXNIT',
    'cd/mâ2' : 'XXXNIT',
    'cd/mâ²' : 'XXXNIT',
    'nit'    : 'XXXNIT',
    'watt'      : 'w',
    '(\s[^\w])': ' ',                
    '([^\w\s]|_)+(?=\s|$)': ' ',
    'nan'    : ' ',
    'measured diagonally'   :  'XXXDIAG',
    'measureed diagonal'   :  'XXXDIAG',
    'diagonal'   :  'XXXDIAG',
    'diag'   :  'XXXDIAG',
    'class'  :   ''
}

replaceDictBU = {
    'inches' : '',
    '-inch'  : '',
    'inch'   : '',
    '"'      : '',
    'hertz'  : '',
    '-hz'    : '',
    'hz'     : '',
    'lbs'    : '',
    'lb'     : '',
    'pounds' : '',
    'mm'     : '',
    'cd/m2'  : '',
    'cd/mâ2' : '',
    'cd/mâ²' : '',
    'nit'    : '',
    'watt'      : 'w',
    '(\s[^\w])': ' ',                
    '([^\w\s]|_)+(?=\s|$)': ' ',
    'nan'    : ' ',
    'measured diagonally'   :  '',
    'measureed diagonal'   :  '',
    'diagonal'   :  '',
    'diag'   :  '',
    'class'  :   ''
}

print("Preprocessing the data ...")
total_df = preProcessData(total_df, replaceDict)

endPreProcess = time.time() - startTime
del replaceDict, replaceDictBU
# %% Combining similair headers to clean the dataFrame even more
headerStringComparison(95,combine = True)
headerStringComparison(80,combine = True, strCompare = ["Brand"])
print("Preprocessing the data Done")
# %% Create columns, title list and word list
"""
     adding two columns to the total_df dataframe
     title list contains all words in the title. No Duplicates are allowd and is filtered for wordlist
     word list contains all words in the features and is filtered for wordlist 
"""
print("Creating title/FeratureMap list ...")
tValue = time.time()

cols = total_df[total_df.columns[-(len(total_df.columns) - 2):]].columns
featuresString_df = total_df[total_df.columns[-(len(total_df.columns) - 2):]][cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

wordFilterList = ["-", "/", "newegg.com", "thenerds.net", "bestbuy.com", "best", "buy", "tv", "and", "hdtv", "yes", "no", "lcd", "ledlcd", "led-lcd"]

featureMapWordsList = []
for i in range(len(featuresString_df)):
    new_words = [x.strip(' ') for x in featuresString_df[i].split()]
    new_words = [word for word in new_words if word not in wordFilterList]
    new_words = list(set(new_words))
    if "" in new_words: new_words.remove("")
    if onlyAlphaNumeric: new_words = alphaNumericList(new_words)
    new_words = findInchDiag(new_words) 
    featureMapWordsList.append(new_words)
            
titleWordsList = []
for i in range(len(total_df['title'])):
    new_words = [x.strip(' ') for x in total_df['title'][i].split()]
    if str.isspace(total_df.Brand[i]) == False:
        new_words.append(total_df.Brand[i])
    new_words = [x.strip(' ') for x in new_words]
    new_words = [word for word in new_words if word not in wordFilterList]
    new_words = list(set(new_words))
    if "" in new_words: new_words.remove("")
    if onlyAlphaNumeric: new_words = alphaNumericList(new_words)
    new_words = findInchDiag(new_words) 
    titleWordsList.append(new_words)

for i in range(len(titleWordsList)):
    set1 = set(titleWordsList[i])
    set2 = set(featureMapWordsList[i])
    set2.difference_update(set1)
    titleWordsList[i] = list(set1)
    featureMapWordsList[i] = list(set2)

total_df["title list"] = titleWordsList
total_df["word list"] = featureMapWordsList
print("Creating title/FeratureMap list Done")

# %% add pairs due to modelID in the title
modelID = []
for i,titleList in enumerate(total_df['title list']):
    modelID.append(longestStringInList(titleList))
total_df["Extracted Model ID"] = modelID
# candidatesByModelID = duplicateIndices(total_df["Extracted Model ID"])
# candidatesByModelID = cleanCandidatesByModelID(candidatesByModelID)
# %% Creating model Words
"""
    Creating the model words by Counting the freqeuncy and Filtering for wordlength. 
    Model words can be created using the title and the featurewords can be added
"""
print("Creating ModelWords list ...")

modelWordsTitleList = createTokenTitleList(total_df['title list'], filterHitTH,minWordLength)
modelWordsList = createTokenTitleList(total_df['word list'], filterHitTH2,minWordLength)
totalModelWords = modelWordsTitleList + sorted(set(modelWordsList) - set(modelWordsTitleList))
if importantWordsBool == True:
        modelWordsTitleList.extend(flatten(importantWords()))
endModelWords = time.time() - tValue
print("Creating ModelWords list Done")
# %% Create Input matrix
"""
    Creating the Input matrix. Input Matrix contains Binary vectors when word is in the title or featuremap of an particulair item
"""
tValue = time.time()
print("Creating the input Matrix ...")
wordListSubset_df = total_df[['title list', 'word list']].copy()

# Creating Modelwords featuremap
if modelWordsFeatureMap == True:
    indexList = totalModelWords.copy()
else:
    indexList = modelWordsTitleList.copy()

N = len(indexList)
M = len(wordListSubset_df.index)
inputMatrix = np.zeros((N, M))
modelWordListToCheck = indexList.copy()

for index in range(len(wordListSubset_df)):  # for index in (wordListSubset_df.index)):  
    for column in range(len(modelWordListToCheck)):
        if modelWordListToCheck[column] in wordListSubset_df['title list'][index]:
            inputMatrix[column][index] = 1
        else:
            if modelWordsFeatureMap == True:
                if modelWordListToCheck[column] in wordListSubset_df['word list'][index]:
                    inputMatrix[column][index] = 1

endInputMatrix = time.time() - tValue
tValue = time.time()

print("Creating the input Matrix Done")
del N,M
# %% Creating HashFunctions
"""
    Creating two Types of hash functions, which is used in the gridsearch to define which gives the best Result
"""
a = random.sample(range(inputMatrix.shape[0]), inputMatrix.shape[0])
b = random.sample(range(inputMatrix.shape[0]), inputMatrix.shape[0])

hashesRandomVector = [hashFactoryRandom(i) for i in range(inputMatrix.shape[0])]
hashesString = hashFunctionsString(inputMatrix.shape[0])

if hashRandomVectorBool == True:
    hashes = hashesRandomVector.copy()
else:
    hashes = hashesString.copy()
    
del hashesRandomVector, hashesString
# %% Create Signature Matrix
"""
    Creating the signatureMatrix
"""

signatureMatrix = np.asarray(minhash(inputMatrix, hashes[:sizeSignatureMatrix]))
endSignatureMatrix = time.time() - tValue

# %% Find duplicates - Hardcoded
"""
    Finding Duplicates Using LSH -> hardcoded
"""
tValue = time.time()


lengthSigMatrix = signatureMatrix.shape[0]
bands, rows = getBandAndRows(numberBands, lengthSigMatrix)

t = math.pow((1 / bands), (1 / rows))
print("bands. rows: " + str(bands) + ", " + str(rows))
print("threshold value: " + str(t))

buckets = {}
for rep in range(repetitions):
    shuffeledSM = signatureMatrix.copy()
    np.random.shuffle(shuffeledSM)
    
    for p in range(shuffeledSM.shape[1]):
        for b in range(bands):
            h = str(rep) +' '+ str(b) +' '+(str([round(item) for item in shuffeledSM[:,p][b:b+rows]]))
            if h not in buckets.keys():
                buckets[h] = [p]
            else:
                if p not in buckets[h]:
                    buckets[h].append(p)
                    
buckets = {k:v for k,v in buckets.items() if len(v) >= 2}

candidatePairsHC = list(buckets.values())

candidatePairsClean = cleanCandidatePairs(candidatePairsHC)
del lengthSigMatrix
# %% Find duplicates - Fast LSH
"""
    Finding Duplicates Using LSH --> Fast and Clean But not efficient
"""
if fastLSH == True:
    def fastCandidatePairsFunction(sig_mat, b, r):
        n, d = sig_mat.shape
        assert (n == b * r)
        hashbuckets = collections.defaultdict(set)
        bands = np.array_split(sig_mat, b, axis=0)
        for i, band in enumerate(bands):
            for j in range(d):
                # The last value must be made a string, to prevent accidental
                # key collisions of r+1 integers when we really only want
                # keys of r integers plus a band index
                band_id = tuple(list(band[:, j]) + [str(i)])
                hashbuckets[band_id].add(j)
        candidate_pairs = set()
        for bucket in hashbuckets.values():
            if len(bucket) > 1:
                for pair in itertools.combinations(bucket, 2):
                    candidate_pairs.add(pair)
        return candidate_pairs
    
    fastCandidatePairs = []
    for rep in range(repetitions):
        fastCandidatePairs.extend(fastCandidatePairsFunction(signatureMatrix, bands, rows))
    
    fastCandidatePairsClean = cleanCandidatePairs(fastCandidatePairs)

endLSH = time.time() - tValue
# %% Calculating results after LSH
tValue = time.time()
if fastLSH == True:
    candidatePairs = fastCandidatePairsClean.copy()
else:
    candidatePairs = candidatePairsClean.copy()
    
candidatePairs = listOfTuples(candidatePairs)

from sklearn.cluster import AgglomerativeClustering
def mainFunction(hyperSimParameterList ,tValue = tValue, candidatePairs=candidatePairs):
    clusteringBool, epsilon, threshold = hyperSimParameterList
    
    # %% Comparing Pairs
    """
        Comparing the pairs using jaccard Similarity to calculate a simmilarity score
    """
    print("Applying Jaccard for the similarity Matrix ...")
    finalpairs = list(candidatePairs) #FCP3
    
    if clusteringBool == True:
        def jacsim(a,b):
            v1 = signatureMatrix[:,a]
            v2 = signatureMatrix[:,b]
            v3 = v1-v2
            v3[v3!=0] = 1
            jdis  = sum(v3)/len(v3)
            return 1 - jdis
        
        DistanceMatrix = np.ones((len(total_df),len(total_df)))*10000000
        for i in range(DistanceMatrix.shape[0]):
            for j in range(DistanceMatrix.shape[1]):
                if tuple([i,j]) in finalpairs:
                    DistanceMatrix[i][j] = 1 - jacsim(i,j)
        
        clustering = AgglomerativeClustering(affinity='precomputed', linkage='single', distance_threshold = epsilon, n_clusters = None).fit_predict(DistanceMatrix)
        bucketscl = {}
        for i in range(len(clustering)):
            if clustering[i] not in bucketscl.keys():
                bucketscl[clustering[i]] = [i]
            else:
                bucketscl[clustering[i]].append(i)
        bucketscl = {k:v for k,v in bucketscl.items() if len(v)>=2}        
                
        result = []
        for key in bucketscl.keys():
            for comb in list(combinations(bucketscl[key],2)):
                if comb[0]> comb[1]:
                    comb = tuple([comb[1],comb[0]])
                result.append(comb)
    
    else:
        similarity = {}    
        for pair in finalpairs:            
            pairToCheck = list(pair)
            l1 = inputMatrix[:,pairToCheck[0]]
            l2 = inputMatrix[:,pairToCheck[1]]        
            similarity[(pairToCheck[0], pairToCheck[1])] = jaccard(l1, l2)

     
    # %% Cleaning Results
    """
        Filtering the results of the jaccard similarity
    """
    if clusteringBool == True:
        listSimilarity = result.copy()
    else:
        similarityFiltered = filterSimilarity(similarity,threshold)
        listSimilarity = list(similarityFiltered.keys())

    
    
    # %%Calculate Scores
    possibleCombinations = (math.factorial(len(total_df)))/(2*(math.factorial(len(total_df)-2)))
    fraction = len(candidatePairs)/possibleCombinations
    
    lshDuplicatesFoundList = getIntersection(candidatePairs, listTruePairs)
    if len(candidatePairs) != 0:
        lshPairQuality = len(lshDuplicatesFoundList) / len(candidatePairs)
        lshPairCompleteness = len(lshDuplicatesFoundList) / len(listTruePairs)
        lshF1 = 2 * lshPairQuality * lshPairCompleteness / ( lshPairQuality + lshPairCompleteness )
    else:
        lshPairQuality = 0
        lshPairCompleteness = 0
        lshF1 = 0
    
    DuplicatesFoundList = getIntersection(listSimilarity, listTruePairs)
    if len(finalpairs) != 0:    
        pairQuality = len(DuplicatesFoundList) / len(listSimilarity)
        pairCompleteness = len(DuplicatesFoundList) / len(listTruePairs)
        F1 = 2 * pairQuality * pairCompleteness / ( pairQuality + pairCompleteness)
    else:
        pairQuality = 0
        pairCompleteness = 0
        F1 = 0
    hyperSimParameterList = [clusteringBool, epsilon, threshold]
    return [hyperSimParameterList, lshF1, lshPairQuality,lshPairCompleteness,F1,pairQuality,pairCompleteness, len(candidatePairs),len(lshDuplicatesFoundList),len(lshDuplicatesFoundList),fraction,bands, rows]#,F1, pairQuality, pairCompleteness]

#%% Fixed Parameters
import itertools

# ModelWord hyperParameters
minWordLength = 0
filterHitTH = 0  # Frequency of words in title to be taken into account of the title modelwords
filterHitTH2 = 2  # Frequency of words in FeaturesMap to be taken into account of the title modelwords
modelWordsFeatureMap = False
onlyAlphaNumeric = True
#Hash hyper Parameters
hashRandomVectorBool = False
# LSH hyperParameters
sizeSignatureMatrix = 900
numberBands = 170  # Number of Bands
repetitions = 1
fastLSH = False

# Optimize Parameters
clusteringBool = [True]
epsilon = [0.1,0.12,0.14,0.16,0.18,0.2] #[0.05,0.1,0.25,0.5,0.75,0.8]
threshold = [0.6]#,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8] #,0.3,0.4,0.5,0.6,0.7,0.8] #,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

hyperSimParameterList = [clusteringBool, epsilon,threshold]
hyperList = list(itertools.product(*hyperSimParameterList))
# hyperList = hyperList[:2]
# optimize_df = optimizeREAD_df
optimizeIndex = ["Hyperparameters", "lsh F1", "LSH PQ" , "LSH PC","F1","PQ","PC", "len(candidatePairs", "len(DuplicatesFoundList)", "lshDuplicatesFoundList","Fraction","bands","rows"]
optimizeSim_df = pd.DataFrame(index=optimizeIndex)
#%%
optimizeSim_df["HyperParameterList"] = [hyperSimParameterList, "", "" , "", "", "","","","","","","",""]
i = 1
for j in hyperList:
    nameString = "Run " + str(i)
    try:
        optimizeSim_df[nameString] = mainFunction(j)
        print(nameString + " succesfull")
    except:
        print(nameString + " unsuccesfull")
        pass
    i += 1
    optimizeSim_df.to_json("C:/Users/Diede/Documents/Erasmus/Master Econometrie/Computer Science for Business Analytics/paper/OptimizeNCNK2.json")
