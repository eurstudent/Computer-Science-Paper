# Computer-Science-Paper
Detecting Duplicate TV's Using Minhash and LSH

Files in Repository:
   - Main_Clean.py             -> Main Script with full implementation
   - Main_LSH_GRIDSEARCH.py    -> Script for optimizing Parameters Until LSH
   - Main_SIM_GriDSEARCH.py       -> Script for optimizing Threshold/Epsilon fo Jaccard/Clustering
   - Read_Json.py              -> Script for reading JSON file with results and finding optimized hyper-parameters

# Main_Clean.py
----------------------------------------------------------------------------------------------------------------------------------------

### Hyper Parameters ###
minWordLength {\
  Defines the length a model word needs to be\
  }\
filterHitTH   {\
  Frequency of word in title that is needed to be taken into account of the title modelwords\
  }\
filterHitTH2  {  \
  Frequency of word in FeaturesMap that is needed to be taken into account of the modelwords\
  }\
modelWordsFeatureMap {\
  Boolean: Include words that occur in the featuremap in the modelword list\
  }\
onlyAlphaNumeric     {\
  Boolean: Only Include Alphanumeric words in the modelwords. alphanumeric= strings containing letters and numbers and/or symbols\
  }\
hashRandomVectorBool {\
  True: Use Hash functions Based on Random vector hash -> results in hash values ranging from (0 , 1000)\
  False: Use Hash based on python native hash function -> results in hash values ranging from (10^5 , 10^7)\
 }\
sizeSignatureMatrix  {\
  Determines the size of the signature matrix. Often chosen as ~50% of the products. \
  }\
numberBands    {\
  Determines the size of the bands. sizeSignatureMatrix = r * b . Larger value for b will result in smaller value for r which will result in the LSH making more candidate pairs   because less rows means larger probability of being similair and therefor becoming a pair. Increasing b will therefor increase False negatives. (Pair quality often decreases     as Pair completeness increases.\
  }\
repetitions    {\
  Amount of repetitions LSH is repeated for a shuffeled signature matrix. Same reasaning as above applies.\
  }\
fastLSH   {\
  Fast LSH is a different Locallity sensitive Algorithm which is less accurate but is very fast. -> used for quick comparing new implementations\
  }\
Important Words {\
  Boolean: If True algorithm Will add prior beliefs to the model words which will be given a weight in the algorithm.
  }\
threshold  {\
  Jaccard Similarity threshold. Determines the amount of Candidate pairs that is said to be a true pair.\
  }\
Clustering  {\
  Boolean: if True Apply single linkage hierachial clustering instead of filtering on jaccard Similarity \
  }\
epsilon   {\
  epsilon is the threshold to which the clusters are made. A low threshold will result in sharper conditions for the clustering method\
  }
  
# Script\
In script general clarification is found at the beginning of each part. \
\
Additional Clarification:\
#Data Cleaning
Data is Properly cleaned and columns are automatically added together using FuzzyWuzzy Wratio. This is done for all columns with a high threshold and for the brand Column with a slightly lower threshold because I wanted to find all brands in the Featuremap so they could be added to the title Column.\

The unit replacements are shown below and in the code the XXX is replaced and used to concatenate the unit to the correct number
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
    '(\s[^\w])': ' ',            # Remove special charecters befor string 
    '([^\w\s]|_)+(?=\s|$)': ' ', # Remove special charecters after string
    'nan'    : ' ',
    'measured diagonally'   :  'XXXDIAG',
    'measureed diagonal'   :  'XXXDIAG',
    'diagonal'   :  'XXXDIAG',
    'diag'   :  'XXXDIAG',
    'class'  :   ''
}

#CandidatePair Cleaning
The candiate pair list that is the result of LSH is filtered befor clustering/similarity filtering. This is done because it contains duplicates. In addition to duplicate filtering it is also checked that the candidates are from different shops (assumption: The shop does not sell the same product twice. is allowed to make but not true) and that the candidate pairs have the same brand only remove candidate pair if both brands are present and different.

### Aditional implementations ###
   - Two hash Functions: One hash function which creates smaller numbers but uses the proposed hash function which implements the modulus of a prime number. The other hash function creates large numbers which are produces to reduce the amount of hash collisions. They often differ in performance for different Hyperparameters. Trying them both for the best result is why they are both added.
   - Two Types of LSH: fast LSH is quick but does not as a good job as the hardcoded LSH. I used the fast LSH to quickly run my program and see the effect of my implementations.
   - Simmilarity: Two types of simmilarity filtering. One is simple and based on the jaccard simmilarity score and an applied threshold. The other is based on hierachial clustering. The reason why i have added them both is because it took me a while to get the hierachial clustering working.
