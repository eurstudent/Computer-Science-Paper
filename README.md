# Computer-Science-Paper
Detecting Duplicate TV's Using Minhash and LSH

Files in Repository:
   - Main_Clean.py             -> Main Script with full implementation
   - Main_LSH_GRIDSEARCH.py    -> Script for optimizing Parameters Until LSH
   - Main_Simmilarity.py       -> Script for optimizing Threshold/Epsilon fo Jaccard/Clustering
   - Read_Json.py              -> Script for reading JSON file with results and finding optimized hyper-parameters

# Main_Clean.py
----------------------------------------------------------------------------------------------------------------------------------------

### Hyper Parameters ###
minWordLength {
  Defines the length a model word needs to be
  }
filterHitTH   {
  Frequency of word in title that is needed to be taken into account of the title modelwords
  }
filterHitTH2  {  
  Frequency of word in FeaturesMap that is needed to be taken into account of the modelwords
  }
modelWordsFeatureMap {
  Boolean: Include words that occur in the featuremap in the modelword list
  }
onlyAlphaNumeric     {
  Boolean: Only Include Alphanumeric words in the modelwords. alphanumeric= strings containing letters and numbers and/or symbols
  }
hashRandomVectorBool {
  True: Use Hash functions Based on Random vector hash -> results in hash values ranging from (0 , 1000)
  False: Use Hash based on python native hash function -> results in hash values ranging from (10^5 , 10^7)
 }
sizeSignatureMatrix  {
  Determines the size of the signature matrix. Often chosen as ~50% of the products. 
  }
numberBands    {
  Determines the size of the bands. sizeSignatureMatrix = r * b . Larger value for b will result in smaller value for r which will result in the LSH making more candidate pairs   because less rows means larger probability of being similair and therefor becoming a pair. Increasing b will therefor increase False negatives. (Pair quality often decreases     as Pair completeness increases.
  }
repetitions    {
  Amount of repetitions LSH is repeated for a shuffeled signature matrix. Same reasaning as above applies.
  }
fastLSH   {
  Fast LSH is a different Locallity sensitive Algorithm which is less accurate but is very fast. -> used for quick comparing new implementations
  }
candidatePairsByModelID {
  Boolean: If True algorithm will try to find modelID's in title and/or Featuremap. After LSH products will be checked if two times the same modelID is found. To keep this        method robust and scalable. The candidate pair must still go through Similarity clustering
  }
threshold  {
  Jaccard Similarity threshold. Determines the amount of Candidate pairs that is said to be a true pair.
  }
Clustering  {
  Boolean: if True Apply single linkage hierachial clustering instead of filtering on jaccard Similarity 
  }
epsilon   {
  epsilon is the threshold to which the clusters are made. A low threshold will result in sharper conditions for the clustering method
  }
  
# Script
In script Multiple Comments are added to 
