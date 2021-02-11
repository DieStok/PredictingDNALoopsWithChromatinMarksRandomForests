#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:53:47 2019

@author: dstoker

#update 27-07-2019:
This script was made to see what would happen if we cluster the feature data,
or perform different normalisations on it, and as a local test of feature calculation
processing. In the end, the functions in this script were transferred to the script
TrainRandomForestHPC_IndicesTrainTestInput_27_05_2019.py and no normalisation takes
place on the features.


Process the features, check for normalcy, etc. This is done on the final dictionary
that has the feature array, labels, names, and loopIDs.
"""

import sys, getopt
import pandas as pd, numpy as np
import os, time, errno, glob
import datetime
from collections import Counter
import subprocess

import re
import gzip
import shutil
import pybedtools as pbed
import pickle 
import seaborn as sb
import scipy as scp
import plotnine as pltn
import sklearn as sk
import sklearn.model_selection
import math, random
import matplotlib.pyplot as plt



#load features

featureDictPath = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/SavedFeatureDicts/finalFeatureDictionary_2019-04-08.pkl"

with open(featureDictPath, "rb") as f:
    featureDict = pickle.load(f)
    
#also have a list of the chromatin mark names
with open("/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/FilesForHPCProcessing/BEDIntersectDataFrames/chromatinMarks.txt", "r") as f:
    chromatinMarks = [line.rstrip() for line in f.readlines()]

chromatinMarks 


xListNonZero, yListNonZero = featureDict["featureArray"].nonzero()

print("Proportion of non-zero values: " + (str(len(xListNonZero)/(15300*40000))))

featureDict["featureArray"][xListNonZero[0],yListNonZero[0]]

#is de proportie verschillend tussen GM en heart?

heartLoopFeaturesOnly = featureDict["featureArray"][featureDict["classLabelArray"] == "Heart",:]
xListNonZeroHeart, yListNonZeroHeart = heartLoopFeaturesOnly.nonzero()

print("Proportion of non-zero values heart: " + (str(len(xListNonZeroHeart)/(15300*40000))))

GMLoopFeaturesOnly = featureDict["featureArray"][featureDict["classLabelArray"] == "GM12878",:]
GMLoopFeaturesOnly.size
xListNonZeroGM, yListNonZeroGM = GMLoopFeaturesOnly.nonzero()

print("Proportion of non-zero values GM: " + (str(len(xListNonZeroGM)/(15300*40000))))


#test for normality in the whole matrix and submatrices

#null hypothesis = normal distr. p <= 0.05 --> not normal
normalcy = scp.stats.mstats.normaltest(featureDict["featureArray"], axis = 0)
normalcy

normalFeatures = normalcy.pvalue[normalcy.pvalue > 0.05].shape[0]
normalFeatures

normalcy.pvalue[normalcy.pvalue != 3.28648865*(10^(-15))]


#not normally distributed whatsoever

banaan = np.sum(GMLoopFeaturesOnly)
np.sum(GMLoopFeaturesOnly)
np.sum(np.isnan(GMLoopFeaturesOnly))

#The nans are no longer there, so this code is deprecated.
#whereAreThemNans = np.isnan(featureDict["featureArray"])
#xCoordNans, yCoordNans = np.where(whereAreThemNans == True)
#featureDict["featureArray"][xCoordNans[0], yCoordNans[0]]
#featureDict["loopID"][xCoordNans[0]]
#featureDict["namesFeatureArrayColumns"][yCoordNans[0]]

#apparently there is a column named leftAnchorleft_25. That is strange. Investigate
#featureDict["namesFeatureArrayColumns"][featureDict["namesFeatureArrayColumns"] == "leftAnchorLeft_25"]
#just one here. Is this feature there for all loops?

#featureDict["namesFeatureArrayColumns"][yCoordNans[1:2500]]
#np.where(featureDict["namesFeatureArrayColumns"][yCoordNans[1:2500]] != "leftAnchorLeft_25")
#np.where(featureDict["namesFeatureArrayColumns"][yCoordNans[1:2500]] == "leftAnchorLeft_25")
#np.sum(whereAreThemNans)




featuresNotNormalised = featureDict["featureArray"]
#see the unique values per feature
uniquesPerFeature = np.unique(featuresNotNormalised[:, 0:500], axis = 0)
uniquesPerFeature
print(featureDict["namesFeatureArrayColumns"][0], " : ", uniquesPerFeature[0])

for num in range(0, 9):
    print(featureDict["namesFeatureArrayColumns"][num], " : ", uniquesPerFeature[num])


freekBoy = featuresNotNormalised[:, 0]
freekBoy
print(np.unique(freekBoy))


#well, that is quite inconclusive. What I have learned is that the p-values are not
#very indicative: they are mostly a vague blend of -1 (nothing assigned in Broadpeak format)
#and 0: no intersects given by BedIntersect, so default value assigned by my scripts.
#I will remove the p-values from the features.

columnsNotPValue = [k for k in featureDict["namesFeatureArrayColumns"] if not k.endswith("avgPValueNegativeLog10") and not k.endswith("medPValueNegativeLog10")]
columnsNotPValue
indicesFeatureArrayToTake = [featureDict["namesFeatureArrayColumns"].tolist().index(k) for k in featureDict["namesFeatureArrayColumns"] if \
                             k in columnsNotPValue]
indicesFeatureArrayToTake

featuresNotNormalised = featureDict["featureArray"][:, indicesFeatureArrayToTake]
featuresNotNormalised.shape





#normalise features using quantile normalisation

featuresQuantNorm     = sk.preprocessing.quantile_transform(featuresNotNormalised,
                                    output_distribution = "normal",
                                    random_state = 42,
                                    copy = True)

featuresQuantNorm

#compare features normalised vs not normalised
featuresNotNormalised[:, 0]
featuresQuantNorm[:, 0]

quantilesQuantNormFeature = np.quantile(featuresQuantNorm[:, 0], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
quantilesNotNormFeature   = np.quantile(featuresNotNormalised[:, 0], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

quantilesQuantNormFeature
quantilesNotNormFeature
#let's quickly make a heatmap to see what is going on
sb.set()
sb.heatmap(featuresNotNormalised[0:500, 0:300])
sb.set()
sb.heatmap(featuresQuantNorm[0:500, 0:300])
sb.distplot(featuresNotNormalised[:, 0])
sb.distplot(featuresQuantNorm[:,0])




#perhaps I should set 0 to nans now, because they are influencing the ranks very much...
#let's try. Note that for pointPeaksInArea,0 means something, so those should be conserved.

columnNamesNoPValues = featureDict["namesFeatureArrayColumns"][indicesFeatureArrayToTake]
featuresNotNormalisedZeroesToNans = pd.DataFrame(featuresNotNormalised.copy(),
                                                 columns = columnNamesNoPValues)
columnNamesList = columnNamesNoPValues.tolist()


columnsToChangeToNan = [entry for entry in columnNamesList \
                        if not  "Overlap" in entry   \
                        and not entry.endswith("PeaksInInterval") \
                        and not entry.endswith("intersectsPerFactor")]
len(columnNamesList)
len(columnsToChangeToNan)


featuresNotNormalisedZeroesToNans[columnsToChangeToNan] = featuresNotNormalisedZeroesToNans[columnsToChangeToNan].replace(0.0, np.nan)

pd.set_option('expand_frame_repr', True)
pd.set_option("display.max_columns", 34)
featuresNotNormalisedZeroesToNans.head()

#okay, so certain columns have been set to nan. Let us visualise these.

sb.distplot(featuresNotNormalisedZeroesToNans.loc[: , "inbetween_10000_ARID3A-human_avgQValueNegativeLog10"][~pd.isnull(featuresNotNormalisedZeroesToNans.loc[: , "inbetween_10000_ARID3A-human_avgQValueNegativeLog10"])])
sb.distplot(featuresNotNormalisedZeroesToNans.loc[: , "rightAnchorRight_2500_ZZZ3-human_sumPointPeaksInInterval"][~pd.isnull(featuresNotNormalisedZeroesToNans.loc[: , "rightAnchorRight_2500_ZZZ3-human_sumPointPeaksInInterval"])])


#randomly plot 20 features and see how they look:

toPlot = random.sample(featuresNotNormalisedZeroesToNans.columns.tolist(), 20)

for column in toPlot:
    filteredColumn = featuresNotNormalisedZeroesToNans.loc[:, column][~pd.isnull(featuresNotNormalisedZeroesToNans.loc[:, column])]
    sb.distplot(filteredColumn)
    plt.show()

#heatmap for a 1000 random features
    
toPlot200 = random.sample(featuresNotNormalisedZeroesToNans.columns.tolist(), 200)
toPlot200Samples = random.sample(range(0,40000), 200)

heatmapPlot = sb.heatmap(featuresNotNormalisedZeroesToNans.loc[:, toPlot200].iloc[toPlot200Samples, :])


#quantile normalise the thing

quantNormFeaturesNanCorrected = sk.preprocessing.quantile_transform(featuresNotNormalisedZeroesToNans,
                                    output_distribution = "normal",
                                    random_state = 42,
                                    copy = True)

dataFrameQuantNormNanCorrected = pd.DataFrame(quantNormFeaturesNanCorrected, columns = columnNamesNoPValues)
#it says all-NaN slice encountered. Wonder where that happens?
whereAreThemNaNs = featuresNotNormalisedZeroesToNans.isnull().sum()
k = whereAreThemNaNs == 40000
k[k == True]

#rightAnchorRight_2500_SUPT20H-human_avgQValueNegativeLog10, apparently SUPT20H is never there?

#re plot quant normalised features
for column in toPlot:
    filteredColumn = dataFrameQuantNormNanCorrected.loc[:, column][~pd.isnull(dataFrameQuantNormNanCorrected.loc[:, column])]
    sb.distplot(filteredColumn)
    plt.show()



#let us cluster a subset of the matrix and see what that brings
    
dataFrameQuantNormNanCorrected.set_index(featureDict["classLabelArray"], inplace = True)    
    

#make a clustermap for a subset of features at first

toPlot1000Samples = random.sample(range(0,40000), 1000)
subsetToPlotCluster = dataFrameQuantNormNanCorrected.loc[:, toPlot200].iloc[toPlot1000Samples, :]
#is it okay to remove nans here? I need to, anyway. I will fill them in with the averages as a naive method.
subsetToPlotCluster = subsetToPlotCluster.fillna(subsetToPlotCluster.mean())
#one column is fully NA
subsetToPlotCluster.head()
subsetToPlotCluster = subsetToPlotCluster.drop(subsetToPlotCluster.columns[subsetToPlotCluster.isna().all()], axis = 1)
subsetToPlotCluster.shape

#because I am using correlation, and a small subset, some columns have only 0.
#0 does not correlate with anything, so this is what gives the errors. Columns
#with only 0s should thus also be removed (or with no variance, see: https://github.com/scikit-learn/scikit-learn/issues/10076 )
columnsWithoutAnyVariance = subsetToPlotCluster.columns[subsetToPlotCluster.nunique() == 1]
subsetToPlotCluster       = subsetToPlotCluster.drop(columnsWithoutAnyVariance, axis = 1)
subsetToPlotCluster.head(0)
k = subsetToPlotCluster.sort_index()

clusterTryOut = scp.cluster.hierarchy.linkage(subsetToPlotCluster,
                                                method = "single",
                                                metric = "correlation")


#colour by chromatin mark and loop type
loopColour = dict(zip(k.index.unique(), "rbg"))
loopColourFinal = k.index.map(loopColour)


columnNamesForColour = subsetToPlotCluster.columns.tolist()
listWithChroms       = [re.search("_([A-Za-z0-9]+-human)_", entry)[1] for entry in columnNamesForColour]
palette = sb.husl_palette(len(set(listWithChroms)), s = .45)
chromMarkColourDict = dict(zip(set(listWithChroms), palette))
chromMarkColour = [chromMarkColourDict[entry] for entry in listWithChroms]                                         #map(listWithChroms, chromMarkColour)

oefeningClusterMap = sb.clustermap(k,
                                   method = "single", metric = "correlation",
                                   figsize = (30, 30),
                                   row_colors = loopColourFinal,
                                   col_colors = chromMarkColour,
                                   row_cluster = False,
                                   xticklabels = True)


oefeningClusterMap.savefig("/2TB-disk/Project/Documentation/Meetings/11-04-2019/ClusterMap_RowsNonClustered_12_04_2019.png")
#oefeningClusterMap.savefig("/2TB-disk/Project/Documentation/Meetings/11-04-2019/FirstTryClusterMap.png")










#After meeting 11-04-2019: make Z-scores etc.
from scipy.stats import zscore

featuresNotNormalisedZeroesToNans.set_index(featureDict["classLabelArray"], inplace = True)
featuresNotNormalisedZeroesToNans.head()

medianValues = featuresNotNormalisedZeroesToNans.median()
stdValues    = featuresNotNormalisedZeroesToNans.std(ddof = 0)



zScoreNormFeats = (featuresNotNormalisedZeroesToNans - featuresNotNormalisedZeroesToNans.median()) / featuresNotNormalisedZeroesToNans.std(ddof = 0)

#doesn't work, might be because medians all 0?
featuresNotNormalisedZeroesToNans.loc[: , "inbetween_10000_ASH2L-human_totalOverlap"]

#remove those columns which do not have standard deviations
columnsToKeep = stdValues[stdValues != 0].index.tolist()

featuresNotNormalisedZeroesToNans = featuresNotNormalisedZeroesToNans.loc[:, columnsToKeep]

recalcMedian = featuresNotNormalisedZeroesToNans.median()
recalcStd    = featuresNotNormalisedZeroesToNans.std(ddof = 0)

zScoreNormFeats = (featuresNotNormalisedZeroesToNans - recalcMedian) / recalcStd
diagNans = zScoreNormFeats.apply(lambda x: sum(np.isnan(x)))
diagNans[diagNans == 40000]

#well, whatever, drop those columns
zScoreNormFeats.drop(diagNans[diagNans == 40000].index.values, axis = 1, inplace = True)
zScoreNormFeats.shape

#change everything more than extremes to extremes

extremeMax = zScoreNormFeats.max().median()
extremeMin = zScoreNormFeats.min().median()

#hmm...well let's just bound at plus and minus 5, even though it can go up to 200 deviations up!

zScoreNormFeats[zScoreNormFeats > 5] = 5
zScoreNormFeats[zScoreNormFeats < -5] = -5

#now for every anchor position, for every interval, take the median of every similar
#feature per loop tissue

uniqueAnchors = list(set([re.search("(\w+_\d+)_", col)[1] for col in zScoreNormFeats.columns.values]))
uniqueFeatures = list(set([re.search("_(\w+\d*$)", col)[1] for col in zScoreNormFeats.columns.values]))
uniqueLoops    = ["Heart", "GM12878"]

anchor, feat, loop = uniqueAnchors[0], uniqueFeatures[0], uniqueLoops[0]

dictCombinedFeats = {}
for anchor in uniqueAnchors:
    for feat in uniqueFeatures:
        for loop in uniqueLoops:
            combinedFeatureValue = zScoreNormFeats.filter(
                                    like = anchor).filter(
                                        like = loop, axis = "index").filter(
                                            like = feat).median(axis = 1)
            combinedFeatureName = anchor + "_" + feat
            if loop in dictCombinedFeats.keys():
                dictCombinedFeats[loop] = pd.concat([dictCombinedFeats[loop],
                                 combinedFeatureValue], axis = 1)
                dictCombinedFeats[loop].rename(columns = {0 : combinedFeatureName}, 
                                 inplace = True)
            else :
                dictCombinedFeats[loop] = pd.DataFrame(combinedFeatureValue,
                                 columns = [combinedFeatureName])

dictCombinedFeats["Heart"]
dictCombinedFeats["GM12878"]

#combine in one dataframe, then process with seaborn
totalDFCombinedFeats = pd.concat([dictCombinedFeats["Heart"], dictCombinedFeats["GM12878"]])

f, axes = plt.subplots(9, 9, figsize=(60, 60), sharex=True)
for i, feature in enumerate(totalDFCombinedFeats.columns):
    sb.distplot(totalDFCombinedFeats.filter(
            like = "Heart", axis = "index")[feature] ,
            color="skyblue", ax=axes[i%8, i//8], bins = 80, kde = False)
    sb.distplot(totalDFCombinedFeats.filter(
            like = "GM12878", axis = "index")[feature] ,
            color="red", ax=axes[i%8, i//8], bins = 80, kde = False)

f.savefig("/2TB-disk/Project/Documentation/Meetings/11-04-2019/FeaturesCombinedForAllChromatinMarks_HeartGMSeparate_12_04_2019.png")

v = re.search("(\w+_\d+)_", zScoreNormFeats.columns.values[0])

########################
#######################
#Nog een keer, maar nu met means in Z score en means bij filteren
#######################
########################

columnNamesNoPValues = featureDict["namesFeatureArrayColumns"][indicesFeatureArrayToTake]
featuresNotNormalisedZeroesToNans = pd.DataFrame(featuresNotNormalised.copy(),
                                                 columns = columnNamesNoPValues)
columnNamesList = columnNamesNoPValues.tolist()


columnsToChangeToNan = [entry for entry in columnNamesList \
                        if not  "Overlap" in entry   \
                        and not entry.endswith("PeaksInInterval") \
                        and not entry.endswith("intersectsPerFactor")]
len(columnNamesList)
len(columnsToChangeToNan)


featuresNotNormalisedZeroesToNans[columnsToChangeToNan] = featuresNotNormalisedZeroesToNans[columnsToChangeToNan].replace(0.0, np.nan)
featuresNotNormalisedZeroesToNans.set_index(featureDict["classLabelArray"], inplace = True)

stdValues    = featuresNotNormalisedZeroesToNans.std(ddof = 0)
columnsToKeep = stdValues[stdValues != 0].index.tolist()

featuresNotNormalisedZeroesToNansMean = featuresNotNormalisedZeroesToNans.loc[:, columnsToKeep]
featuresNotNormalisedZeroesToNansMean

recalcMean = featuresNotNormalisedZeroesToNansMean.mean()
recalcStd    = featuresNotNormalisedZeroesToNansMean.std(ddof = 0)

zScoreNormFeatsMean = (featuresNotNormalisedZeroesToNansMean - recalcMean) / recalcStd
diagNansMean = zScoreNormFeatsMean.apply(lambda x: sum(np.isnan(x)))
diagNansMean[diagNansMean == 40000]

#well, whatever, drop those columns
zScoreNormFeatsMean.drop(diagNansMean[diagNansMean == 40000].index.values, axis = 1, inplace = True)
zScoreNormFeatsMean.shape
zScoreNormFeatsMean.head()
#change everything more than extremes to extremes

extremeMax = zScoreNormFeatsMean.max().median()
extremeMin = zScoreNormFeatsMean.min().median()

#hmm...well let's just bound at plus and minus 5, even though it can go up to 200 deviations up!

zScoreNormFeatsMean[zScoreNormFeatsMean > 5] = 5
zScoreNormFeatsMean[zScoreNormFeatsMean < -5] = -5

#now for every anchor position, for every interval, take the median of every similar
#feature per loop tissue

uniqueAnchorsMean = list(set([re.search("(\w+_\d+)_", col)[1] for col in zScoreNormFeatsMean.columns.values]))
uniqueFeaturesMean = list(set([re.search("_(\w+\d*$)", col)[1] for col in zScoreNormFeatsMean.columns.values]))
uniqueLoopsMean    = ["Heart", "GM12878"]

anchor, feat, loop = uniqueAnchorsMean[0], uniqueFeaturesMean[0], uniqueLoopsMean[0]

dictCombinedFeatsMean = {}
for anchor in uniqueAnchorsMean:
    for feat in uniqueFeaturesMean:
        for loop in uniqueLoopsMean:
            combinedFeatureValue = zScoreNormFeatsMean.filter(
                                    like = anchor).filter(
                                        like = loop, axis = "index").filter(
                                            like = feat).mean(axis = 1)
            combinedFeatureName = anchor + "_" + feat
            if loop in dictCombinedFeatsMean.keys():
                dictCombinedFeatsMean[loop] = pd.concat([dictCombinedFeatsMean[loop],
                                 combinedFeatureValue], axis = 1)
                dictCombinedFeatsMean[loop].rename(columns = {0 : combinedFeatureName}, 
                                 inplace = True)
            else :
                dictCombinedFeatsMean[loop] = pd.DataFrame(combinedFeatureValue,
                                 columns = [combinedFeatureName])

dictCombinedFeatsMean["Heart"]
dictCombinedFeatsMean["GM12878"]

#combine in one dataframe, then process with seaborn
totalDFCombinedFeatsMean = pd.concat([dictCombinedFeatsMean["Heart"], dictCombinedFeatsMean["GM12878"]])

f2, axes2 = plt.subplots(8, 10, figsize=(60, 60), sharex=False)
for i, feature in enumerate(totalDFCombinedFeatsMean.columns):
    print(feature)
    #if needs to be zoomed in, do that
    if ("intersectsPerFactor" in feature or "Overlap" in feature or "sumPointPeaksInInterval" in feature):
        axes2[i%8, i//8].set_xlim(-2, 3)
        axes2[i%8, i//8].set_ylim(0, 2000)
    sb.distplot(totalDFCombinedFeatsMean.filter(
            like = "Heart", axis = "index")[feature] ,
            color="skyblue", ax=axes2[i%8, i//8], bins = 80, kde = False,
            hist_kws=dict(alpha=0.2), rug = False)
    sb.distplot(totalDFCombinedFeatsMean.filter(
            like = "GM12878", axis = "index")[feature] ,
            color="red", ax=axes2[i%8, i//8], bins = 80, kde = False,
            hist_kws=dict(alpha=0.2), rug = False)

f2.savefig("/2TB-disk/Project/Documentation/Meetings/11-04-2019/FeaturesCombinedForAllChromatinMarks_HeartGMSeparate_MeanInZscoreAndMeanToCombineFeats_15_04_2019_takeVier.png")


from sklearn.ensemble import RandomForestClassifier
flip = RandomForestClassifier(random_state = 42)
#take just a subset of the actual data for testing handling of nans
testData = featuresNotNormalised
#sum(testData.index.values == "Heart") #approx equal amount of samples
numericLabels = sk.preprocessing.LabelBinarizer()
#input to one hot encoder should be reshaped
reshapedLabels = featureDict["classLabelArray"].reshape(-1, 1)
numericLabels.fit(reshapedLabels)
transformedClassLabels = numericLabels.transform(reshapedLabels)
print(transformedClassLabels)
print(testData.shape)
transformedClassLabels = np.ravel(transformedClassLabels)

X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(testData,
                                                                        transformedClassLabels,
                                                                        test_size=0.33,
                                                                        random_state=42)


#
# Update 21-05-2019
#to do:
#1. split into test set that I do not touch and training set
#2. split training set into k folds (first cross-validation, for determining model performance)
#3. split each of these again into k folds (second cross-validation, for determining hyperparameters)
#


#shameless copy utility function from sklearn website
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_roc_auc'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_roc_auc'][candidate],
                  results['std_test_roc_auc'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


X_trainingData, X_finalTest, Y_trainingData, Y_finalTest = sk.model_selection.train_test_split(testData,
                                                                        transformedClassLabels,
                                                                        test_size=0.1,
                                                                        random_state=42)


crossValOne = sk.model_selection.StratifiedKFold(n_splits = 10, random_state = 42)
crossValTwo = sk.model_selection.StratifiedKFold(n_splits = 5, random_state  = 85)

#to optimise:
    #n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
    #for max_features, just pick between "sqrt", "log2" and None
    #for the others, just sample uniformly for now


param_dist_old = {
              "n_estimators"     : [10, 50, 100, 150, 200, 300],
              "max_depth"        : [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
              "max_features"     : ["sqrt", "log2", None],
              "min_samples_split": [2, 5, 10, 25, 50, 75],
              "min_samples_leaf" : [1, 5, 10],
              "bootstrap"        : [True, False],
              "criterion"        : ["gini", "entropy"]}


param_dist = {
              "n_estimators"     : [ 50, 125, 200],
              "max_depth"        : [1, 20, 50, 100, None],
              "max_features"     : ["sqrt", "log2", None],
              "min_samples_split": [2, 5, 10, 25],
              "min_samples_leaf" : [1, 5, 10],
              "bootstrap"        : [True, False],
              "criterion"        : ["gini", "entropy"]}
    
randomSearch = sk.model_selection.RandomizedSearchCV(flip, param_dist,
                                          n_iter = 2,
                                          scoring = ["precision", "recall", "roc_auc"],
                                          cv = 2,
                                          iid = False,
                                          refit = "roc_auc")


allROCAUCs          = []
allROCFPRs          = []
allROCTPRs          = []
allPredictions      = []
allPredictionsProbs = []
allTrueClasses      = []
allConfMatrices     = []
allNormConfMatrices = []
bestParams          = []

for trainOne, testOne in crossValOne.split(X_trainingData, Y_trainingData):
    
    allTrueClasses.append(Y_trainingData[testOne])
    #cross-validate hyperparameters on a subset of the current data, then use the best
    #hyperparams for a classifier).
    randomSearch.fit(X_trainingData[trainOne],
                     Y_trainingData[trainOne])
    report(randomSearch.cv_results_)
    break
    #with the estimator with the best parameters, test on the test set
    #need to keep all these scores and predictions for the final evaluation
    bestRF     = randomSearch.best_estimator_
    bestParams.append(randomSearch.best_params_)
    
    prediction                   = bestRF.predict(X_trainingData[testOne])
    allPredictions.append(prediction)
    predictionProbs              = bestRF.predict_proba(X_trainingData[testOne])
    allPredictionsProbs.append(predictionProbs)
    predictionProbsPositiveClass = predictionProbs[:,1]
    
    roc_auc = sk.metrics.roc_auc_score(Y_trainingData[testOne],
                                       predictionProbsPositiveClass)
    allROCAUCs.append(roc_auc)
    
    #plot it (optionally)
    fpr, tpr, thresholds = sk.metrics.roc_curve(Y_trainingData[testOne],
                                                predictionProbsPositiveClass,
                                                pos_label=1)
    
    allROCFPRs.append(fpr)          
    allROCTPRs.append(tpr)          
#    fig, ax = plt.subplots()
#    lw = 2
#    plt.plot(fpr, tpr, color='darkorange',
#                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
#    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('ROC curve_', fontsize = 8)
#    plt.legend(loc="lower right")
#    plt.show()
    
#    precision, recall, _ = sk.metrics.precision_recall_curve(Y_trainingData[testOne],
#                                                  predictionProbsPositiveClass)
#    #plot it, optionally
#    from inspect import signature
#    
#    step_kwargs = ({'step': 'post'}
#               if 'step' in signature(plt.fill_between).parameters
#               else {})
#    plt.step(recall, precision, color='b', alpha=0.2,
#         where='post')
#    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
#    
#    plt.xlabel('Recall')
#    plt.ylabel('Precision')
#    plt.ylim([0.0, 1.05])
#    plt.xlim([0.0, 1.0])
#    plt.show()
#    #plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
#    #          average_precision))
    
    confMatrix = sk.metrics.confusion_matrix(Y_trainingData[testOne],
                                             prediction)
    confMatrixNormed = confMatrix.astype('float') / confMatrix.sum(axis=1)[:, np.newaxis]
    
    allConfMatrices.append(confMatrix)
    allNormConfMatrices.append(confMatrixNormed)
    
#    confusionMatrixPlot = plot_confusion_matrix(Y_trainingData[testOne],
#                                                    prediction,
#                                                    classes = np.array(["GM12878", "Heart"]),
#                                                    normalize = True,
#                                                    title = "ConfMatrix_")


#plot combined ROC plot:
    
    fig, ax = plt.subplots()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve_', fontsize = 8)
    plt.legend(loc="lower right")
    plt.show()

#Heart = 1, GM = 0

flip.fit(X_train, Y_train)

predictedFlip = flip.predict(X_test)
predictedProbaFlip = flip.predict_proba(X_test)
sk.metrics.confusion_matrix(Y_test, predictedFlip)


#shamelessly copy from sklearn website:
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(set(transformedClassLabels)))
    plt.xticks(tick_marks, ["GM12878", "Heart"], rotation=45)
    plt.yticks(tick_marks, ["GM12878", "Heart"])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
    
        # Compute confusion matrix
        cm = sk.metrics. confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[sk.utils.multiclass.unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        fig, ax = plt.subplots()
        plt.grid(b=None)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
    
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return (fig, ax, cm)


# Compute confusion matrix
cm = sk.metrics.confusion_matrix(Y_test, predictedFlip)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
confusionMatrixTest = plot_confusion_matrix(Y_test, predictedFlip, classes = np.array(["GM12878", "Heart"]), normalize = True)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
#cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#print('Normalized confusion matrix')
#print(cm_normalized)
#plt.figure()
#plt.grid(b=None)
#plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
#
#plt.show()


sk.metrics.roc_auc_score(Y_test, predictedProbaFlip[:,1])

#draw an ROC curve?

fpr, tpr, thresholds = sk.metrics.roc_curve(Y_test, predictedProbaFlip[:,1], pos_label=1)
fig, ax = plt.subplots()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % sk.metrics.roc_auc_score(Y_test, predictedProbaFlip[:,1]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


flip.



flip.
#flip.feature_i
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
#quick plot of feature importances:

col = featureDict["namesFeatureArrayColumns"]

#modelname.feature_importance_
sortedImportance = np.argsort(flip.feature_importances_)
sortedCol        = np.flip(col[sortedImportance])
sortedFeatImp    = np.flip(flip.feature_importances_[sortedImportance])
#y = np.flip(np.sort(flip.feature_importances_[0:5]))

usedCols = sortedCol[0:20]
usedVals = sortedFeatImp[0:20]

len(flip.feature_importances_)

a4_dims = (5, 10)
fig, ax = plt.subplots(figsize=a4_dims)
coolPlot = sb.barplot(x=usedVals, y=usedCols, ax = ax)
coolPlot.figure.savefig("/2TB-disk/Project/Documentation/Meetings/03-05-2019/initialFeatImportanceLocal_3.png")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
#plot
fig, ax = plt.subplots() 
width = 0.4 # the width of the bars 
ind = np.arange(len(sortedFeatImp)) # the x locations for the groups
ax.barh(ind, sortedFeatImp, width, color="green")
ax.set_yticks(ind+width/10)
ax.set_yticklabels(sortedCol, minor=False)

plt.title("Feature importance in RandomForest Classifier")
plt.xlabel("Relative importance")
plt.ylabel("feature") 
plt.figure(figsize=(100,100))
fig.set_size_inches(150, 150, forward=True)

plt.savefig("/2TB-disk/Project/Documentation/Meetings/03-05-2019/initialFeatImportanceLocal.png")

flip.predict(X_test)




#Okay, that was one part. What I want




#note that quantile normalisation makes things ranked 

##axis = 0 should be column-wise, i.e. for each column. axis=0 is standard, so this should be correct.
#
#featuresZScoreQuantNorm = sk.preprocessing.scale(featuresQuantNorm)
#featuresZScoreQuantNorm
#featuresZScoreQuantNorm.shape
#
#
#testFeature = featuresZScoreQuantNorm[:,0]
#np.quantile(testFeature, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#np.std(testFeature)
#sb.distplot(testFeature)
#
#testSample = featuresZScoreQuantNorm[0,:]
#np.quantile(testSample, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#np.std(testSample)
#sb.distplot(testSample)
#
##okay this seems to have normalised over rows, not columns, which is where I have my features. 
##Do again but for columns
#
#featuresQuantNormColumn     = sk.preprocessing.quantile_transform(featuresNotNormalised,
#                                    output_distribution = "normal",
#                                    random_state = 42,
#                                    axis = 1)
#
#featuresQuantNormColumn
#
##then z-score to get it between 0-1, though note that this 'distribution' is now 1000 discrete possible
##values per feature (because quantile normalisation does that)
#
#featuresZScoreQuantNormColumn = sk.preprocessing.scale(featuresQuantNormColumn, axis = 1)
#featuresZScoreQuantNormColumn
#featuresZScoreQuantNormColumn.shape
#
#
#testFeature2 = featuresZScoreQuantNormColumn[:,0]
#np.quantile(testFeature2, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#np.std(testFeature2)
#sb.set()
#sb.distplot(testFeature2)
#
#testSample2 = featuresZScoreQuantNormColumn[0,:]
#np.quantile(testSample2, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#np.std(testSample2)
#sb.set()
#sb.distplot(testSample2)



#transpose the feature array




#take a subset for clustering locally 


#cluster (single linkage, but also try other methods)
scp.cluster.hierarchy.linkage()

#collapse tree on some level and take median of leaf features to make metafeature (I guess?)



#make heatmap of these new features








sb.set()
colourPaletteChromMarks = sb.husl_palette(len(chromatinMarks), s=.25)
colourPaletteChromMarks
#
colourDictChromMarks = dict(zip(chromatinMarks, colourPaletteChromMarks))
colourDictChromMarks
#
sb.clustermap(featureDict["featureArray"])
sb.heatmap(featureDict["featureArray"])




#
#sb.h



#cluster the feature columns 



