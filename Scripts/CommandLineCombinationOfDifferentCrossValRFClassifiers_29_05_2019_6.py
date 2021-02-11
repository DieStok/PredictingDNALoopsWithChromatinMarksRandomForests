#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:31:29 2019

@author: dstoker

Script that combines the results (scores) from different cross-validations and 
plots them.

#update 26-07-2019
This code is a mess, as the figures were made with haste for the presentation, and
things have been shoehorned in. The script is ment to be run via a command-line on
the HPC, where it combines multiple classifier output files for a specific run
into figures summarising that run.

I make summarised versions of the data, of hyperparameters used (though in a very inexpert fashion),
and later also full data when I found out that was better for plotting. Hence the mess.
Confusion matrices could not be output: they would not save no matter what I tried. 
They can be manually opened by locally using LoadFiguresISaved.py




######Also identifies the best parameters for a 'production grade' model. That is to say:
######    takes for numeric parameters of the best estimators the median or mode
######   for text-based ("gini" or "entropy") and boolean: the most common one
######    (not yet implemented)
    
For now (29_05_2019) I think it is best if the tool is command line based and makes a list
of all .pkl files in an input file path, and does the evaluation for all possible
separate anchors and for the combined version. (that is to say, I allow the RF scripts
to be run either for anchors of a specific size, or to run on feature data for anchors
of multiple sizes at the same time (to see what the estimator learns as most informative features)
So, this script will sort .pkl files into what anchor or combination of anchor feature data
they are for, and produce plots and statistics for every one of them for comparison.)

Second argument required is the .pkl file that was used to make the classification, 
as I can get the feature names from there.



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
import scipy as scp
import sklearn as sk
import sklearn.model_selection
import sklearn.preprocessing
from sklearn.ensemble import RandomForestClassifier
import math, random
import matplotlib.pyplot as plt
import json
import matplotlib.pyplot as plt
import seaborn as sb

pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 100)


#matplotlib magic for axes not being cut-off?
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

####
####
#### HARDCODED OUTPUT DIRECTORY FOLLOWS BELOW.
####
####

outputDir = "/hpc/cog_bioinf/ridder/users/dstoker/figuresForAllClassifiers_17_06_2019/"

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def make_DataFrame_row_dict_from_pickle(loadedPickle, anchorSize, featNameList):
    currentFileRep      = loadedPickle["rep"]
    currentFileCrossVal = loadedPickle["crossValFold"]
    
    
    trueNeg, falsePos, falseNeg, truePos = loadedPickle["data"]["confMatrix"].ravel()
    trueNegNorm, falsePosNorm, falseNegNorm, truePosNorm = loadedPickle["data"]["confMatrixNormed"].ravel()
        
    infoDict = {"anchorSize"               : anchorSize,
                "rep"                      : currentFileRep,
                "crossVal"                 : currentFileCrossVal,
                "roc_auc"                  : loadedPickle["data"]["roc_auc"],
                "precision"                : loadedPickle["data"]["precision"],
                "recall"                   : loadedPickle["data"]["recall"],
                "falsePositiveRate"        : loadedPickle["data"]["falsePositiveRate"],
                "truePositiveRate"         : loadedPickle["data"]["truePositiveRate"],
                "trueNegConf"              : trueNeg,
                "falsePosConf"             : falsePos,
                "falseNegConf"             : falseNeg,
                "truePosConf"              : truePos,
                "trueNegConfNorm"          : trueNegNorm,
                "falsePosConfNorm"         : falsePosNorm,
                "falseNegConfNorm"         : falseNegNorm,
                "truePosConfNorm"          : truePosNorm,
                "paramNEstimators"         : loadedPickle["data"]["bestParams"]["n_estimators"],
                "paramMaxDepth"            : loadedPickle["data"]["bestParams"]["max_depth"],
                "paramMaxFeatures"         : loadedPickle["data"]["bestParams"]["max_features"],
                "paramMinSamplesSplit"     : loadedPickle["data"]["bestParams"]["min_samples_split"],
                "paramMinSamplesLeaf"      : loadedPickle["data"]["bestParams"]["min_samples_leaf"],
                #"paramBootstrap"           : loadedPickle["data"]["bestParams"]["bootstrap"],
                #"paramInfoCriterion"       : loadedPickle["data"]["bestParams"]["criterion"],
                "fullFeatImportanceArray"  : loadedPickle["data"]["bestFeatImport"],
                "featImportanceArrayNames" : featNameList,
                "indexListTestSetIndices"  : loadedPickle["data"]["testedOn"],
                "indexListTrainSetIndices" : loadedPickle["data"]["trainedOn"]
                }
    
    dataFrameRow = pd.DataFrame(pd.Series(infoDict).reset_index(drop = True)).transpose()
    #print(dataFrameRow)
    dataFrameRow.set_axis(infoDict.keys(),1,inplace=True)
    
    return dataFrameRow


##for testing:
#argv = ["/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/ClassifierCrossValFilesToCombine_11_06_2019/Testing_11_06_2019"]
#argv[0] = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/ClassifierCrossValFilesToCombine_11_06_2019/Testing_11_06_2019"
#
##needs to be the .pkl file that has the features used to train these classifiers. Allows for calculation of
##median feature importances.
#argv.append("/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/SavedFeatureDicts/finalFeatureDictionaryUniformRandomVersusGM_40000SingleAnchorsEach_28_05_2019.pkl")

def main(argv):
    if os.path.isdir(argv[0]):
        if argv[0].endswith("/"):
            pass
        else:
            argv[0] += "/"
        filesInPath = glob.glob(argv[0] + "*")
        pklFilesInPath = [file for file in filesInPath if file.endswith(".pkl")]
    else:
        sys.exit("Path is not a directory. Check input!")
    
    
    with open (argv[1], "rb") as file:
        pklForThisRun = pickle.load(file)
    featNamesForThisRun = pklForThisRun['namesFeatureArrayColumns']
    columnsNotPValue = [k for k in featNamesForThisRun if not k.endswith("avgPValueNegativeLog10") and not k.endswith("medPValueNegativeLog10")]
    indicesFeatureArrayToTake = [featNamesForThisRun.tolist().index(k) for k in featNamesForThisRun if \
                                 k in columnsNotPValue]
    featNamesForThisRun = featNamesForThisRun[indicesFeatureArrayToTake]
    
    #29_05_2019: this is hardcoded to specific naming convention used in the RF script for saving
    #if that changes, this needs to be changed as well.
    uniqueAnchorSizesToCalculateFor = set([re.match(".*AnchorSize_([0-9]+)_Data.*", fileName)[1] for fileName in pklFilesInPath if not re.match(".*AnchorSize_([0-9]+)_Data.*", fileName) is None])
    #testing
    #pathToTry = '/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/ClassifierCrossValFilesToCombine_11_06_2019/Testing_11_06_2019/RandomForestClassification_AnchorSize_2500_Data_rep_4_of5_fold_7_of10_FromInputFile_finalFeatureDictionaryUniformRandomVersusGM_40000SingleAnchorsEach_28_05_2019_2019-06-06.pkl'
    #k = re.match(".*AnchorSize_([0-9]+)_Data.*", pathToTry)
    
    repsAndCrossVals = re.match(".*Data_rep_\d+_of(\d+)_fold_\d+_of(\d+)_.*", pklFilesInPath[0])
    totalReps      = repsAndCrossVals[1]
    totalCrossVals = repsAndCrossVals[2]
    
    
    allDataFramesInformationAboutClassifiers = []
    #if there are no classifiers that used specific anchor Sizes, only do the combined:
    if (uniqueAnchorSizesToCalculateFor != set([None])):
        
        for anchorSizeSpecificRF in uniqueAnchorSizesToCalculateFor:
            #get files with that anchor in their name
            matchesSizedFiles = [re.match(".*AnchorSize_([0-9]+)_Data.*", fileName) for fileName in pklFilesInPath]
            filesToTake = [True if entry is not None and entry[1] == anchorSizeSpecificRF else False for entry in matchesSizedFiles]
            filesForThisSingleAnchorSize = [pklFilesInPath[i] for i in np.where(filesToTake)[0]]
            #read in each file, get the rep and crossVal they belong to
            
            #here, get the feature names from the .pkl that belong to the feature importances
            print(featNamesForThisRun)
            matchLocations = np.where([re.match("^\w+_(\d+)_.*", entry)[1] == anchorSizeSpecificRF for entry in featNamesForThisRun])[0]
            featNamesThisAnchor = featNamesForThisRun[matchLocations]
            
            specificAnchorSizeClassifierInformationDataFrame = None
            for file in filesForThisSingleAnchorSize:
                pickleFile          = load_obj(file)
                rowToAddToDF = make_DataFrame_row_dict_from_pickle(pickleFile, anchorSizeSpecificRF, featNamesThisAnchor)
                
                if specificAnchorSizeClassifierInformationDataFrame is None:
                    #generate the DataFrame with all the info about the classifier performance in folds and repeats
                    
                    specificAnchorSizeClassifierInformationDataFrame = rowToAddToDF
                else:
                    specificAnchorSizeClassifierInformationDataFrame = pd.concat([specificAnchorSizeClassifierInformationDataFrame,
                                                                                  rowToAddToDF])
                
            allDataFramesInformationAboutClassifiers.append(specificAnchorSizeClassifierInformationDataFrame)
    
    #if there is a file with all anchor Sizes together:
    allAnchorsTogetherFilesBoolean = np.where([re.match(".*_AllAnchorsTogether_.*", fileName) for fileName in pklFilesInPath])[0]
    if (len(allAnchorsTogetherFilesBoolean) > 0):
        filesToTakeAllAnchorsTogether = [pklFilesInPath[i] for i in allAnchorsTogetherFilesBoolean]
        
        dataFrameAllSizesTogether = None
        for file in filesToTakeAllAnchorsTogether:
            pickleFile          = load_obj(file)
            rowToAddToDF = make_DataFrame_row_dict_from_pickle(pickleFile,
                                                               "allSizesTogether",
                                                               featNamesForThisRun)
            
            if (dataFrameAllSizesTogether is None):
                dataFrameAllSizesTogether = rowToAddToDF
            else:
                dataFrameAllSizesTogether = pd.concat([dataFrameAllSizesTogether,
                                                       rowToAddToDF])
        allDataFramesInformationAboutClassifiers.append(dataFrameAllSizesTogether)
        
    
    #Okay, I now have a list of dataframes organised per anchor. I think it is easier to make
    #one large dataframe and use groupby for plotting commands
    
    totalDataFrameAllClassifierRepeatsAndFolds = pd.concat(allDataFramesInformationAboutClassifiers)
    
    #change all None values to nan
    totalDataFrameAllClassifierRepeatsAndFolds.fillna(pd.np.nan, inplace = True)
    
    totalDataFrameAllClassifierRepeatsAndFolds = totalDataFrameAllClassifierRepeatsAndFolds.astype({"anchorSize" : str,
                                                       "rep" : int,
                                                       "crossVal" : int,
                                                       "roc_auc" : float,
                                                       "precision" : float,
                                                       "recall" : float,
                                                       "trueNegConf" : int,
                                                       "falsePosConf" : int,
                                                       "falseNegConf" : int,
                                                       "truePosConf" : int,
                                                       "trueNegConfNorm" : float,
                                                       "falsePosConfNorm" : float,
                                                       "falseNegConfNorm" : float,
                                                       "truePosConfNorm" : float,
                                                       "paramNEstimators" : int,
                                                       "paramMaxDepth" : float,
                                                       "paramMaxFeatures" : str,
                                                       "paramMinSamplesSplit" : int,
                                                       "paramMinSamplesLeaf" : float,
                                                       })
    #first: classify by anchor size and repeat,
    #take the median and std of all numeric performance measures
    #get a dataframe that holds info about hyperparams as well
    #later plot these
    summaryStatisticsDF = None
    dataFrameHyperParamValues = None
    ROCDataFrame = None
    print(totalDataFrameAllClassifierRepeatsAndFolds.head())
    print(totalDataFrameAllClassifierRepeatsAndFolds.tail())
    groupedByAnchorSize = totalDataFrameAllClassifierRepeatsAndFolds.groupby(["anchorSize"])
    
    ROCcurvePerAnchorDict = {}
    featureImportancesPerAnchorDict = {}
    for name, group in groupedByAnchorSize:
        
        groupedPerRep = group.groupby("rep")
        
        
        #Get the feature importances, sorted per rep, median and std over crossVals, with their names
        totalDataFrameFeatImportancesThisAnchor = None
        nonSummarisedValuesThisAnchor = None
        dataFrameForROCCurveThisAnchor = None
        
        for repNumber, groupOfCrossValDataPerRep in groupedPerRep:
            dataFrameFeatImportancesThisRep = None
            dfLongFormThisRep = None
            dataFrameROCCurveThisRep = None
            #print(name2)
            #print(group2["fullFeatImportanceArray"])
            for index, separateCrossValDataRow in groupOfCrossValDataPerRep.iterrows():
                #print(row)
                #print(separateCrossValDataRow)
                
                
                
                
                featureImportanceArray = separateCrossValDataRow.loc["fullFeatImportanceArray"]
                dfToAdd = pd.DataFrame(featureImportanceArray, columns = ["rep" + str(repNumber)])
                dataFrameFeatImportancesThisRep = pd.concat([dataFrameFeatImportancesThisRep,
                                                             dfToAdd], axis = 1)
                
#                print(featureImportanceArray); print(len(featureImportanceArray))
#                print([separateCrossValDataRow.loc["crossVal"]] * len(featureImportanceArray)); print(len([separateCrossValDataRow.loc["crossVal"]] * len(featureImportanceArray)))
#                print([separateCrossValDataRow.loc["rep"]] * len(featureImportanceArray)); print(len([separateCrossValDataRow.loc["rep"]] * len(featureImportanceArray)))
#                print(separateCrossValDataRow["featImportanceArrayNames"]); print(len(separateCrossValDataRow["featImportanceArrayNames"]))
#                print([separateCrossValDataRow.loc["anchorSize"]] * len(featureImportanceArray)); print(len([separateCrossValDataRow.loc["anchorSize"]] * len(featureImportanceArray)))
#                break
                #another long form dataFrame for feature importance later
                dfToAddLongForm = pd.DataFrame({"featureImportanceArray" : featureImportanceArray,
                                                "crossVal"  : [separateCrossValDataRow.loc["crossVal"]] * len(featureImportanceArray),
                                                "rep"       : [separateCrossValDataRow.loc["rep"]] * len(featureImportanceArray),
                                                "featNames" : separateCrossValDataRow["featImportanceArrayNames"],
                                                "anchor"    : separateCrossValDataRow.loc["anchorSize"] # * len(featureImportanceArray)
                                                })
                dfLongFormThisRep = pd.concat([dfLongFormThisRep, dfToAddLongForm], axis = 0)
                
                dfToAddROCCurve = pd.DataFrame({"truePositiveRate" : separateCrossValDataRow.loc["truePositiveRate"],
                                                "falsePositiveRate" : separateCrossValDataRow.loc["falsePositiveRate"],
                                                "crossVal" : [separateCrossValDataRow.loc["crossVal"]] * len(separateCrossValDataRow.loc["truePositiveRate"]),
                                                "rep" : [separateCrossValDataRow.loc["rep"]] * len(separateCrossValDataRow.loc["truePositiveRate"]),
                                                "anchor" : separateCrossValDataRow.loc["anchorSize"] #* len(featureImportanceArray)
                                                })
                dataFrameROCCurveThisRep = pd.concat([dataFrameROCCurveThisRep,
                                                      dfToAddROCCurve],
                                                        axis = 0)
                
                
#            print(dataFrameFeatImportancesThisRep)
#            print(dataFrameFeatImportancesThisRep.median())
#            print(dataFrameFeatImportancesThisRep.std())
#            print(row)
#            
#            print(pd.DataFrame(row["featImportanceArrayNames"]).shape)
#            print(dataFrameFeatImportancesThisRep.median(axis = 1).shape)
#            print(dataFrameFeatImportancesThisRep.std(axis = 1).shape)
            dataFrameFeatImportancesMedianStdFeatNames = pd.concat([pd.DataFrame(dataFrameFeatImportancesThisRep.median(axis = 1)),
                                                                    pd.DataFrame(dataFrameFeatImportancesThisRep.std(axis = 1)),
                                                                    pd.DataFrame(separateCrossValDataRow["featImportanceArrayNames"])
                                                                    ],
                                                                    axis = 1,
                                                                    sort = False
                                                                    )
            dataFrameFeatImportancesMedianStdFeatNames.columns = ["medianFeatureImportancesOverCrossVals",
                                                                  "stdFeatureImportancesOverCrossVals",
                                                                  "featureNames"]
            dataFrameFeatImportancesMedianStdFeatNames.sort_values(ascending = False,
                                                                   by = "medianFeatureImportancesOverCrossVals",
                                                                   inplace = True)
            dataFrameFeatImportancesMedianStdFeatNames.reset_index(drop = True, inplace = True)
            dataFrameFeatImportancesMedianStdFeatNames.columns += "_rep" + str(repNumber)

            print(dataFrameFeatImportancesMedianStdFeatNames.head(5))
            print(dataFrameFeatImportancesMedianStdFeatNames.tail(5))
            totalDataFrameFeatImportancesThisAnchor = pd.concat([totalDataFrameFeatImportancesThisAnchor,
                                                                dataFrameFeatImportancesMedianStdFeatNames],
                                                                axis = 1,
                                                                sort = False)
            nonSummarisedValuesThisAnchor = pd.concat([nonSummarisedValuesThisAnchor,
                                                       dfLongFormThisRep],
                                                         axis = 0)
            
            dataFrameForROCCurveThisAnchor = pd.concat([dataFrameForROCCurveThisAnchor,
                                                        dataFrameROCCurveThisRep],
                                                        axis = 0)
            
        
        ROCcurvePerAnchorDict[name] = dataFrameForROCCurveThisAnchor
        
        featureImportancesPerAnchorDict[name] = {"summaryStats" : totalDataFrameFeatImportancesThisAnchor,
                                       "completeValues" : nonSummarisedValuesThisAnchor}
        
        
        medianValues = groupedPerRep[["roc_auc", "precision", "recall", "truePosConf",
                                      "falsePosConf", "trueNegConf", "falseNegConf",
                                      "truePosConfNorm", "falsePosConfNorm", "trueNegConfNorm",
                                      "falseNegConfNorm"]].median()
        medianValues.columns += "_median"
        stdValues   = groupedPerRep[["roc_auc", "precision", "recall", "truePosConf",
                                      "falsePosConf", "trueNegConf", "falseNegConf",
                                      "truePosConfNorm", "falsePosConfNorm", "trueNegConfNorm",
                                      "falseNegConfNorm"]].std()
        stdValues.columns += "_std"
        amountOfCrossValsSummed = groupedPerRep["crossVal"].count()
        amountOfCrossValsSummed.name = "crossValsSummedForThisRep"
        #now put this in a dataFrame and then plot conf matrices and bar plots with CIs

        dfPlotting = pd.concat([medianValues, stdValues, amountOfCrossValsSummed], axis = 1)
        dfPlotting["anchorType"] = name
        
        if (summaryStatisticsDF is None):
            summaryStatisticsDF = dfPlotting
        else:
            summaryStatisticsDF = pd.concat([summaryStatisticsDF, dfPlotting])
        
        #now one for checking hyperparameters
        
        def get_None_count(column):
            return sum(pd.isna(column))
        
        def get_max_feats_counts(column):
            return (sum(column == "sqrt"), sum(column == "log2"), sum(pd.isna(column)))
        
        
        k = groupedPerRep["paramMaxFeatures"].apply(get_max_feats_counts)
        
        
        countsForMaxFeatsDF = pd.DataFrame([[row[0] for row in k], [row[1] for row in k], [row[2] for row in k]]).transpose()
        countsForMaxFeatsDF.columns = ["countMaxFeatsSqrt", "countMaxFeatsLog2", "countMaxFeatsNone"]
        countsForMaxFeatsDF["rep"] = range(1,int(totalReps)+1)
        countsForMaxFeatsDF.set_index("rep", inplace = True)
        
        medianValues= groupedPerRep[["paramNEstimators", "paramMaxDepth", "paramMinSamplesSplit",
                                     "paramMinSamplesLeaf"]].median()
        medianValues.columns += "_median"
        medianValues.rename({"paramMaxDepth_median" : "paramMaxDepth_median_noteThatMaxDepthCanBeNone"}, inplace = True)
        stdValues = groupedPerRep[["paramNEstimators", "paramMaxDepth", "paramMinSamplesSplit",
                                     "paramMinSamplesLeaf"]].std()
        stdValues.columns +="_std"
        
        if dataFrameHyperParamValues is None:
            dataFrameHyperParamValues = pd.concat([medianValues, stdValues, countsForMaxFeatsDF], axis = 1)
        else:
            dataFrameHyperParamValues = pd.concat([dataFrameHyperParamValues,
                                                   pd.concat([medianValues, stdValues, countsForMaxFeatsDF], axis = 1)])
    
    
    #done. Plotting:
    
    #what a fucking mess with seaborn and who knows what. Test of patience for sure.
    #in the end just went the matplotlib route.
    listOfBarPlots = []
    listOfBarPlotsToMake = ["roc_auc", "precision", "recall"]
    summaryStatisticsDF["repNumber"] = summaryStatisticsDF.index
    for plot in listOfBarPlotsToMake:
        
        fig, ax = plt.subplots(1,3, figsize = (10,5), sharey = "all")
        fig.suptitle(plot)
        #facetGrid = sb.FacetGrid(summaryStatisticsDF, col = "anchorType")
        #facetGrid.map_dataframe(errplot, "repNumber", "roc_auc_median", "roc_auc_std")
        
        for number, (name, group) in enumerate(summaryStatisticsDF.groupby("anchorType")):
            print(number)
            print(name)
            print(group)
            
            xValues = ["rep " + str(val) for val in range(1, len(group[plot + "_median"]) + 1)]
            #k = sb.catplot( x = "repNumber", y = "roc_auc_median", ci = None, ax = ax[number], kind ="bar",
            #               data = group)
            bar1 = ax[number].bar(xValues, group[plot + "_median"], yerr = group[plot + "_std"], capsize = 6)
            ax[number].set_ylim(0,1)
            ax[number].set_ylabel("Score (median ± std over folds)")
            #ax[number].yaxis.set_tick_params(labelleft = True)
            ax[number].set_title("Anchor size: " + name)
            
            for num, rect in enumerate(bar1) :
                #height = rect.get_height()
                height = 0.9
                ax[number].text(rect.get_x() + rect.get_width()/2.0,
                  height , '%d' % group["crossValsSummedForThisRep"].iloc[num],
                  ha='center', va='bottom')
        listOfBarPlots.append([fig, ax])
    

#Make one barplot panel total:
    dictPlotsClassifierPerformanceScores = {}
    totalDFForOneBarPlot = None
    for scoreToUse in ["roc_auc_median", "precision_median", "recall_median"]:
        toAppend = pd.DataFrame({"valueScore" : summaryStatisticsDF[scoreToUse],
                                 "valueName"  : [scoreToUse] * len(summaryStatisticsDF[scoreToUse]),
                                 "anchorType" : summaryStatisticsDF["anchorType"]})
        totalDFForOneBarPlot = pd.concat([totalDFForOneBarPlot,
                                          toAppend])
        
        
    #here, I define a colour palette for the anchor types present in this data that is
    #shared between the single ROC plot and the bar plot presented here.
    fig, ax = plt.subplots(1,1, figsize = (10,5))
    colourSetPerAnchorType = sb.color_palette(n_colors = len(np.unique(totalDFForOneBarPlot.anchorType)))
    colourDictToUseForHueInROCAndMedianScorePlots = dict(zip(np.unique(totalDFForOneBarPlot.anchorType), colourSetPerAnchorType))
    barPlotScoreMetricsAllTogetherPerAnchorType = sb.stripplot(data = totalDFForOneBarPlot, x = "valueName",
                      y = "valueScore", hue = "anchorType", dodge = True,
                      jitter = True, palette = colourDictToUseForHueInROCAndMedianScorePlots,
                      ax = ax)
    
    
    medianWidth = 0.28
    barPlotScoreMetricsAllTogetherPerAnchorTypeIntervals = len(np.unique(totalDFForOneBarPlot["anchorType"]))
    #so need to plot at 3 separate locations within the 1
    toTake = np.linspace(0, 1, barPlotScoreMetricsAllTogetherPerAnchorTypeIntervals + 2)[1:-1]
    #To make it work with centered coordinates:
    #I do not know what should follow the else clause, so for now I LEFT IT THE SAME.
    toTake = toTake - 0.5 if len(toTake) % 3 == 0 else toTake - 0.5
    for tick, text in zip(barPlotScoreMetricsAllTogetherPerAnchorType.axes.get_xticks(), barPlotScoreMetricsAllTogetherPerAnchorType.axes.get_xticklabels()):
        xLabel = text.get_text()  # "X" or "Y"
        #print("xLabel:");print(xLabel)
        #print("tick:");print(tick)
        subset = totalDFForOneBarPlot[totalDFForOneBarPlot['valueName']==xLabel]
        for num, subcat in enumerate(np.unique(subset["anchorType"])):
            #print(tick + toTake[num])
            #print(subcat)
            
            # calculate the median value for all replicates of either X or Y
            medianVal = subset[subset.anchorType == subcat].median()[0]
            #print(medianVal)
            # plot horizontal lines across the column, centered on the tick
            barPlotScoreMetricsAllTogetherPerAnchorType.axes.plot([tick + toTake[num]-medianWidth/2, tick + toTake[num]+medianWidth/2],
                           [medianVal, medianVal],
                    lw=2, color='k', linestyle = "--")
    #barPlotScoreMetricsAllTogetherPerAnchorType.axes.set_ylim([0.2, 1.05])
    barPlotScoreMetricsAllTogetherPerAnchorType.axes.set_ylabel("Score (median over 10 folds for 5 reps)")
    barPlotScoreMetricsAllTogetherPerAnchorType.axes.set_xlabel("")
    dictPlotsClassifierPerformanceScores["medianScoresCombinedPerAnchor"] = [barPlotScoreMetricsAllTogetherPerAnchorType.figure,
                                        barPlotScoreMetricsAllTogetherPerAnchorType]
    
    #do the same, but separate plot per value
    
    for value in np.unique(totalDFForOneBarPlot["valueName"]):
        subsetOfDataForThisValueOnly = totalDFForOneBarPlot[totalDFForOneBarPlot.valueName == value]
        fig, ax = plt.subplots()
        hueOrder = subsetOfDataForThisValueOnly.groupby([
                "anchorType"]).valueScore.median().sort_values(ascending = False).index
        plotThisScoreValue = sb.stripplot(data = subsetOfDataForThisValueOnly, x = "valueName",
                      y = "valueScore", hue = "anchorType", dodge = True,
                      jitter = 0.03, palette = colourDictToUseForHueInROCAndMedianScorePlots,
                      ax = ax, hue_order = hueOrder)
        
        #copy of the above. Should make it into a function but that would take too long.
#        medianWidth = 0.05
#        plotThisScoreValueIntervals = len(np.unique(subsetOfDataForThisValueOnly["anchorType"]))
#        #so need to plot at 3 separate locations within the 1
#        toTake = np.linspace(0, 1, plotThisScoreValueIntervals + 2)[1:-1]
#        #To make it work with centered coordinates:
#        toTake = toTake - 0.5 if len(toTake) % 3 == 0 else -0.75
#        for tick, text in zip(plotThisScoreValue.axes.get_xticks(), plotThisScoreValue.axes.get_xticklabels()):
#            xLabel = text.get_text()  # "X" or "Y"
#            for num, subcat in enumerate(np.unique(subsetOfDataForThisValueOnly["anchorType"])):
#                print(tick + toTake[num])
#                print(subcat)
#                
#                # calculate the median value for all replicates of either X or Y
#                medianVal = subsetOfDataForThisValueOnly[subsetOfDataForThisValueOnly.anchorType == subcat].median()[0]
#                print(medianVal)
#                # plot horizontal lines across the column, centered on the tick
#                plotThisScoreValue.axes.plot([tick + toTake[num]-0.04, tick + toTake[num]+0.02],
#                               [medianVal, medianVal],
#                        lw=2, color='k', linestyle = "-")
        #plotThisScoreValue.axes.set_ylim([0.2, 1.05])
        plotThisScoreValue.axes.set_ylabel("Score (median over 10 folds for 5 repeats)")
        plotThisScoreValue.axes.set_xlabel("")
        dictPlotsClassifierPerformanceScores[value + "SeparatePlot"] = [plotThisScoreValue.figure, plotThisScoreValue]
    
    #also add in the per-rep barplots from above and then we can use this dictionary
    #as the classifier performance scores dictionary for the final pickle
    dictPlotsClassifierPerformanceScores["scoresPerRepPerMetric"] = listOfBarPlots
    



#make confusion matrices
    #confMatricesPerRepFigs = []
    #confMatricesPerRepAxes = []
    confMatricesPerRep     = {}
    
    for number, (name, group) in enumerate(summaryStatisticsDF.groupby("anchorType")):
        confMatricesPerRep[name + "_confMatrixPerRep"] = []
        for rep, repData in group.iterrows():
            
            cm = np.array([[repData["truePosConf_median"], repData["falsePosConf_median"]],
                  [repData["falseNegConf_median"], repData["trueNegConf_median"]]])
            cmstd = np.array([[repData["truePosConf_std"], repData["falsePosConf_std"]],
                  [repData["falseNegConf_std"], repData["trueNegConf_std"]]])
            
            print(cmstd)
            
            cmap = plt.cm.Blues
            fig, ax = plt.subplots()
            fig.suptitle("Anchorsize: " + name, y = 0.98 + 0.075, ha='center', va='bottom')
            plt.grid(False)
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
            ax.figure.colorbar(im, ax=ax)
            classes = ["True", "False"]
            title = "Confusion Matrix " + "rep " + str(rep)
            # We want to show all ticks...
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   # ... and label them with the respective list entries
                   xticklabels=classes, yticklabels=classes,
                   title=title,
                   ylabel='True label',
                   xlabel='Predicted label')
            ax.title.set_fontsize(8)
        
            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
        
            # Loop over data dimensions and create text annotations.
            fmt = '.2f' 
            thresh = cm.max() / 6 * 4.
            print(thresh)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt) + " ± " + format(cmstd[i, j], fmt) + "\n(median ± std.)",
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            #florpie, borpie, glorpie = fig, ax, cm
            #confMatricesPerRepFigs.append(fig)
            #confMatricesPerRepAxes.append(ax)
            confMatricesPerRep[name + "_confMatrixPerRep"].append([fig, ax, cm])
        

#confMatricesPerRepFigs[0]


    #confMatricesNormPerRepFigs = []
    #confMatricesNormPerRepAxes = []
    confMatricesNormPerRep     = {}
    
    for number, (name, group) in enumerate(summaryStatisticsDF.groupby("anchorType")):
        confMatricesNormPerRep[name + "_normalisedConfMatrixPerRep"] = []
        for rep, repData in group.iterrows():
            
            cm = np.array([[repData["truePosConfNorm_median"], repData["falsePosConfNorm_median"]],
                  [repData["falseNegConfNorm_median"], repData["trueNegConfNorm_median"]]])
            cmstd = np.array([[repData["truePosConfNorm_std"], repData["falsePosConfNorm_std"]],
                  [repData["falseNegConfNorm_std"], repData["trueNegConfNorm_std"]]])
            
            print(cmstd)
            
            cmap = plt.cm.Blues
            fig, ax = plt.subplots()
            fig.suptitle("Anchorsize: " + name, y = 0.98 + 0.075, ha='center', va='bottom')
            plt.grid(False)
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
            ax.figure.colorbar(im, ax=ax)
            classes = ["True", "False"]
            title = "Normalised Confusion Matrix " + "rep " + str(rep)
            # We want to show all ticks...
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   # ... and label them with the respective list entries
                   xticklabels=classes, yticklabels=classes,
                   title=title,
                   ylabel='True label',
                   xlabel='Predicted label')
            ax.title.set_fontsize(8)
        
            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
        
            # Loop over data dimensions and create text annotations.
            fmt = '.2f' 
            thresh = cm.max() / 6 * 4.
            print(thresh)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt) + " ± " + format(cmstd[i, j], fmt) + "\n(median ± std.)",
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            #florpie, borpie, glorpie = fig, ax, cm
            #confMatricesNormPerRepFigs.append(fig)
            #confMatricesNormPerRepAxes.append(ax)
            confMatricesNormPerRep[name + "_normalisedConfMatrixPerRep"].append([fig, ax, cm])
            
        
        
    #One normalised confusion matrix for every classifier (i.e. one for 2500, one for 10000, one for both together)
    confMatricesNormPerAnchor     = {}
    
    for number, (name, group) in enumerate(summaryStatisticsDF.groupby("anchorType")):
        print(name); print(group)
        dataToGetMedians = group.median()
        dataToGetStd     = group.std()
        
        cm = np.array([[dataToGetMedians.loc["truePosConfNorm_median"],
                        dataToGetMedians.loc["falsePosConfNorm_median"]],
              [dataToGetMedians.loc["falseNegConfNorm_median"],
               dataToGetMedians.loc["trueNegConfNorm_median"]]])
        cmstd = np.array([[dataToGetStd.loc["truePosConfNorm_median"],
                        dataToGetStd.loc["falsePosConfNorm_median"]],
              [dataToGetStd.loc["falseNegConfNorm_median"],
               dataToGetStd.loc["trueNegConfNorm_median"]]])
        
        print(cmstd)
        
        cmap = plt.cm.Blues
        fig, ax = plt.subplots()
        #fig.suptitle("Anchorsize: " + name, y = 0.98 + 0.075, ha='center', va='bottom')
        plt.grid(False)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        classes = ["True", "False"]
        title = "Normalised Confusion Matrix " + "for anchor size " + str(name)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        ax.title.set_fontsize(11)
    
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' 
        thresh = cm.max() / 6 * 4.
        print(thresh)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt) + " ± " + format(cmstd[i, j], fmt) + "\n(median ± std.)",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        confMatricesNormPerAnchor[name + "normConfMatrixPerAnchor"] = [fig, ax, cm]
        
        
        
    confMatricesPerAnchor     = {}
    
    for number, (name, group) in enumerate(summaryStatisticsDF.groupby("anchorType")):
        print(name); print(group)
        dataToGetMedians = group.median()
        dataToGetStd     = group.std()
        
        cm = np.array([[dataToGetMedians.loc["truePosConf_median"],
                        dataToGetMedians.loc["falsePosConf_median"]],
              [dataToGetMedians.loc["falseNegConf_median"],
               dataToGetMedians.loc["trueNegConf_median"]]])
        cmstd = np.array([[dataToGetStd.loc["truePosConf_median"],
                        dataToGetStd.loc["falsePosConf_median"]],
              [dataToGetStd.loc["falseNegConf_median"],
               dataToGetStd.loc["trueNegConf_median"]]])
        
        print(cmstd)
        
        cmap = plt.cm.Blues
        fig, ax = plt.subplots()
        #fig.suptitle("Anchorsize: " + name, y = 0.98 + 0.075, ha='center', va='bottom')
        plt.grid(False)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        classes = ["True", "False"]
        title = "Confusion Matrix " + "for anchor size " + str(name)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        ax.title.set_fontsize(11)
    
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' 
        thresh = cm.max() / 6 * 4.
        print(thresh)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt) + " ± " + format(cmstd[i, j], fmt) + "\n(median ± std.)",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        confMatricesPerAnchor[name + "confMatrixPerAnchor"] = [fig, ax, cm]
        
        
        confMatrixDictTotal = {}
        
        for (k1,v1), (k2, v2) in zip(confMatricesPerRep.items(),
            confMatricesNormPerRep.items()):
            confMatrixDictTotal[k1] = v1,
            confMatrixDictTotal[k2] = v2
        
        for (k1,v1), (k2, v2) in zip(confMatricesPerAnchor.items(),
            confMatricesNormPerAnchor.items()):
            confMatrixDictTotal[k1] = v1,
            confMatrixDictTotal[k2] = v2
        


##plotting feature importances for each rep, for each anchor
#    featImportancePlotsPerAnchor = {}
#    for anchor, dataFrameFeatImps in featureImportancesPerAnchorDict.items():
#        dataFrameFeatImps = dataFrameFeatImps["summaryStats"]
#        figFeatImp, axFeatImp = plt.subplots(1,5, figsize = (40,5), sharey = True)
#        uniqueReps = set([re.match(".+_(rep\d+)$", colName)[1] for colName in dataFrameFeatImps.columns])
#        
#        #first, gather all the features that are high, and assign each of them a colour for consistency between plots.
#        listOfFeats = []
#        for repNr in uniqueReps:
#            featNames = dataFrameFeatImps.loc[:, "featureNames_" + repNr][0: 20]
#            listOfFeats.append(featNames.tolist())
#        listOfFeats = [entry for sublist in listOfFeats for entry in sublist]
#        uniqueFeatNamesInPlot = set(listOfFeats)
#        colours = sb.color_palette("hls", n_colors = len(uniqueFeatNamesInPlot))
#        colorDict = dict(zip(uniqueFeatNamesInPlot, colours))
#            
#        for repNr in uniqueReps:
#            #reps are not ordered, want them ordered in plots. Reps start at rep 1, so -1 for correct Python indexing
#            plotToAssignTo = int(re.match("rep(\d+)", repNr)[1]) - 1
#            
#            featImportances = dataFrameFeatImps.loc[:, "medianFeatureImportancesOverCrossVals_" + repNr][0:20]
#            featNames = dataFrameFeatImps.loc[:, "featureNames_" + repNr][0: 20]
#            featStds  = dataFrameFeatImps.loc[:, "stdFeatureImportancesOverCrossVals_" + repNr][0:20]
#            coolPlot = sb.barplot(x=featNames, y=featImportances, ax = axFeatImp[plotToAssignTo],
#                                  palette = colorDict, yerr = featStds , capsize = 2#,
#                                  #edgecolor = "black", linewidth = 1
#                                  )
#            axFeatImp[plotToAssignTo].set_xticklabels(labels = featNames, rotation = 65, ha = "right")
#            #I want one colour for every feature in the figure, shared per feature
#            
#        featImportancePlotsPerAnchor[anchor] = [figFeatImp, axFeatImp]
#    



#using the complete values, draw a boxplot
        
    
    
#    tips = sb.load_dataset("tips")
#    IE = sb.boxplot(x=tips["total_bill"])
#    
#    
#    featImportanceBoxPlotsPerAnchor = {}
#    for anchor, dataFrameFeatImps in featureImportancesPerAnchorDict.items():
#        dataFrameFeatImps = dataFrameFeatImps["completeValues"]
#        figFeatImp, axFeatImp = plt.subplots(1,5, figsize = (40,5), sharey = True)
#        uniqueReps = set(dataFrameFeatImps["rep"])
#        
#        #first, gather all the features that are high, and assign each of them a colour for consistency between plots.
#        listOfFeats = []
#        for repNr in uniqueReps:
#            featNames = dataFrameFeatImps.loc[:, "featureNames_" + repNr][0: 20]
#            listOfFeats.append(featNames.tolist())
#        listOfFeats = [entry for sublist in listOfFeats for entry in sublist]
#        uniqueFeatNamesInPlot = set(listOfFeats)
#        colours = sb.color_palette("hls", n_colors = len(uniqueFeatNamesInPlot))
#        colorDict = dict(zip(uniqueFeatNamesInPlot, colours))
#            
#        for repNr in uniqueReps:
#            #reps are not ordered, want them ordered in plots. Reps start at rep 1, so -1 for correct Python indexing
#            plotToAssignTo = int(re.match("rep(\d+)", repNr)[1]) - 1
#            
#            featImportances = dataFrameFeatImps.loc[:, "medianFeatureImportancesOverCrossVals_" + repNr][0:20]
#            featNames = dataFrameFeatImps.loc[:, "featureNames_" + repNr][0: 20]
#            featStds  = dataFrameFeatImps.loc[:, "stdFeatureImportancesOverCrossVals_" + repNr][0:20]
#            coolPlot = sb.barplot(x=featNames, y=featImportances, ax = axFeatImp[plotToAssignTo],
#                                  palette = colorDict, yerr = featStds , capsize = 2#,
#                                  #edgecolor = "black", linewidth = 1
#                                  )
#            axFeatImp[plotToAssignTo].set_xticklabels(labels = featNames, rotation = 65, ha = "right")
#            #I want one colour for every feature in the figure, shared per feature
#            
#        featImportanceBoxPlotsPerAnchor[anchor] = [figFeatImp, axFeatImp]
    
    
    
    featImportanceBoxPlotsPerAnchor = {}
    for anchor, dataFrameFeatImps in featureImportancesPerAnchorDict.items():
        dataFrameFeatImps = dataFrameFeatImps["completeValues"]
        
        figFeatImpBox, axFeatImpBox = plt.subplots(1,5, figsize = (80,15), sharey = True)
        
        
        #first get a list of all features important in any rep
        #use this to populate a dictionary with unique colours for each of these feats:
        dataAllReps = None
        for repNr in dataFrameFeatImps["rep"].unique():
            
            dataSelection = featureImportancesPerAnchorDict[anchor]["completeValues"][featureImportancesPerAnchorDict[anchor]["completeValues"]["rep"] == repNr]
            #for the data for this rep, group by crossVal, and per crossVal, sort
            topDataForEveryCrossVal = None
            for name, group in dataSelection.groupby("crossVal"):
                group.sort_values(inplace = True, ascending = False, by = "featureImportanceArray")
                #take the top 20
                groupToKeep = group.head(20)
                topDataForEveryCrossVal = pd.concat([topDataForEveryCrossVal,
                                                     groupToKeep],
                axis = 0)
            dataAllReps = pd.concat([dataAllReps, 
                                     topDataForEveryCrossVal],
                                    axis = 0)
        uniqueFeaturesToPlot = dataAllReps["featNames"].unique()
        random.shuffle(uniqueFeaturesToPlot)
        colours = sb.color_palette("hls", n_colors = len(uniqueFeaturesToPlot))
        colorDict = dict(zip(uniqueFeaturesToPlot, colours))
        
        for repNr in dataFrameFeatImps["rep"].unique():
            sb.boxplot(data = dataAllReps[dataAllReps["rep"] == repNr],
               x = "featNames",
               y = "featureImportanceArray",
               ax = axFeatImpBox[int(repNr) - 1],
               palette = colorDict
               )
            sb.swarmplot(data = dataAllReps[dataAllReps["rep"] == repNr],
               x = "featNames",
               y = "featureImportanceArray",
               ax = axFeatImpBox[int(repNr) - 1],
               palette = colorDict,
               edgecolor = "black",
               linewidth = .4,
               size = 2)
            axFeatImpBox[int(repNr) - 1].set_xticklabels(labels = axFeatImpBox[int(repNr) - 1].get_xticklabels(),
                        rotation = 65, ha = "right")
            axFeatImpBox[int(repNr) - 1].yaxis.set_tick_params(labelleft = True)
            axFeatImpBox[int(repNr) - 1].set_xlabel("Feature names rep " + str(repNr))
            
        featImportanceBoxPlotsPerAnchor[anchor] = [figFeatImpBox, axFeatImpBox]


    featImportanceBoxPlotsOneForWholeClassifier = {}
    dataToPutIn = pd.concat([value["completeValues"] for key, value in featureImportancesPerAnchorDict.items()],
                             axis = 0)
    plotForAllAnchorsMeanMedianCombinedFig, plotForAllAnchorsMeanMedianCombinedAx = plt.subplots(1,3, figsize = (40,15), sharey = True)
    plotForAllAnchorsMeanFig, plotForAllAnchorsMeanAx = plt.subplots(1,3, figsize = (40,15), sharey = True)
    plotForAllAnchorsMedianFig, plotForAllAnchorsMedianAx = plt.subplots(1,3, figsize = (40,15), sharey = True)
    
    featureNamesMostImportantAcrossClassifiers = {}
    dataForTheseFeatureNames = {}
    #perAnchorDataForBoxPlotsFeatImp = None
    for num, (anchor, thisAnchorData) in enumerate(dataToPutIn.groupby("anchor")):
        repDataFeatImpMeansAndMedians = None
        perRepDataThisAnchor = thisAnchorData.groupby("rep")
        for repNr, thisRepData in perRepDataThisAnchor:
            groupedByFeat = thisRepData.groupby("featNames")
            meanPerFeatureThisRep = pd.DataFrame({"summaryStatValue"   : groupedByFeat["featureImportanceArray"].mean(),
                                                  "summaryStatName" : ["mean"] *  len(groupedByFeat),
                                                  "rep"    : [int(repNr)] * len(groupedByFeat),
                                                  "anchor" : [anchor] * len(groupedByFeat)})
            medianPerFeatureThisRep = pd.DataFrame({"summaryStatValue"   : groupedByFeat["featureImportanceArray"].median(),
                                                  "summaryStatName" : ["median"] *  len(groupedByFeat),
                                                  "rep"    : [int(repNr)] * len(groupedByFeat),
                                                  "anchor" : [anchor] * len(groupedByFeat)
                                                  })
            repDataFeatImpMeansAndMedians = pd.concat([repDataFeatImpMeansAndMedians,
                                                       meanPerFeatureThisRep,
                                                       medianPerFeatureThisRep],
                axis = 0)
            
            
            
        #okay, we have the data per rep. Now let us plot it for this anchor.
        topNFeaturesToSelect = 15
        #calculate summary stats for this anchor, see which N features are highest, draw boxplots for those
        #Just take the median of the medians, this is only to determine the top N highest
        medianOnly = repDataFeatImpMeansAndMedians[
                repDataFeatImpMeansAndMedians["summaryStatName"] == "median"]
        groupedPerFeatThisAnchor = medianOnly.groupby(
                medianOnly.index)
        medianFeatImpThisAnchor = pd.DataFrame({"summaryStatValue" : groupedPerFeatThisAnchor["summaryStatValue"].median()})
        medianFeatImpThisAnchor.sort_values(by = "summaryStatValue", ascending = False,
                                            inplace = True)
        #pick the highest ones, plot those
        highestValuedFeatImp = medianFeatImpThisAnchor.index.values[0:topNFeaturesToSelect]
        subsetOfDataToPlot = repDataFeatImpMeansAndMedians[
                repDataFeatImpMeansAndMedians.index.isin(highestValuedFeatImp)]
        subsetOfDataToPlot.reset_index(inplace = True)
        sb.boxplot(data = subsetOfDataToPlot, x = "featNames", y = "summaryStatValue",
                   hue = "summaryStatName", ax = plotForAllAnchorsMeanMedianCombinedAx[num])
        sb.swarmplot(data = subsetOfDataToPlot, x = "featNames", y = "summaryStatValue",
                   hue = "summaryStatName", ax = plotForAllAnchorsMeanMedianCombinedAx[num],
                   edgecolor = "black", linewidth = .2)
        plotForAllAnchorsMeanMedianCombinedAx[num].set_xticklabels(labels = plotForAllAnchorsMeanMedianCombinedAx[num].get_xticklabels(),
                            rotation = 65, ha = "right")
        plotForAllAnchorsMeanMedianCombinedAx[num].yaxis.set_tick_params(labelleft = True)
        plotForAllAnchorsMeanMedianCombinedAx[num].set_xlabel("Top " + str(topNFeaturesToSelect) + \
                                   " feature importances for anchor size: " + str(anchor))
        
        #add to dicts for unified plot
        featureNamesMostImportantAcrossClassifiers[anchor] = highestValuedFeatImp
        dataForTheseFeatureNames[anchor] = subsetOfDataToPlot
        
        ##Mean only
        
        
        subsetToPlotMeanOnly = subsetOfDataToPlot[subsetOfDataToPlot["summaryStatName"] == "mean"]
        
        my_order = subsetToPlotMeanOnly.groupby(by=["featNames"])["summaryStatValue"].mean().sort_values(ascending=False).index
        
        sb.swarmplot(data = subsetToPlotMeanOnly, x = "featNames", y = "summaryStatValue",
                    ax = plotForAllAnchorsMeanAx[num], hue = "featNames",
                    order = my_order, hue_order = my_order)
        plotForAllAnchorsMeanAx[num].set_xticklabels(labels = plotForAllAnchorsMeanAx[num].get_xticklabels(),
                            rotation = 65, ha = "right")
        plotForAllAnchorsMeanAx[num].yaxis.set_tick_params(labelleft = True)
        plotForAllAnchorsMeanAx[num].set_xlabel("Top " + str(topNFeaturesToSelect) + \
                                   " feature importances for anchor size: " + str(anchor))
        plotForAllAnchorsMeanAx[num].set_xlabel("Top " + str(topNFeaturesToSelect) + \
                                   " feature importances for anchor size: " + str(anchor))
        
        plotForAllAnchorsMeanAx[num].set_ylabel("Mean feature importances")
        
        
        #draw mean lines
        meanWidth = 0.48
    
        for tick, text in zip(plotForAllAnchorsMeanAx[num].get_xticks(), plotForAllAnchorsMeanAx[num].get_xticklabels()):
            featName = text.get_text()  # "X" or "Y"
    
            # calculate the median value for all replicates of either X or Y
            meanVal = subsetToPlotMeanOnly[subsetToPlotMeanOnly['featNames']==featName]["summaryStatValue"].mean()
    
            # plot horizontal lines across the column, centered on the tick
            plotForAllAnchorsMeanAx[num].plot([tick-meanWidth/2, tick+meanWidth/2], [meanVal, meanVal],
                    lw=2, color='k', linestyle = "--")
        
        
        
        #median only
        
    #    
        
        subsetToPlotMedianOnly = subsetOfDataToPlot[subsetOfDataToPlot["summaryStatName"] == "median"]
        
        my_order = subsetToPlotMedianOnly.groupby(by=["featNames"])["summaryStatValue"].median().sort_values(ascending=False).index
        
        sb.swarmplot(data = subsetToPlotMedianOnly, x = "featNames", y = "summaryStatValue",
                    ax = plotForAllAnchorsMedianAx[num], hue = "featNames",
                    order = my_order, hue_order = my_order)
        plotForAllAnchorsMedianAx[num].set_xticklabels(labels = plotForAllAnchorsMedianAx[num].get_xticklabels(),
                            rotation = 65, ha = "right")
        plotForAllAnchorsMedianAx[num].yaxis.set_tick_params(labelleft = True)
        plotForAllAnchorsMedianAx[num].set_xlabel("Top " + str(topNFeaturesToSelect) + \
                                   " feature importances for anchor size: " + str(anchor))
        
        plotForAllAnchorsMedianAx[num].set_ylabel("Median feature importances")
        
        #draw median lines
        medianWidth = 0.48
    
        for tick, text in zip(plotForAllAnchorsMedianAx[num].get_xticks(), plotForAllAnchorsMedianAx[num].get_xticklabels()):
            featName = text.get_text()  # "X" or "Y"
    
            # calculate the median value for all replicates of either X or Y
            medianVal = subsetToPlotMedianOnly[subsetToPlotMedianOnly['featNames']==featName]["summaryStatValue"].median()
    
            # plot horizontal lines across the column, centered on the tick
            plotForAllAnchorsMedianAx[num].plot([tick-medianWidth/2, tick+medianWidth/2], [medianVal, medianVal],
                    lw=2, color='k', linestyle = "--")
        
        
        
        
        
        
        
        
    featImportanceBoxPlotsOneForWholeClassifier["medianOnly"]        = [plotForAllAnchorsMedianFig, plotForAllAnchorsMedianAx]
    featImportanceBoxPlotsOneForWholeClassifier["meanOnly"]          = [plotForAllAnchorsMeanFig, plotForAllAnchorsMeanAx]
    featImportanceBoxPlotsOneForWholeClassifier["compareMeanMedian"] = [plotForAllAnchorsMeanMedianCombinedFig, plotForAllAnchorsMeanMedianCombinedAx]
    
    
    
    #comparison of the most important features across the 3 classifiers.
    #based purely on the feature names.
    #I do this on the already selected top 10 features per classifier. 
    #The reason is that if you do this pre-selection (by taking the median
    #across features across classifiers), the classifier with both anchors
    #matters too much: it will always have the same feature twice (once for each anchor size)
    #and hence influence the median more.
    #Better to select from the top for each class. separately, and only then
    #do this matching. If both variants are in the top 10 in the classifier with
    #both anchor types, it is apparently really important.That is fine.
    
    
#    featureNamesMostImportantAcrossClassifiers.items()
#    dataForTheseFeatureNames.items()
#    dataFrameComparisonBetweenClassifiers = None
#    for key, values in featureNamesMostImportantAcrossClassifiers.items():
#        regexMatches = [re.match("^\w+_\d+_([0-9A-Za-z-]+_.+$)", entry)[1] for entry in values]
#        pdCompatible = {"anchor" : [key] * len(values),
#                        "fullNames" : values,
#                        "featNamesWithoutAnchorSize" : regexMatches}
#        dataFrameComparisonBetweenClassifiers = pd.concat([dataFrameComparisonBetweenClassifiers,
#                   pd.DataFrame(pdCompatible)])
#    
#    #Will do this only for the median
#    columnToAddMedianValuesPerAnchorOfTopNFeatures = None
#    for key, values in dataForTheseFeatureNames.items():
#        namesToFind = dataFrameComparisonBetweenClassifiers[
#                dataFrameComparisonBetweenClassifiers.anchor == key]["fullNames"]
#        subsetToSearch = values[values.summaryStatName == "median"]
#        subsetToSearch = subsetToSearch[subsetToSearch.featNames.isin(namesToFind)]
#        
#        #gather these values and take their medians
#        valuesToAddHere = subsetToSearch.groupby("featNames")["summaryStatValue"].median().reset_index().iloc[:,1]
#        columnToAddMedianValuesPerAnchorOfTopNFeatures = pd.concat([
#                columnToAddMedianValuesPerAnchorOfTopNFeatures,
#                valuesToAddHere])
#        
#    dataFrameComparisonBetweenClassifiers = pd.concat([dataFrameComparisonBetweenClassifiers,
#                                                       columnToAddMedianValuesPerAnchorOfTopNFeatures],
#        axis = 1)
#    
#    #add in count for hue
#    dataFrameComparisonBetweenClassifiers["count"] = ""
#    
#    #make the plot
#    figCombinedFeatImportances, axCombinedFeatImportances = plt.subplots(1,1, figsize = (10,20))
#    dataFrameForPlottingOrder = {"count" : dataFrameComparisonBetweenClassifiers.groupby(by=[
#            "featNamesWithoutAnchorSize"])["summaryStatValue"].count(),
#            "median" : dataFrameComparisonBetweenClassifiers.groupby(by=[
#            "featNamesWithoutAnchorSize"])[
#            "summaryStatValue"].median()}
#    dataFrameForPlottingOrder = pd.DataFrame(dataFrameForPlottingOrder)
#    dataFrameForPlottingOrder.sort_values(by = ["count", "median"],
#                                          ascending = False, inplace = True)
#    dataFrameForPlottingOrder
#    my_order = dataFrameForPlottingOrder.index
#    my_hue_order = [str(entry) for entry in sorted(np.unique(dataFrameForPlottingOrder["count"].reset_index(drop = True).tolist()), reverse = True)]
#    #add the count back into the original dataFrame for the hue
#    for featureName, row in dataFrameForPlottingOrder.iterrows():
#                dataFrameComparisonBetweenClassifiers.loc[
#                        dataFrameComparisonBetweenClassifiers[
#                                "featNamesWithoutAnchorSize"] == featureName, "count"] = str(int(row["count"]))
#    
#    
#    sb.swarmplot(data = dataFrameComparisonBetweenClassifiers,
#                 x = "featNamesWithoutAnchorSize",
#                 y = "summaryStatValue",
#                ax = axCombinedFeatImportances, hue = "count",
#                order = my_order, hue_order = my_hue_order
#                )
#    axCombinedFeatImportances.set_xticklabels(labels = axCombinedFeatImportances.get_xticklabels(),
#                        rotation = 65, ha = "right")
#    axCombinedFeatImportances.yaxis.set_tick_params(labelleft = True)
#    axCombinedFeatImportances.set_xlabel("Top " + str(topNFeaturesToSelect) + \
#                               " feature importances across classifiers")
#    
#    axCombinedFeatImportances.set_ylabel("Median feature importances")
#    
#    #draw median lines
#    medianWidth = 0.48
#    
#    for tick, text in zip(axCombinedFeatImportances.get_xticks(),
#                          axCombinedFeatImportances.get_xticklabels()):
#        featName = text.get_text()  # "X" or "Y"
#    
#        # calculate the median value for all replicates of either X or Y
#        medianVal = dataFrameComparisonBetweenClassifiers[
#                dataFrameComparisonBetweenClassifiers["featNamesWithoutAnchorSize"]==featName]["summaryStatValue"].median()
#        
#        # plot horizontal lines across the column, centered on the tick
#        axCombinedFeatImportances.plot([
#                tick-medianWidth/2, tick+medianWidth/2],
#                [medianVal, medianVal],
#                lw=2, color='k', linestyle = "--")
#        
    #Note that the plot now generated combines:
    #-features in the left and right part of the anchor
    #-features across anchor sizes
    #so, irrespective of anchor size and position in the anchor, what is important?
    
    #further plots could look at:
    #-anchor position and feature name without exact feature (is there
    #a difference in what is important at what part of the anchor? Or would
    #we do better to just combine it?)
    
    #Just the chromatin mark level: which chromatin marks are chosen?



    #Okay then. So, I want to cluster by factor name, or by anchor position, or by
    #sub-anchor position. Moreover, I would like to summarise leftAnchorRight and
    #rightAnchorLeft together, because they are both on the in-between side, and
    #leftAnchorLeft and rightAnchorRight together because they are both on the
    #outsides of the loop (see final presentation picture)
    
    fullDataFrameComparisonBetweenClassifiers = None
    for key, values in featureNamesMostImportantAcrossClassifiers.items():
        
        lenValues = len(values)
        rangeValues = range(0, lenValues)
        valuesToTakeForAnchorFeatures = values
        inbetweenRegexMatches = [re.match("^inbetween.*", entry) for entry in values]
        inbetweenRegexMatchesIndex = [i for i, match in enumerate(inbetweenRegexMatches) if match is not None]
        notInbetweenMatchesIndex = [i for i, match in enumerate(inbetweenRegexMatches) if match is None]
        
        #if there are inbetween features
        #Note that the code for this is very messy and tacked-on, which is exactly what it is.
        #Sorry for that.
        if len(inbetweenRegexMatches) > 0:
            valuesToTakeForAnchorFeatures = [value for i, value in enumerate(values) if i not in inbetweenRegexMatchesIndex]
            
            inbetweenRegexMatchesData = [re.match("^(inbetween)_(\d+)_(([0-9A-Za-z-]+)_(.+$))", values[i]) for i in inbetweenRegexMatchesIndex]
            
            fullNameInDataInbetween                = [entry[0] for entry in inbetweenRegexMatchesData]
            classifierInputAnchorSizeInbetween     = [key] * len(fullNameInDataInbetween)
            #all these anchor splits don't exist for inbetween, so just paste inbetween in every column to be sure for now
            fullAnchorPositionDesignationInbetween = [entry[1] for entry in inbetweenRegexMatchesData]
            anchorPositionLeftRightInbetween       = [entry[1] for entry in inbetweenRegexMatchesData]
            withinAnchorPositionLeftRightInbetween = [entry[1] for entry in inbetweenRegexMatchesData]
            anchorSizeInbetween                    = [entry[2] for entry in inbetweenRegexMatchesData]
            markNameAndFeatureNameInbetween        = [entry[3] for entry in inbetweenRegexMatchesData]
            markNameOnlyInbetween                  = [entry[4] for entry in inbetweenRegexMatchesData]
            featureNameOnlyInbetween               = [entry[5] for entry in inbetweenRegexMatchesData]
            
            
            
            
        
        
        regexMatches = [re.match(
                "^((left|right|)(AnchorLeft|AnchorRight))_(\d+)_(([0-9A-Za-z-]+)_(.+$))",
                entry) for entry in valuesToTakeForAnchorFeatures]
        fullNameInData                = [entry[0] for entry in regexMatches]
        classifierInputAnchorSize     = [key] * len(fullNameInData)
        fullAnchorPositionDesignation = [entry[1] for entry in regexMatches]
        anchorPositionLeftRight       = [entry[2] for entry in regexMatches]
        withinAnchorPositionLeftRight = [entry[3] for entry in regexMatches]
        anchorSize                    = [entry[4] for entry in regexMatches]
        markNameAndFeatureName        = [entry[5] for entry in regexMatches]
        markNameOnly                  = [entry[6] for entry in regexMatches]
        featureNameOnly               = [entry[7] for entry in regexMatches]
        
        #if there are inbetween features, we need to combine these lists
        if(len(inbetweenRegexMatches) > 0):
            finalFullNameInData, finalClassifierInputAnchorSize, \
            finalFullAnchorPositionDesignation, finalAnchorPositionLeftRight, \
            finalWithinAnchorPositionLeftRight, finalAnchorSize, \
            finalMarkNameAndFeatureName, finalMarkNameOnly, \
            finalFeatureNameOnly = \
            [], [], [], [], [], [], [], [], []
            
            lenValues = len(values)
            rangeValues = range(0, lenValues)
            counterNormalList = 0
            counterInbetweenList = 0
            
            for i in rangeValues:
                if i in notInbetweenMatchesIndex:
                    finalFullNameInData.append(fullNameInData[counterNormalList])
                    finalClassifierInputAnchorSize.append(classifierInputAnchorSize[counterNormalList])
                    finalFullAnchorPositionDesignation.append(fullAnchorPositionDesignation[counterNormalList])
                    finalAnchorPositionLeftRight.append(anchorPositionLeftRight[counterNormalList])
                    finalWithinAnchorPositionLeftRight.append(withinAnchorPositionLeftRight[counterNormalList])
                    finalAnchorSize.append(anchorSize[counterNormalList])
                    finalMarkNameAndFeatureName.append(markNameAndFeatureName[counterNormalList])
                    finalMarkNameOnly.append(markNameOnly[counterNormalList])
                    finalFeatureNameOnly.append(featureNameOnly[counterNormalList])
                    counterNormalList += 1
                else:
                    finalFullNameInData.append(fullNameInDataInbetween[counterInbetweenList])
                    finalClassifierInputAnchorSize.append(classifierInputAnchorSizeInbetween[counterInbetweenList])
                    finalFullAnchorPositionDesignation.append(fullAnchorPositionDesignationInbetween[counterInbetweenList])
                    finalAnchorPositionLeftRight.append(anchorPositionLeftRightInbetween[counterInbetweenList])
                    finalWithinAnchorPositionLeftRight.append(withinAnchorPositionLeftRightInbetween[counterInbetweenList])
                    finalAnchorSize.append(anchorSizeInbetween[counterInbetweenList])
                    finalMarkNameAndFeatureName.append(markNameAndFeatureNameInbetween[counterInbetweenList])
                    finalMarkNameOnly.append(markNameOnlyInbetween[counterInbetweenList])
                    finalFeatureNameOnly.append(featureNameOnlyInbetween[counterInbetweenList])
                    counterInbetweenList += 1
                    
            dFThisClassifier = pd.DataFrame({
                "fullNameInData" : finalFullNameInData,
                "classifierInputAnchorSize" : finalClassifierInputAnchorSize,
                "fullAnchorPositionDesignation" : finalFullAnchorPositionDesignation,
                "anchorPositionLeftRight" : finalAnchorPositionLeftRight,
                "withinAnchorPositionLeftRight" : finalWithinAnchorPositionLeftRight,
                "anchorSize" : finalAnchorSize,
                "markNameAndFeatureName" : finalMarkNameAndFeatureName,
                "markNameOnly" : finalMarkNameOnly,
                "featureNameOnly" : finalFeatureNameOnly})
        #if all this is not happening, and there is no inbetween, just rename the earlier list.
        else:
            dFThisClassifier = pd.DataFrame({
                "fullNameInData" : fullNameInData,
                "classifierInputAnchorSize" : classifierInputAnchorSize,
                "fullAnchorPositionDesignation" : fullAnchorPositionDesignation,
                "anchorPositionLeftRight" : anchorPositionLeftRight,
                "withinAnchorPositionLeftRight" : withinAnchorPositionLeftRight,
                "anchorSize" : anchorSize,
                "markNameAndFeatureName" : markNameAndFeatureName,
                "markNameOnly" : markNameOnly,
                "featureNameOnly" : featureNameOnly})
            
                
        
        
        fullDataFrameComparisonBetweenClassifiers = pd.concat([fullDataFrameComparisonBetweenClassifiers,
                                                               dFThisClassifier])
        
        print(fullDataFrameComparisonBetweenClassifiers.head())
        print(fullDataFrameComparisonBetweenClassifiers.tail())
            
     #Will do this only for the median
    columnToAddMedianValuesPerAnchorOfTopNFeatures = None
    for key, values in dataForTheseFeatureNames.items():
        namesToFind = fullDataFrameComparisonBetweenClassifiers[
                fullDataFrameComparisonBetweenClassifiers.classifierInputAnchorSize == key]["fullNameInData"]
        subsetToSearch = values[values.summaryStatName == "median"]
        subsetToSearch = subsetToSearch[subsetToSearch.featNames.isin(namesToFind)]
        
        #gather these values and take their medians
        valuesToAddHere = subsetToSearch.groupby("featNames")["summaryStatValue"].median().reset_index().iloc[:,1]
        columnToAddMedianValuesPerAnchorOfTopNFeatures = pd.concat([
                columnToAddMedianValuesPerAnchorOfTopNFeatures,
                valuesToAddHere])
        
    fullDataFrameComparisonBetweenClassifiers = pd.concat([fullDataFrameComparisonBetweenClassifiers,
                                                       columnToAddMedianValuesPerAnchorOfTopNFeatures],
        axis = 1)
    
    #now we have a dataframe with for the three separate classifiers the median feature
    #importance of each of the top 15 features over all repeats. This dataFrame
    #can be used to make many different slices of the data.
    
    #first, let's make another column that says whether a feature is on the 
    #in-between side or the outsides of the loop.
    inBetweenSideOrOutside = []
    for index, row in fullDataFrameComparisonBetweenClassifiers.iterrows():
        inBetween = "inBetweenSide" if row["fullAnchorPositionDesignation"] == "leftAnchorRight" or \
                    row["fullAnchorPositionDesignation"] == "rightAnchorLeft" else None
        outerSide = "outerEdge" if row["fullAnchorPositionDesignation"] == "leftAnchorLeft" or \
                    row["fullAnchorPositionDesignation"] == "rightAnchorRight" else None
        if (inBetween)   : inBetweenSideOrOutside.append(inBetween)
        elif (outerSide) : inBetweenSideOrOutside.append(outerSide)
        #if these are single anchors which don't have this characteristic, just do ""
        else             : inBetweenSideOrOutside.append("")
    
    fullDataFrameComparisonBetweenClassifiers["withinAnchorInBetweenSideOrEdgeSide"] = inBetweenSideOrOutside
    
    #make counts for various combinations, so you can see in how many classifiers
    #every feature was chosen as important.
    
    
    #now for the plots: 
    #1. left anchor and right anchor, by chrom mark name only
    #2. left anchor and right anchor, by feature name only
    #3. left anchor and right anchor, by chrom mark_feature name combination
    #wouldn't expect too much difference here I think, perhaps some because convergent
    #CTCF motifs? No, you wouldn't see it in the kind of features I make.
    
    #make columns for the above
    fullDataFrameComparisonBetweenClassifiers["anchorPositionAndChromMarkName"] = \
        fullDataFrameComparisonBetweenClassifiers.anchorPositionLeftRight + \
        fullDataFrameComparisonBetweenClassifiers.markNameOnly
        
    fullDataFrameComparisonBetweenClassifiers["anchorPositionAndFeatureName"] = \
        fullDataFrameComparisonBetweenClassifiers.anchorPositionLeftRight + \
        fullDataFrameComparisonBetweenClassifiers.featureNameOnly
        
    fullDataFrameComparisonBetweenClassifiers["anchorPositionAndChromMarkFeatCombo"] = \
        fullDataFrameComparisonBetweenClassifiers.anchorPositionLeftRight + \
        fullDataFrameComparisonBetweenClassifiers.markNameAndFeatureName
        
    #4. inBetween side and outerEdge, by chrom mark only (i.e. what chrom marks are important on the inside and outside. Is there a difference?)
    #5. inBetween side and outerEdge, by feature name only
    #6. inBetween side and outerEdge, by chrom mark_feature name combination
    
    fullDataFrameComparisonBetweenClassifiers["withinAnchorSideAndChromMarkName"] = \
        fullDataFrameComparisonBetweenClassifiers.withinAnchorInBetweenSideOrEdgeSide + \
        fullDataFrameComparisonBetweenClassifiers.markNameOnly
        
    fullDataFrameComparisonBetweenClassifiers["withinAnchorSideAndFeatureName"] = \
        fullDataFrameComparisonBetweenClassifiers.withinAnchorInBetweenSideOrEdgeSide + \
        fullDataFrameComparisonBetweenClassifiers.featureNameOnly
        
    fullDataFrameComparisonBetweenClassifiers["withinAnchorSideAndChromMarkFeatCombo"] = \
        fullDataFrameComparisonBetweenClassifiers.withinAnchorInBetweenSideOrEdgeSide + \
        fullDataFrameComparisonBetweenClassifiers.markNameAndFeatureName
    
    #7. chrom mark only, in general (i.e. what chrom marks are informative for this question at all?)
    #8. feat name only (i.e. what features are chosen at all for this question?)
    #9. the combination
    
    #don't need to make columns for this, already exists as slices.
    
    #10. within an anchor, differences in chrom marks between left and right
    #11. within an anchor, differences in features between left and right
    #12. within an anchor, differences in chrom mark_features between L and R
    #--> if these differences are basically nonexistent, then there is no good reason
    #to do the split into left and right parts of an anchor.
    
    fullDataFrameComparisonBetweenClassifiers["anchorSideAndChromMark"] = \
        fullDataFrameComparisonBetweenClassifiers.withinAnchorPositionLeftRight + \
        fullDataFrameComparisonBetweenClassifiers.markNameOnly
    
    fullDataFrameComparisonBetweenClassifiers["anchorSideAndFeatName"] = \
        fullDataFrameComparisonBetweenClassifiers.withinAnchorPositionLeftRight + \
        fullDataFrameComparisonBetweenClassifiers.featureNameOnly
    
    fullDataFrameComparisonBetweenClassifiers["anchorSideAndChromMarkFeatNameCombo"] = \
        fullDataFrameComparisonBetweenClassifiers.withinAnchorPositionLeftRight + \
        fullDataFrameComparisonBetweenClassifiers.markNameAndFeatureName
    
    #######################
    #now do the actual plotting with a function
    #######################
    
    def plot_feature_importances(columnToGroupBy : str,
                                 dataFrameToPlotFrom = fullDataFrameComparisonBetweenClassifiers,
                                 topNFeaturesToSelect = topNFeaturesToSelect):
        
        #first add in counts
        countsColumnName = columnToGroupBy + "_counts"
        dataFrameToPlotFrom[countsColumnName] = ""
        
        grouped = dataFrameToPlotFrom.groupby(by=[columnToGroupBy])
        dataFrameForPlottingOrder = pd.DataFrame({
                "count" : grouped["summaryStatValue"].count(),
            "median" : grouped["summaryStatValue"].median()})
        dataFrameForPlottingOrder.sort_values(by = ["count", "median"],
                                              ascending = False, inplace = True)
        
        for featureName, row in dataFrameForPlottingOrder.iterrows():
                dataFrameToPlotFrom.loc[
                        dataFrameToPlotFrom[
                                columnToGroupBy] == featureName,
                                countsColumnName] = str(int(row["count"]))
        
        #okay, now I have counts to group by if I so wish. Make the plot,
        #return axes, figure, and data used to draw this.
        
        
        my_order = dataFrameForPlottingOrder.index
        my_hue_order = [str(entry) for entry in sorted(np.unique(dataFrameForPlottingOrder["count"].reset_index(drop = True).tolist()), reverse = True)]
        
        figCombinedFeatImportances, axCombinedFeatImportances = \
            plt.subplots(1,1, figsize = (10,20))
        
        sb.swarmplot(data = dataFrameToPlotFrom,
                 x = columnToGroupBy,
                 y = "summaryStatValue",
                ax = axCombinedFeatImportances, hue = countsColumnName,
                order = my_order, hue_order = my_hue_order
                )
        axCombinedFeatImportances.set_xticklabels(labels = axCombinedFeatImportances.get_xticklabels(),
                            rotation = 65, ha = "right")
        axCombinedFeatImportances.yaxis.set_tick_params(labelleft = True)
        axCombinedFeatImportances.set_xlabel("Top " + str(topNFeaturesToSelect) + \
                                   " feature importances across classifiers")
        
        axCombinedFeatImportances.set_ylabel("Median feature importances")
        
        #draw median lines
        medianWidth = 0.48
        
        for tick, text in zip(axCombinedFeatImportances.get_xticks(),
                              axCombinedFeatImportances.get_xticklabels()):
            featName = text.get_text()  # "X" or "Y"
        
        # calculate the median value for all replicates of either X or Y
            medianVal = dataFrameToPlotFrom[
                dataFrameToPlotFrom[columnToGroupBy]==featName]["summaryStatValue"].median()
            print(medianVal)
        # plot horizontal lines across the column, centered on the tick
            axCombinedFeatImportances.plot([
                tick-medianWidth/2, tick+medianWidth/2],
                [medianVal, medianVal],
                lw=2, color='k', linestyle = "--")
        
        #return axes, figure, and data used to make it
        
        return [figCombinedFeatImportances, axCombinedFeatImportances,
                dataFrameToPlotFrom, dataFrameForPlottingOrder]
    
    #note that the below do not work in the single anchor case
    leftAnchorRightAnchorChromNameOnly = plot_feature_importances(
            "anchorPositionAndChromMarkName")
    
    leftAnchorRightAnchorFeatNameOnly = plot_feature_importances(
            "anchorPositionAndFeatureName")
    
    leftAnchorRightAnchorChromMarkFeatNameCombo =  plot_feature_importances(
            "anchorPositionAndChromMarkFeatCombo")
    
    
    inBetweenSideOrOutsideWithinAnchorChromNameOnly = plot_feature_importances(
            "withinAnchorSideAndChromMarkName")
    
    inBetweenSideOrOutsideWithinAnchorFeatNameOnly = plot_feature_importances(
            "withinAnchorSideAndFeatureName")
    
    inBetweenSideOrOutsideWithinAnchorChromNameFeatNameCombo = plot_feature_importances(
            "withinAnchorSideAndChromMarkFeatCombo")
    
    #the following do work in the single anchor case
    
    
    chromMarkNameOnly = plot_feature_importances(
            "markNameOnly")
    
    featNameOnly = plot_feature_importances(
            "featureNameOnly")
    
    chromMarkFeatNameCombo = plot_feature_importances(
            "markNameAndFeatureName")
    
    
    #within an anchor, differences between left and right (this is different from
    #the grouping by outside (leftAnchorLeft and rightAnchorRight together AND
    #leftAnchorRight and rightAnchorLeft together)). Here we are seeing whether
    #something different happens with regards to feat importance in the different
    #sections of an anchor, rather than between close to in-between and on the outside (
    #although the two are related.)
    
    withinAnchorDifferenceLeftRightChromMarkNameOnly = plot_feature_importances(
            "anchorSideAndChromMark")
    
    withinAnchorDifferenceLeftRightFeatNameOnly = plot_feature_importances(
            "anchorSideAndFeatName")
    
    withinAnchorDifferenceLeftRightChromNameFeatNameCombo = plot_feature_importances(
            "anchorSideAndChromMarkFeatNameCombo")
    
    
    #make a final dict to output this as
    differentSlicesFeatureImportanceDataPlotsDict = {
            "leftAnchorRightAnchorChromNameOnly" : leftAnchorRightAnchorChromNameOnly,
            "leftAnchorRightAnchorFeatNameOnly"  : leftAnchorRightAnchorFeatNameOnly,
            "leftAnchorRightAnchorChromMarkFeatNameCombo" : leftAnchorRightAnchorChromMarkFeatNameCombo,
            "inBetweenSideOrOutsideWithinAnchorChromNameOnly" : inBetweenSideOrOutsideWithinAnchorChromNameOnly,
            "inBetweenSideOrOutsideWithinAnchorFeatNameOnly" : inBetweenSideOrOutsideWithinAnchorFeatNameOnly,
            "inBetweenSideOrOutsideWithinAnchorChromNameFeatNameCombo" : inBetweenSideOrOutsideWithinAnchorChromNameFeatNameCombo,
            "chromMarkNameOnly" : chromMarkNameOnly,
            "featNameOnly" : featNameOnly,
            "chromMarkFeatNameCombo" : chromMarkFeatNameCombo,
            "withinAnchorDifferenceLeftRightChromMarkNameOnly" : withinAnchorDifferenceLeftRightChromMarkNameOnly,
            "withinAnchorDifferenceLeftRightFeatNameOnly" : withinAnchorDifferenceLeftRightFeatNameOnly,
            "withinAnchorDifferenceLeftRightChromNameFeatNameCombo" : withinAnchorDifferenceLeftRightChromNameFeatNameCombo
            }
    
    #combine all feature importance plots into one dictionary.
    
    featureImportancesSubDictionary = {
            "boxPlotsFeatureImportancePerAnchorPerRep" : featImportanceBoxPlotsPerAnchor,
            "swarmPlotsFeatureImportanceAcrossClassifiersShownPerClassifierWithCompleteOriginalFeatureNamesMedianOnly" : featImportanceBoxPlotsOneForWholeClassifier["medianOnly"],
            "swarmPlotsFeatureImportanceAcrossClassifiersShownPerClassifierWithCompleteOriginalFeatureNamesMeanOnly" : featImportanceBoxPlotsOneForWholeClassifier["meanOnly"],
            "swarmPlotsFeatureImportanceAcrossClassifiersShownPerClassifierWithCompleteOriginalFeatureNamesComparisonMeanAndMedianAcrossValues" : featImportanceBoxPlotsOneForWholeClassifier["compareMeanMedian"]}
    
    for key, value in differentSlicesFeatureImportanceDataPlotsDict.items():
        featureImportancesSubDictionary[key + "FeatureImportancesDifferentSlices"] = value
    
    
    





#If I want to show these plots




#make ROC curves



#fpr, tpr, thresholds = sk.metrics.roc_curve(Y_test, predictedProbaFlip[:,1], pos_label=1)
#fig, ax = plt.subplots()
#lw = 2
#plt.plot(fpr, tpr, color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % sk.metrics.roc_auc_score(Y_test, predictedProbaFlip[:,1]))
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()


#    ROCCurvePerAnchor = {}
#    for anchor, dataFrameROCCurve in ROCcurvePerAnchorDict.items():
#        
#        figROC, axROC = plt.subplots(1,5, figsize = (80,15), sharey = True)
#        
#        
#        summaryStatisticsDFSubset = summaryStatisticsDF[["roc_auc_median", "roc_auc_std", "anchorType"]]
#        uniqueCrossVals = dataFrameROCCurve["crossVal"].unique()
#        colours = sb.color_palette("hls", n_colors = len(uniqueCrossVals))
#        colorDict = dict(zip(uniqueCrossVals, colours))
#        
#        for repNr, groupedByRep in dataFrameROCCurve.groupby("rep"):
#            print("Working on rep: " + str(repNr))
#            for crossValNr, groupedByCrossVal in groupedByRep.groupby("crossVal"):
#                print("working on crossVal: " + str(crossValNr))
#                fpr, tpr = groupedByCrossVal["falsePositiveRate"], groupedByCrossVal["truePositiveRate"]
#                roc_aucMedianThisAnchorThisRep = summaryStatisticsDFSubset[summaryStatisticsDFSubset["anchorType"] == anchor].iloc[repNr-1]["roc_auc_median"]
#                roc_aucStdThisAnchorThisRep    = summaryStatisticsDFSubset[summaryStatisticsDFSubset["anchorType"] == anchor].iloc[repNr-1]["roc_auc_std"]
#                sb.lineplot(fpr, tpr,
#                            ax = axROC[int(repNr) - 1],
#                            hue = crossValNr, palette = colorDict#,
#                            #label='ROC curve (median AUC for CVs = %0.2f ± %0.2f)' % (roc_aucMedianThisAnchorThisRep, roc_aucStdThisAnchorThisRep))
#                            )
#                
#                
#            
#        break
#    
    
    
    #Why is this taking so long relative to the others?
    
    ###
    ###
    ### SINCE AN ROC CURVE PER REP PER ANCHOR TOOK SO LONG, I WON'T MAKE THEM FOR NOW.
    ###
    ###
#    ROCCurvePerAnchor = {}
#    for anchor, dataFrameROCCurve in ROCcurvePerAnchorDict.items():
#        
#        figROC, axROC = plt.subplots(1,5, figsize = (80,15), sharey = True)
#        
#        
#        summaryStatisticsDFSubset = summaryStatisticsDF[["roc_auc_median", "roc_auc_std", "anchorType"]]
#        uniqueCrossVals = dataFrameROCCurve["crossVal"].unique()
#        colours = sb.color_palette("hls", n_colors = len(uniqueCrossVals))
#        colorDict = dict(zip(uniqueCrossVals, colours))
#        
#        for repNr in sorted(dataFrameROCCurve["rep"].unique()):
#            print("Working on rep: " + str(repNr))
#            repSelect = dataFrameROCCurve[dataFrameROCCurve["rep"] == repNr]
#            for crossValNr in sorted(repSelect["crossVal"].unique()):
#                print("working on crossVal: " + str(crossValNr))
#                
#                crossValSelect = repSelect[repSelect["crossVal"] == crossValNr]
#                fpr = crossValSelect["falsePositiveRate"]
#                tpr = crossValSelect["truePositiveRate"]
#                roc_aucMedianThisAnchorThisRep = summaryStatisticsDFSubset[summaryStatisticsDFSubset["anchorType"] == anchor].iloc[repNr-1]["roc_auc_median"]
#                roc_aucStdThisAnchorThisRep    = summaryStatisticsDFSubset[summaryStatisticsDFSubset["anchorType"] == anchor].iloc[repNr-1]["roc_auc_std"]
#                sb.lineplot(fpr, tpr,
#                            ax = axROC[int(repNr) - 1],
#                            hue = crossValNr, palette = colorDict#,
#                            #label='ROC curve (median AUC for CVs = %0.2f ± %0.2f)' % (roc_aucMedianThisAnchorThisRep, roc_aucStdThisAnchorThisRep))
#                            )
#                axROC[int(repNr) - 1].set_xlim([0.0, 1.0])
#                axROC[int(repNr) - 1].set_ylim([0.0, 1.05])
#                axROC[int(repNr) - 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#                
#            
#        ROCCurvePerAnchor[anchor] = [figROC, axROC]
    
    
    
    

    figROCFinal, axROCFinal = plt.subplots(1,3, figsize = (50,15), sharey = True)
    summaryStatisticsDFSubset = summaryStatisticsDF[["roc_auc_median", "roc_auc_std", "anchorType"]]
    
    
    #make different colour for each anchor
#    coloursForAnchors = sb.color_palette(n_colors = len(ROCcurvePerAnchorDict.keys()))
#    coloursToUseByAnchorName = dict(zip(ROCcurvePerAnchorDict.keys(), coloursForAnchors))
    
    #one ROC curve per anchor:
    allAnchorDataTogetherROCOnePanel = None
    for num, (anchor, dataFrameROCCurve) in enumerate(ROCcurvePerAnchorDict.items()):
        dataFrameRepMedianROCData = None
        for repNr, repData in dataFrameROCCurve.groupby("rep"):
            repTPRDF = None
            repFPRDF = None
            for crossValNr, crossValData in repData.groupby("crossVal"):
                TPR, FPR = crossValData["truePositiveRate"], crossValData["falsePositiveRate"]
                repTPRDF = pd.concat([repTPRDF, TPR],
                                     axis = 1)
                repFPRDF = pd.concat([repFPRDF, FPR],
                                     axis = 1)
            #okay, now I have them all in columns. Take the rowwise mean
            medianTPRRep = repTPRDF.median(axis = 1)
            medianFPRRep = repFPRDF.median(axis = 1)
            toAddRepMedianROCData = pd.DataFrame({
                    "medianTPR" : medianTPRRep,
                    "medianFPR" : medianFPRRep,
                    "rep"     : [int(repNr)] * len (medianTPRRep),
                    })
            dataFrameRepMedianROCData = pd.concat([dataFrameRepMedianROCData,
                                                 toAddRepMedianROCData],
            axis = 0)
            
        
        #now, calculate for this anchor the mean ROC curve, and the std.
        #then draw it, with errorbars
        
        groupedByPoint = dataFrameRepMedianROCData.groupby(dataFrameRepMedianROCData.index)
        stdROCDataThisAnchor = groupedByPoint[["medianTPR", "medianFPR"]].std()
        totalROCDataThisAnchor = groupedByPoint[["medianTPR", "medianFPR"]].median()
        totalROCDataThisAnchor = pd.concat([totalROCDataThisAnchor,
                                            stdROCDataThisAnchor],
            axis = 1)
        totalROCDataThisAnchor.columns = ["medianTPR", "medianFPR", "stdTPR", "stdFPR"]
        
        #get ROC_auc:
        
        thisAnchorData = summaryStatisticsDFSubset[summaryStatisticsDFSubset.anchorType == anchor]
        thisAnchorMedianROCAUC = thisAnchorData["roc_auc_median"].median()
        thisAnchorStdROCAUC    = thisAnchorData["roc_auc_median"].std()
        
        sb.lineplot(data = totalROCDataThisAnchor,
                    x = "medianFPR", y = "medianTPR",
                            ax = axROCFinal[num],
                            label="Anchor size: " + str(anchor) + " (median AUC for 5 repeats = %0.2f ± %0.2f)" % \
                            (thisAnchorMedianROCAUC, thisAnchorStdROCAUC))
        axROCFinal[num].set_xlim([0.0, 1.0])
        axROCFinal[num].set_ylim([0.0, 1.05])
        axROCFinal[num].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axROCFinal[num].yaxis.set_tick_params(labelleft = True)
        axROCFinal[num].set_xlabel("Median False Positive Rate (FPR) for anchor size: " + \
                  str(anchor))
        axROCFinal[num].set_ylabel("Median True Positive Rate (TPR)")
        #give errorBars for just 100 values, equally spaced
#        indicesToTake = np.round(np.linspace(0, len(totalROCDataThisAnchor["medianFPR"]),
#                                    num = 50))
##        axROCFinal[num].errorbar(totalROCDataThisAnchor["medianFPR"][indicesToTake],
#                  totalROCDataThisAnchor["medianTPR"][indicesToTake],
#                  yerr = totalROCDataThisAnchor["stdTPR"][indicesToTake] ,
#                  xerr = totalROCDataThisAnchor["stdFPR"][indicesToTake],
#                  capsize = .2, color = "black", linestyle = "--")
        allAnchorDataTogetherROCOnePanel = pd.concat([allAnchorDataTogetherROCOnePanel,
                                pd.concat([totalROCDataThisAnchor, pd.DataFrame(
                                        [anchor] * len(totalROCDataThisAnchor))],
        axis = 1)], axis = 0)
        
        
    #draw all anchors in one plot:
    namesToGive = allAnchorDataTogetherROCOnePanel.columns.values[0:-1].tolist()
    namesToGive.append("anchor")
    allAnchorDataTogetherROCOnePanel.columns = namesToGive
    figROCFinalOnePanel, axROCFinalOnePanel = plt.subplots(1,1, figsize = (10,10))
    eachAnchorROCAUCDict = {}
    for anchor in ROCcurvePerAnchorDict.keys():
        thisAnchorData = summaryStatisticsDFSubset[summaryStatisticsDFSubset.anchorType == anchor]
        thisAnchorMedianROCAUC = thisAnchorData["roc_auc_median"].median()
        thisAnchorStdROCAUC    = thisAnchorData["roc_auc_median"].std()
        eachAnchorROCAUCDict[anchor] = {"medianROCAUC" : thisAnchorMedianROCAUC,
                            "stdROCAUC" : thisAnchorStdROCAUC}
    
    
    sb.lineplot(data = allAnchorDataTogetherROCOnePanel,
                x = "medianFPR", y = "medianTPR", hue = "anchor",
                        ax = axROCFinalOnePanel,
                        palette = colourDictToUseForHueInROCAndMedianScorePlots
                        #label="Anchor size: " + str(anchor) + " (median AUC for 5 repeats = %0.2f ± %0.2f)" % \
                        #(thisAnchorMedianROCAUC, thisAnchorStdROCAUC)
                        )
    axROCFinalOnePanel.set_xlim([0.0, 1.0])
    axROCFinalOnePanel.set_ylim([0.0, 1.05])
    axROCFinalOnePanel.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axROCFinalOnePanel.set_xlabel("Median False Positive Rate (FPR)")
    axROCFinalOnePanel.set_ylabel("Median True Positive Rate (TPR)")
    anchorLabels = [anchor + " " + "(median AUC for 5 repeats = %0.2f ± %0.2f)" % (data["medianROCAUC"], data["stdROCAUC"]) for anchor, data in eachAnchorROCAUCDict.items()]
    anchorLabels.insert(0, "Anchor size used for classifier")
    
    for t, l in zip(axROCFinalOnePanel.legend_.texts, anchorLabels): t.set_text(l)
    figROCFinalOnePanel
    
    
    #make subDictionary per type of plot
    
    finalPlotDictionaryToWriteToDisk = {
        "finalROCPlotMedianOverReps" : [figROCFinalOnePanel, axROCFinalOnePanel],
        "featureImportancePlots" : featureImportancesSubDictionary,
        "scorePlots" : dictPlotsClassifierPerformanceScores,
        "confusionMatrices" : confMatrixDictTotal,
        }
    
    finalDataDictionaryToWriteToDisk = {
            "dataFrameHyperParamValues" : dataFrameHyperParamValues,
            "dataScoresAndPerformances" : featureImportancesPerAnchorDict
            }
    
    totalDictToWrite = {"plots" : finalPlotDictionaryToWriteToDisk,
                        "data" : finalDataDictionaryToWriteToDisk}
    #finally, write all the plots to a .pkl. I will just make one large dictionary.
    with open(outputDir + "plotsAndSummaryDataForFilesFromPath_" + os.path.basename(argv[0][:-1]) + ".pkl",
              "wb") as pickleToWriteTo:
        pickle.dump(totalDictToWrite, pickleToWriteTo)
    
    print("Done making plots. Look for them in file: " + outputDir + "plotsAndSummaryDataForFilesFromPath_" + os.path.basename(argv[0][:-1]) + ".pkl")
    
    
    #Will also just save all plots in subdirectory there, because opening them later is
    #a real pain.
    
    outputJustFigures = outputDir + "plotsInPdfForFilesFromPath_" + os.path.basename(argv[0][:-1]) + "/"
    if os.path.exists(outputJustFigures):
        pass
    else:
        os.mkdir(outputJustFigures)
    
    
    #write all plots to pngs
    figROCFinalOnePanel.savefig(outputJustFigures + "FinalROCPlot.png")
    
    for names, scorePlots in dictPlotsClassifierPerformanceScores.items():
        if names is not "scoresPerRepPerMetric":
            scorePlots[0].savefig(outputJustFigures + names + ".png")
        elif names == "scoresPerRepMetric":
            for n, subPlotList in enumerate(scorePlots):
                subPlotList[0].savefig(outputJustFigures + names + "_" + str(n) +  ".png")
    
    for names, featurePlots in featureImportancesSubDictionary.items():
        if names == 'boxPlotsFeatureImportancePerAnchorPerRep':
            for anchorSize, plotsPerRep in featurePlots.items():
                plotsPerRep[0].savefig(outputJustFigures + names + anchorSize + ".png")
        else:
            featurePlots[0].savefig(outputJustFigures + names + ".png")
    
    for names, confMatrixPlots in finalPlotDictionaryToWriteToDisk["confusionMatrices"].items():
        if names.endswith("PerRep"):
            next
#            for i, plot in enumerate(confMatrixPlots):
#                print(plot[0])
#                print(plot[0][0])
#                plot[0][0].savefig(outputJustFigures + names + "_" + str(i) +  ".png")
#        else:
#            confMatrixPlots[0][0].savefig(outputJustFigures + names + ".png")
            
            #somehow this doesn't work at all. WHY!!?!?
    
    



if __name__ == "__main__":
    main(sys.argv[1:])