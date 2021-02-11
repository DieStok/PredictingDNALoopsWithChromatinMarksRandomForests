#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:21:41 2019

@author: dstoker

Load figure files from classifiers and compare/save them for use in presentation and report.
Also optimise them with the correct labels etc.

Confusion matrices and some other plots could not be saved to .png features when combining
the cross-validation runs into one (see script CommandLineCombinationOfDifferentCrossValRFCLassifiers....py)
#Those can be manually visualised in an interactive python session. Note also that I had to choose
a specific back-end for the figure visualisation that causes all figures to open in separate windows
upon loading them. I therefore close them immediately as they load, but they can all be opened.

#update 26-07-2019:
This also contains code to remake the ROC curve and ROC-median curve for the loop anchor pairs.
I manually redo them by opening the classifiers and the matched feature files.


"""

#%matplotlib notebook
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


pathToReadFrom = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/figuresFinalClassifiers_16_06_2019/"

#specificPickleLoopSizeDistributionAltered

filesInPath = glob.glob(pathToReadFrom + "*")
pklFigFilesInPath = [file for file in filesInPath if file.endswith(".pkl")]

figuresPerClassificationRunDict = {}
for entry in pklFigFilesInPath:
    #nameToGive is just filename minus extension
    nameToGive = os.path.splitext(os.path.basename(entry))[0]
    with open(entry, "rb") as f:
        toAdd = pickle.load(f)
    figuresPerClassificationRunDict[nameToGive] = toAdd
    plt.close("all")
    
    
with open(pklFigFilesInPath[0], "rb") as f:
    k = pickle.load(f)
    
    
    
#%matplotlib qt5







figuresPerClassificationRunDict.keys()
figuresPerClassificationRunDict['plotsAndSummaryDataForFilesFromPath_Uniform'].keys()
figuresPerClassificationRunDict['plotsAndSummaryDataForFilesFromPath_Uniform']["plots"].keys()
figuresPerClassificationRunDict["plotsAndSummaryDataForFilesFromPath_Uniform"]["plots"]["finalROCPlotMedianOverReps"]

figuresPerClassificationRunDict["plotsAndSummaryDataForFilesFromPath_Mimick"]["plots"]["finalROCPlotMedianOverReps"]

allROCPlotsByTreatment = {}
for key in figuresPerClassificationRunDict.keys():
    print(key)
    k = figuresPerClassificationRunDict[key]["plots"]["finalROCPlotMedianOverReps"]
    allROCPlotsByTreatment[key] = k

for key, value in allROCPlotsByTreatment.items():
    print(key)
    value.axes[0].set_title(key)
    value.axes[0].figure
    value.axes[0].figure.show()

figuresPerClassificationRunDict["plotsAndSummaryDataForFilesFromPath_Uniform"]["plots"].keys()
figuresPerClassificationRunDict["plotsAndSummaryDataForFilesFromPath_Uniform"]["plots"]["featureImportancePlots"].keys()
figuresPerClassificationRunDict["plotsAndSummaryDataForFilesFromPath_Uniform"]["plots"]["scorePlots"].keys()
figuresPerClassificationRunDict["plotsAndSummaryDataForFilesFromPath_Uniform"]["plots"]["scorePlots"]["medianScoresCombinedPerAnchor"][0].figure


figuresPerClassificationRunDict.keys()
figuresPerClassificationRunDict['plotsAndSummaryDataForFilesFromPath_dataWithIntersect_NewClassification_29_05_2019'][
        "plots"]["confusionMatrices"]["2500normConfMatrixPerAnchor"][0]


specificLoopFilePath = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/figuresFinalClassifiers_16_06_2019/plotsAndSummaryDataForFilesFromPath_OutputClassificationFakeVSRealLoops.pkl"

with open(specificLoopFilePath, "rb") as f:
    fakeLoopsWithoutInbetweenForSomeReason = pickle.load(f)
    
GetPerformance = fakeLoopsWithoutInbetweenForSomeReason["plots"]["confusionMatrices"]["allSizesTogethernormConfMatrixPerAnchor"]
GetPerformance
GetPerformance[0]











################################################################################
################################################################################
#redo the ROC curve for the fake vs real loop without in-between on 11-07-2019
#first test on one file, then do this for all of them
################################################################################
################################################################################

filePathToOpenROCCurve = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/figuresFinalClassifiers_16_06_2019/RemakeROCCurveFakeVsRealLoops_IndividualClassifierFiles_11_07_2019/RandomForestClassification_AllAnchorsTogether_Data_rep_1_of5_fold_1_of10_FromInputFile_finalFeatureDictionary_10_06_2019_FakeLoops_NoInBetween_2019-06-17.pkl"

with open(filePathToOpenROCCurve, "rb") as f:
    dataForWhichToRectifyROCCurve = pickle.load(f)

fullClassifier = dataForWhichToRectifyROCCurve["data"]["fullClassifier"]
testIndices = dataForWhichToRectifyROCCurve["data"]["testedOn"]

pathFeatureDictUsedForFakeVSRealLoopPredictWithoutInBetween = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/figuresFinalClassifiers_16_06_2019/RemakeROCCurveFakeVsRealLoops_IndividualClassifierFiles_11_07_2019/finalFeatureDictionary_10_06_2019_FakeLoops_NoInBetween.pkl"
with open(pathFeatureDictUsedForFakeVSRealLoopPredictWithoutInBetween, "rb") as f:
    featureDictForThesePredictions = pickle.load(f)

#edit this code:
numericLabels = sk.preprocessing.LabelBinarizer()
#input to labelBinarizer should be reshaped
reshapedLabels = featureDictForThesePredictions["classLabelArray"].reshape(-1, 1)
numericLabels.fit(reshapedLabels)
transformedClassLabels = numericLabels.transform(reshapedLabels)
transformedClassLabels = np.ravel(transformedClassLabels)    



featuresNotNormalised = featureDictForThesePredictions["featureArray"]
columnsNotPValue = [k for k in featureDictForThesePredictions["namesFeatureArrayColumns"] if not k.endswith("avgPValueNegativeLog10") and not k.endswith("medPValueNegativeLog10")]
indicesFeatureArrayToTake = [featureDictForThesePredictions["namesFeatureArrayColumns"].tolist().index(k) for k in featureDictForThesePredictions["namesFeatureArrayColumns"] if \
                                 k in columnsNotPValue]

featuresNotNormalised = featureDictForThesePredictions["featureArray"][:, indicesFeatureArrayToTake]


prediction                   = fullClassifier.predict(featuresNotNormalised[testIndices])
#allPredictions.append(prediction)
predictionProbs              = fullClassifier.predict_proba(featuresNotNormalised[testIndices])
#allPredictionsProbs.append(predictionProbs)
predictionProbsPositiveClass = predictionProbs[:, 1]

#changed the below to 0, was 1. Think this fixes the curve.
falsePositiveRate, truePositiveRate, thresholds = sk.metrics.roc_curve(transformedClassLabels[testIndices],
                                                predictionProbsPositiveClass,
                                                pos_label=0)

currentRep = dataForWhichToRectifyROCCurve["rep"]
currentCrossVal = dataForWhichToRectifyROCCurve["crossValFold"]

dictForDFToMake = {"rep" : currentRep ,
                   "crossVal" : currentCrossVal ,
                   "FPR" : falsePositiveRate,
                   "TPR" : truePositiveRate}

dataFrameToConcat = pd.DataFrame(dictForDFToMake)


#testCurve
figROCFinal, axROCFinal = plt.subplots(1,3, figsize = (50,15), sharey = True)
sb.lineplot(data = dataFrameToConcat,
                    x = "FPR", y = "TPR",
                    ax = axROCFinal[0])
                            
axROCFinal[0].set_xlim([0.0, 1.0])
axROCFinal[0].set_ylim([0.0, 1.05])
axROCFinal[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')


###Works for one file. Below does it for all and combines

filePathWithClassifierFiles = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/figuresFinalClassifiers_16_06_2019/RemakeROCCurveFakeVsRealLoops_IndividualClassifierFiles_11_07_2019/"

listFilesToOpen = glob.glob(filePathWithClassifierFiles + "RandomForestClassification*.pkl")

pathFeatureDictUsedForFakeVSRealLoopPredictWithoutInBetween = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/figuresFinalClassifiers_16_06_2019/RemakeROCCurveFakeVsRealLoops_IndividualClassifierFiles_11_07_2019/finalFeatureDictionary_10_06_2019_FakeLoops_NoInBetween.pkl"
with open(pathFeatureDictUsedForFakeVSRealLoopPredictWithoutInBetween, "rb") as f:
    featureDictForThesePredictions = pickle.load(f)

totalROCRedoDataFrame = None
rocAucDict = {}
confMatrixDictNormed = {}
confMatrixDict = {}

totalROCRedoDataFrameAlternative = None

for entry in listFilesToOpen:
    
    
    with open(entry, "rb") as f:
        dataForWhichToRectifyROCCurve = pickle.load(f)
    
    fullClassifier = dataForWhichToRectifyROCCurve["data"]["fullClassifier"]
    testIndices = dataForWhichToRectifyROCCurve["data"]["testedOn"]
    

    
    #edit this code:
    numericLabels = sk.preprocessing.LabelBinarizer()
    #input to labelBinarizer should be reshaped
    reshapedLabels = featureDictForThesePredictions["classLabelArray"].reshape(-1, 1)
    numericLabels.fit(reshapedLabels)
    transformedClassLabels = numericLabels.transform(reshapedLabels)
    transformedClassLabels = np.ravel(transformedClassLabels)    
    
    
    
    featuresNotNormalised = featureDictForThesePredictions["featureArray"]
    columnsNotPValue = [k for k in featureDictForThesePredictions["namesFeatureArrayColumns"] if not k.endswith("avgPValueNegativeLog10") and not k.endswith("medPValueNegativeLog10")]
    indicesFeatureArrayToTake = [featureDictForThesePredictions["namesFeatureArrayColumns"].tolist().index(k) for k in featureDictForThesePredictions["namesFeatureArrayColumns"] if \
                                     k in columnsNotPValue]
    
    featuresNotNormalised = featureDictForThesePredictions["featureArray"][:, indicesFeatureArrayToTake]
    
    
    prediction                   = fullClassifier.predict(featuresNotNormalised[testIndices])
    #allPredictions.append(prediction)
    predictionProbs              = fullClassifier.predict_proba(featuresNotNormalised[testIndices])
    #allPredictionsProbs.append(predictionProbs)
    predictionProbsPositiveClass = predictionProbs[:, 1]
    
    #make confidence matrices
    confMatrix       = sk.metrics.confusion_matrix(transformedClassLabels[testIndices],
                                             prediction)
    confMatrixNormed = confMatrix.astype('float') / confMatrix.sum(axis=1)[:, np.newaxis]
    
    
    
    
    #changed the below to 0, was 1. Think this fixes the curve.
    falsePositiveRate, truePositiveRate, thresholds = sk.metrics.roc_curve(transformedClassLabels[testIndices],
                                                    predictionProbs[:,1],
                                                    pos_label=0)
    falsePositiveRateOtherWay, truePositiveRateOtherWay, thresholdsOtherWay = sk.metrics.roc_curve(transformedClassLabels[testIndices],
                                                    predictionProbs[:,1],
                                                    pos_label=1)
    
    currentRep = dataForWhichToRectifyROCCurve["rep"]
    currentCrossVal = dataForWhichToRectifyROCCurve["crossValFold"]
    
    roc_auc = sk.metrics.roc_auc_score(transformedClassLabels[testIndices],
                                       predictionProbs[:,0])
    
    dictForDFToMake = {"rep" : currentRep ,
                       "crossVal" : currentCrossVal,
                       "FPR" : falsePositiveRate,
                       "TPR" : truePositiveRate
                       }
    
    if currentRep in rocAucDict.keys():
        rocAucDict[currentRep].append(roc_auc)
    else:
        rocAucDict[currentRep] = [roc_auc]
    
    if currentRep in confMatrixDict.keys():
        confMatrixDict[currentRep].append(confMatrix)
    else:
        confMatrixDict[currentRep] = [confMatrix]
    
    if currentRep in confMatrixDictNormed.keys():
        confMatrixDictNormed[currentRep].append(confMatrixNormed)
    else:
        confMatrixDictNormed[currentRep] = [confMatrixNormed]
    
    
    
    
    dataFrameToConcat = pd.DataFrame(dictForDFToMake)
    dataFrameToConcatAlternative = pd.DataFrame({
            "rep" : currentRep ,
            "crossVal" : currentCrossVal,
            "FPROtherWay" : falsePositiveRateOtherWay,
            "TPROtherWay" : truePositiveRateOtherWay})
    totalROCRedoDataFrame = pd.concat([totalROCRedoDataFrame, dataFrameToConcat])
    totalROCRedoDataFrameAlternative = pd.concat([totalROCRedoDataFrameAlternative,
                                                 dataFrameToConcatAlternative])

#save that redo dataframe so I don't have to recalc it everytime
now = str(datetime.datetime.today()).split()[0]
totalROCRedoDataFrame.to_csv(filePathWithClassifierFiles + "combinedDataForROCCurve_" + now + ".csv")

#group by rep, take the medians. Then take the medians of that, and plot

#yeah I did this in a strange way, I need a different orientation for this. Let's make that. Get all the FPR and TPR per
#rep into a dataframe and then 

TPRAllReps = None
alternativeTPRAllReps = None
FPRAllReps = None
alternativeFPRAllReps = None
for rep in np.unique(totalROCRedoDataFrame["rep"]):
    currentDF = totalROCRedoDataFrame[totalROCRedoDataFrame["rep"] == rep]
    currentDFAlternative = totalROCRedoDataFrameAlternative[totalROCRedoDataFrameAlternative["rep"] == rep]
    dfFPRForMedianThisRep = None
    dfTPRForMedianThisRep = None
    alternativeFPRMedianThisRep = None
    alternativeTPRMedianThisRep = None
    for crossVal in np.unique(currentDF["crossVal"]):
        subsetCurrentDF = currentDF[currentDF["crossVal"] == crossVal]
        subsetCurrentDFAlternative = currentDFAlternative[currentDFAlternative["crossVal"] == crossVal]
        dfFPRForMedianThisRep = pd.concat([dfFPRForMedianThisRep, subsetCurrentDF["FPR"]], axis = 1)
        dfTPRForMedianThisRep = pd.concat([dfTPRForMedianThisRep, subsetCurrentDF["TPR"]], axis = 1)
        alternativeFPRMedianThisRep = pd.concat([alternativeFPRMedianThisRep, subsetCurrentDFAlternative["FPROtherWay"]], axis = 1)
        alternativeTPRMedianThisRep = pd.concat([alternativeTPRMedianThisRep, subsetCurrentDFAlternative["TPROtherWay"]], axis = 1)
    #now with the final dataframe, I can get a median TPR and FPR
    medianFPRThisRep = dfFPRForMedianThisRep.median(axis = 1)
    medianTPRThisRep = dfTPRForMedianThisRep.median(axis = 1)
    TPRAllReps = pd.concat([TPRAllReps, medianTPRThisRep], axis = 1)
    FPRAllReps = pd.concat([FPRAllReps, medianFPRThisRep], axis = 1)
    alternativeTPRAllReps = pd.concat([alternativeTPRAllReps,
                                       alternativeTPRMedianThisRep], axis = 1)
    alternativeFPRAllReps = pd.concat([alternativeFPRAllReps,
                                       alternativeFPRMedianThisRep], axis = 1)


#now take the median of these
    finalMedianTPR = TPRAllReps.median(axis = 1)
    finalMedianFPR = FPRAllReps.median(axis = 1)
    
    alternativeFinalMedianTPR = alternativeTPRAllReps.median(axis = 1)
    alternativeFinalMedianFPR = alternativeFPRAllReps.median(axis = 1)
dfForCorrectROC = pd.DataFrame({"TPR" : finalMedianTPR,
                                "FPR" : finalMedianFPR})

dfForCorrectROCALternative = pd.DataFrame({"alternativeTPR" : alternativeFinalMedianTPR,
                                           "alternativeFPR" : alternativeFinalMedianFPR})

#testCurve
figROCFinalOnePanel, axROCFinalOnePanel = plt.subplots(1,1, figsize = (10,10))
sb.lineplot(data = dfForCorrectROC,
                    x = "FPR", y = "TPR",
                    ax = axROCFinalOnePanel)


                            
axROCFinalOnePanel.set_xlim([0.0, 1.0])
axROCFinalOnePanel.set_ylim([0.0, 1.05])
axROCFinalOnePanel.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axROCFinalOnePanel.set_xlabel("Median False Positive Rate (FPR)")
axROCFinalOnePanel.set_ylabel("Median True Positive Rate (TPR)")

figROCFinalOnePanelAlt, axROCFinalOnePanelAlt = plt.subplots(1,1, figsize = (10,10))
sb.lineplot(data = dfForCorrectROCALternative,
                    x = "alternativeFPR", y = "alternativeTPR",
                    ax = axROCFinalOnePanelAlt)
axROCFinalOnePanelAlt.set_xlim([0.0, 1.0])
axROCFinalOnePanelAlt.set_ylim([0.0, 1.05])
axROCFinalOnePanelAlt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axROCFinalOnePanelAlt.set_xlabel("Median False Positive Rate (FPR)")
axROCFinalOnePanelAlt.set_ylabel("Median True Positive Rate (TPR)")

#plots
figROCFinalOnePanel
figROCFinalOnePanelAlt

#get confidence matrices with std

def output_confidence_matrix_with_std(dictConfMatrices: dict):
    """Takes in the confidence matrices per cross-val. Returns one
    median value +- std (calculated over the median per rep)
    (just to be absolutely clear: first you take the median per rep (i.e. over crossfolds),
     and then the median and std over these median values is reported)"""
     
#    outputMatrix = np.array([[0,0],
#                    [0,0]])
#    outputMatrixStd = np.array([[0,0],
#                    [0,0]])
    
    
    totalTopLeft = []
    totalTopRight = []
    totalBottomLeft = []
    totalBottomRight = []
    
    for rep, listOfMatrices in dictConfMatrices.items():
        
        repTopLeft = []
        repTopRight = []
        repBottomLeft = []
        repBottomRight = []
        for matrix in listOfMatrices:
            topLeft     = matrix[0,0]
            topRight    = matrix[0,1]
            bottomLeft  = matrix[1,0]
            bottomRight = matrix[1,1]
            repTopLeft.append(topLeft)
            repTopRight.append(topRight)
            repBottomLeft.append(bottomLeft)
            repBottomRight.append(bottomRight)
            
        repTopLeft = np.array(repTopLeft)
        repTopRight = np.array(repTopRight)
        repBottomLeft = np.array(repBottomLeft)
        repBottomRight = np.array(repBottomRight)
        
        medTL = np.median(repTopLeft)
        medTR = np.median(repTopRight)
        medBL = np.median(repBottomLeft)
        medBR = np.median(repBottomRight)
        
        totalTopLeft.append(medTL)
        totalTopRight.append(medTR)
        totalBottomLeft.append(medBL)
        totalBottomRight.append(medBR)
        
    
    outputMatrix = np.array([[np.median(np.array(totalTopLeft)),np.median(np.array(totalTopRight))],
                             [np.median(np.array(totalBottomLeft)), np.median(np.array(totalBottomRight))]])
    
    outputMatrixStd = np.array([[np.std(np.array(totalTopLeft)),np.std(np.array(totalTopRight))],
                                 [np.std(np.array(totalBottomLeft)),np.std(np.array(totalBottomRight))]])
    
    
    return (outputMatrix, outputMatrixStd)
    


medianNormConfMatrix, medianNormConfMatrixStd = output_confidence_matrix_with_std(confMatrixDictNormed)

medianConfMatrix, medianConfMatrixStd = output_confidence_matrix_with_std(confMatrixDict)


#now make an  of confusion matrix


cmap = plt.cm.Blues
fig, ax = plt.subplots()
#fig.suptitle("Anchorsize: " + name, y = 0.98 + 0.075, ha='center', va='bottom')
plt.grid(False)
im = ax.imshow(medianNormConfMatrix, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)
classes = ["True", "False"]
title = "Normalised Confusion Matrix"
# We want to show all ticks...
ax.set(xticks=np.arange(medianNormConfMatrix.shape[1]),
       yticks=np.arange(medianNormConfMatrix.shape[0]),
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
thresh = medianNormConfMatrix.max() / 6 * 4.
print(thresh)
for i in range(medianNormConfMatrix.shape[0]):
    for j in range(medianNormConfMatrix.shape[1]):
        ax.text(j, i, format(medianNormConfMatrix[i, j], fmt) + " ± " + format(medianNormConfMatrixStd[i, j], fmt) + "\n(median ± std.)",
                ha="center", va="center",
                color="white" if medianNormConfMatrix[i, j] > thresh else "black")
fig.tight_layout()

fig

#not normalised:

cmap = plt.cm.Blues
fig2, ax2 = plt.subplots()
#fig.suptitle("Anchorsize: " + name, y = 0.98 + 0.075, ha='center', va='bottom')
plt.grid(False)
im = ax2.imshow(medianConfMatrix, interpolation='nearest', cmap=cmap)
ax2.figure.colorbar(im, ax=ax2)
classes = ["True", "False"]
title = "Confusion Matrix"
# We want to show all ticks...
ax2.set(xticks=np.arange(medianConfMatrix.shape[1]),
       yticks=np.arange(medianConfMatrix.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=classes, yticklabels=classes,
       title=title,
       ylabel='True label',
       xlabel='Predicted label')
ax2.title.set_fontsize(8)

# Rotate the tick labels and set their alignment.
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
fmt = '.2f' 
thresh = medianConfMatrix.max() / 6 * 4.
print(thresh)
for i in range(medianConfMatrix.shape[0]):
    for j in range(medianConfMatrix.shape[1]):
        ax2.text(j, i, format(medianConfMatrix[i, j], fmt) + " ± " + format(medianConfMatrixStd[i, j], fmt) + "\n(median ± std.)",
                ha="center", va="center",
                color="white" if medianConfMatrix[i, j] > thresh else "black")
fig2.tight_layout()
fig2

#make an roc_auc median figure
rocAucList = []
for rep, data in rocAucDict.items():
    rocAucList.append(np.median(np.array(data)))

medianRocAucPlotDF = pd.DataFrame({"valueName" : "roc_auc_median",
              "valueScore" : rocAucList})

current_palette = sb.color_palette()
sb.palplot(current_palette)

figRocAuc, axRocAuc = plt.subplots(1,1)

plotThisScoreValue = sb.stripplot(data = medianRocAucPlotDF, x = "valueName",
                      y = "valueScore", dodge = True,
                      jitter = 0.03, ax = axRocAuc)

plotThisScoreValue.axes.set_ylabel("Score (median over 10 folds for 5 repeats)")
plotThisScoreValue.axes.set_xlabel("")
        

#Get a mimick confusion matrix plot --> nevermind, seems it is wrong as well.
figuresPerClassificationRunDict['plotsAndSummaryDataForFilesFromPath_Mimick']["plots"]["confusionMatrices"]["10000normConfMatrixPerAnchor"][0]


#get some hyperparameter tables
figuresPerClassificationRunDict['plotsAndSummaryDataForFilesFromPath_Mimick']["data"]["dataFrameHyperParamValues"]
#never mind, they are not in a state that is fit for inclusion really.
