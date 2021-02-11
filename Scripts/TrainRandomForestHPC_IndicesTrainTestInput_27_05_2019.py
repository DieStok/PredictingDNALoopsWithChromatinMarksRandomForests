#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:58:20 2019

@author: dstoker


#update 27-07-2019
This script does the brunt of the work on the HPC. For a specific cross-validation,
it takes the train and test sets, and performs random forest classification.
It also calculates some metrics of success, such as the ROC AUC and makes an ROC curve.
The latter point is a bit of a problem, since those functions require it to be clear
what is the positive, and what is the negative class. However, I automatically binarize
labels, so if I have as text labels "GM" and "Heart" it is purely dependant on what is first
in the list which is 0 and which is 1. This is unfortunate for read-out of confusion matrices.
Be aware. This should be changed.

Training of Random Forest on the HPC
From a wrapper script,
gets as the first argument:
    train_test data .pkl

gets as the 2nd argument:
    a string that details which repeat and which fold it is (e.g. "rep_1_of5_fold_5_of10")
    This syntax is important: rep number and fold number should be surrounded by _, because
    these numbers are used in the output dictionary pickle
    
gets as the 3rd argument:
    the featureData.pkl to use (i.e. the inputFilePath)
    
gets as 4th argument:
    the outputFileDir, where the metrics (best hyperparams, performance, etc.) can
    be saved
    
gets as 5th (optional) argument:
    the size of the anchors to select: 2500 or 10000 or ....
    

Goal of the script:
    -does internal cross-validation to determine best hyperparameters.
    -gets best hyperparams and performance metrics from this estimator
    -saves these, plus the indices of train and test samples, to a file

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


def main(argv):
    print(argv)
    if len(argv) == 4:
        selectOneAnchorSize = False
    else:
        if len(argv) == 5:
            print(str(len(argv)))
            try:
                anchorSizeToSelect = int(argv[4])
            except:
                print("Anchor size given is not a number. Instead, got: " + argv[4] + ".")
                sys.exit("Anchor size given is not a number. Instead, got: " + argv[4] + "/n" + \
                         "Please make sure a correct anchor size is selected.")
            selectOneAnchorSize = True
            print("Training only on features calculated with a size of " + str(anchorSizeToSelect) + " base pairs.")
        else:
            
            sys.exit("""Arguments are supplied incorrectly. 
                     First argument should be a list of two arrays, like so: 
                     [np.array2string(trainIndices), np.array2string(testIndices)] 
                     Second argument: a string that details rep and crossVal, like so: 
                     "rep1_of5_fold5_of10" 
                     Third argument: input feature file path (.pkl) 
                     Fourth argument: outputFileDir (/bar/foo/yonta) to save files in. 
                     Fifth (optional) argument: anchor size to select (e.g. 2500, 10000). Must be in data.
            """)
    if os.path.isfile(argv[0]):
        print("Reading train_test_split .pkl")
        with open(argv[0], "rb") as f:
            trainTestIndices = pickle.load(f)
    else:
        sys.exit("train_test_split .pkl path provided is not a file!")
    
    repAndCrossValDesignation = argv[1]
    print("Repeat and CrossVal num: " + repAndCrossValDesignation)
    rep      = int(repAndCrossValDesignation.split("_")[1])
    crossVal = int(repAndCrossValDesignation.split("_")[4])
    #correct rep and crossVal with -1 to Python indices

    trainIndices = trainTestIndices[rep-1][crossVal-1][0]
    testIndices  = trainTestIndices[rep-1][crossVal-1][1]
    
    if os.path.isfile(argv[2]):
        featureDataPicklePath = argv[2]
    else:
        sys.exit("Wrong feature pickle path provided. This path does not exist or is not a file.")
    
    if os.path.isdir(argv[3]):
        outputFileDir = argv[3]
    else:
        try: 
            os.makedirs(argv[3])
        except:
            sys.exit("Supplied output path does not match an existing directory and directory could not be made.")
        outputFileDir = argv[3]
    
    
    #read in data, train
    
    with open(featureDataPicklePath, "rb") as f:
        featureDict = pickle.load(f)
    
    #rename and remove p-value columns which are useless
    featuresNotNormalised = featureDict["featureArray"]
    columnsNotPValue = [k for k in featureDict["namesFeatureArrayColumns"] if not k.endswith("avgPValueNegativeLog10") and not k.endswith("medPValueNegativeLog10")]
    indicesFeatureArrayToTake = [featureDict["namesFeatureArrayColumns"].tolist().index(k) for k in featureDict["namesFeatureArrayColumns"] if \
                                 k in columnsNotPValue]
    
    featuresNotNormalised = featureDict["featureArray"][:, indicesFeatureArrayToTake]
    
    #allow selection of anchors of a specific size
    featureNames = featureDict["namesFeatureArrayColumns"][indicesFeatureArrayToTake]
    
    #check whether features of the selected anchor size are present (if optionalArgument was given)
    #select them if so.
    if (selectOneAnchorSize):
        anchorSizesFeatures = np.array([int(name.split("_")[1]) for name in featureNames])
        print(anchorSizesFeatures)
        print(type(anchorSizesFeatures))
        print(anchorSizeToSelect)
        print(type(anchorSizeToSelect))
        if np.sum(anchorSizesFeatures == np.array(anchorSizeToSelect)) > 0:
            selector = anchorSizesFeatures == np.array(anchorSizeToSelect)
            featuresNotNormalised = featuresNotNormalised[:, selector] 
        else:
            sys.exit("The selected anchor size is not present in the feature dictionary provided! /n The sizes present are: " + str(list(np.unique(anchorSizesFeatures))))
            
    

    #get labels and reshape etc.
    numericLabels = sk.preprocessing.LabelBinarizer()
    #input to labelBinarizer should be reshaped
    reshapedLabels = featureDict["classLabelArray"].reshape(-1, 1)
    numericLabels.fit(reshapedLabels)
    transformedClassLabels = numericLabels.transform(reshapedLabels)
    transformedClassLabels = np.ravel(transformedClassLabels)    
     
    param_dist = {
              "n_estimators"     : [50, 150, 300, 600],
              "max_depth"        : [1, 10, 20, 60, 100, None],
              "max_features"     : ["sqrt", "log2", None],
              "min_samples_split": [2, 5, 10, 25, 75],
              "min_samples_leaf" : [1, 5, 10]}
    
    
    #function to report what the classifier did:
    #shameless copy from sklearn website
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
    
    
    classifier = RandomForestClassifier(random_state = 42, verbose = 1, n_jobs = -1)
    randomSearch = sk.model_selection.RandomizedSearchCV(classifier, param_dist,
                                          n_iter = 3,
                                          scoring = ["precision", "recall", "roc_auc"],
                                          cv = 5,
                                          iid = False,
                                          refit = "roc_auc",
                                          n_jobs = -1)
    
    randomSearch.fit(featuresNotNormalised[trainIndices],
                     transformedClassLabels[trainIndices])
    report(randomSearch.cv_results_)
    
    
    bestRF     = randomSearch.best_estimator_
    bestParams = randomSearch.best_params_
    
    prediction                   = bestRF.predict(featuresNotNormalised[testIndices])
    #allPredictions.append(prediction)
    predictionProbs              = bestRF.predict_proba(featuresNotNormalised[testIndices])
    #allPredictionsProbs.append(predictionProbs)
    predictionProbsPositiveClass = predictionProbs[:, 1]
    
    roc_auc = sk.metrics.roc_auc_score(transformedClassLabels[testIndices],
                                       predictionProbsPositiveClass)
    
    confMatrix       = sk.metrics.confusion_matrix(transformedClassLabels[testIndices],
                                             prediction)
    confMatrixNormed = confMatrix.astype('float') / confMatrix.sum(axis=1)[:, np.newaxis]
    
    falsePositiveRate, truePositiveRate, thresholds = sk.metrics.roc_curve(transformedClassLabels[testIndices],
                                                predictionProbsPositiveClass,
                                                pos_label=1)
    
    precision = sk.metrics.precision_score(transformedClassLabels[testIndices],
                                           prediction)
    recall    = sk.metrics.recall_score(transformedClassLabels[testIndices],
                                           prediction)
    
    
    #for reading in later, I want to combine all 10 crossVals of a repeat.
    #therefore, I should save a dict that has:
    #{rep : num,
    # crossVal: num,
    # data: {params : list, ROCauc : num}} etc.
    
    bestPerformingRFData = {"roc_auc"           : roc_auc,
                            "falsePositiveRate" : falsePositiveRate,
                            "truePositiveRate"  : truePositiveRate,
                            "confMatrix"        : confMatrix,
                            "confMatrixNormed"  : confMatrixNormed,
                            "precision"         : precision,
                            "recall"            : recall,
                            "trainedOn"         : trainIndices,
                            "testedOn"          : testIndices,
                            "bestParams"        : bestParams,
                            "bestFeatImport"    : bestRF.feature_importances_,
                            "fullClassifier"    : bestRF}
    
    
    dictOutput = {"rep"          : repAndCrossValDesignation.split("_")[1],
                  "crossValFold" : repAndCrossValDesignation.split("_")[4],
                  "data"         : bestPerformingRFData}
    
    
    #save this
    if outputFileDir.endswith("/"):
        pass
    else:
        outputFileDir += "/"
    now = str(datetime.datetime.today()).split()[0]
    if (selectOneAnchorSize):
        fullNameOutputFile = outputFileDir + \
                        "RandomForestClassification_AnchorSize_" + str(anchorSizeToSelect) + "_Data_" + repAndCrossValDesignation + "_" + \
                        "FromInputFile_" + os.path.splitext(os.path.basename(featureDataPicklePath))[0] + \
                        "_" + now + ".pkl"
    else:
            
        fullNameOutputFile = outputFileDir + \
                            "RandomForestClassification_AllAnchorsTogether_Data_" + repAndCrossValDesignation + "_" + \
                            "FromInputFile_" + os.path.splitext(os.path.basename(featureDataPicklePath))[0] + \
                            "_" + now + ".pkl"
    with open(fullNameOutputFile, "wb") as f:
        pickle.dump(dictOutput, f)
    print("Done training. Output in: " + fullNameOutputFile + " .")
               











#ignore this, this is how the wrapper script will send the indices (well, without json.loads, that is done in this script)
#pizza = json.loads("[" + np.array2string(generatedIndices[0][0][0], max_line_width= np.inf, separator = ",") + "," + np.array2string(generatedIndices[0][0][1], max_line_width= np.inf, separator = ",") + "]")










if __name__ == "__main__":
    main(sys.argv[1:])

