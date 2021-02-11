#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:21:09 2019

@author: dstoker

Wrapper script that reads in a file containing train-test splits for a number
of repeats and a number of cross-validations, and then starts a script
that trains random forest classifiers on these subsets of the data.

#arguments

1: input .pkl file containing the calculated features, their names, class labels, and loopID
2: inpute .pkl file containing the train-test splits to give to instances of the Random Forest Training script
3: output file directory, where Random Forest Training scripts can deposit their outputs
4: .csv file that specifies which reps and crossVals to rerun for which anchors (see 
        ScriptToOutputAListOfMissingRepeatsAndCrossValsToRerun.py).
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
    
    if len(argv) == 4:
        pass
    else:
        sys.exit("Insufficient arguments given./nThis programme needs at least a feature dictionary .pkl (arg1) /n a train-test .pkl /n an output folder. /n and a .csv with reps, crossVals and anchor sizes for which to rerun separate RF classifiers.")
    
    if os.path.isfile(argv[0]):
        inputFeatureDictPath = argv[0]
    else:
        sys.exit("Input feature dictionary .pkl file is not a correct path!")
    if os.path.isfile(argv[1]):
        trainTestSplitFilePath = argv[1]
        print("Reading train-test split file.")
        with open(trainTestSplitFilePath, "rb") as f:
            trainTestSplitInfo = pickle.load(f)
    else:
        sys.exit("train-test split file is not a correct path to a file")
    
    if os.path.isdir(argv[2]):
        outputFileDir = argv[2]
    else:
        try: 
            os.makedirs(argv[2])
        except:
            sys.exit("Supplied output path does not match an existing directory and directory could not be made.")
        outputFileDir = argv[2]
        
    if outputFileDir.endswith("/"):
        pass
    else:
        outputFileDir += "/"
        
    
    #read in the reps and crossVals to rerun
    dataFrameRepsCrossValsAndAnchorsToRunFor = pd.read_csv(argv[3])
    
    anchorSizesToSelect = np.unique(dataFrameRepsCrossValsAndAnchorsToRunFor["anchor"])
    originalNames = anchorSizesToSelect.copy()
    #I coded this such that a script started without additional arguments would calculate for
    #all features together. Hence, allAnchorsTogether should be made into an empty string for
    #jobs to be started correctly.
    if "AllAnchorsTogether" in anchorSizesToSelect:
        print("yes")
        
        place = np.where([entry == "AllAnchorsTogether" for entry in anchorSizesToSelect])
        for i in place[0]:
            anchorSizesToSelect[i] = anchorSizesToSelect[i].replace("AllAnchorsTogether", "") 
        
    
    
    #create a log folder if it doesn't exist
    if os.path.isdir(outputFileDir + "logs/"):
        pass
    else:
        os.makedirs(outputFileDir + "logs/")
    logDir = outputFileDir + "logs/"
    
    for repData in trainTestSplitInfo:
                totalReps = str(len(repData))
    
    
    inputFileNameToAppendToJobName = os.path.basename(os.path.splitext(inputFeatureDictPath)[0])
    #load the test-train file, loop over anchorSizesToSelect, reps and inputs
    print("Starting HPC jobs")
    print("-----------------")
    for anchorToSubset, anchorToAppend in zip(originalNames,anchorSizesToSelect):
        #print(anchorToCalcFor)
        for index, row in \
        dataFrameRepsCrossValsAndAnchorsToRunFor[
                dataFrameRepsCrossValsAndAnchorsToRunFor.anchor == anchorToSubset].iterrows():
            repToRun = row["rep"]
            foldToRun = row["fold"]
            
            
            
            repCrossValInfoToParse        = "rep_" + str(repToRun) + "_of" +\
                                            str(len(trainTestSplitInfo)) + "_fold_" + \
                                            str(foldToRun) + "_of" + \
                                            totalReps
            
            
            #start the jobs
            
            print("RERUNNING job for data of: ")
            print(repCrossValInfoToParse)
            os.system("qsub -cwd -N DieterRFCalc_" + inputFileNameToAppendToJobName + \
             "_" + anchorToAppend + " " +
             "-o " + logDir + repCrossValInfoToParse + "_" + anchorToSubset + \
             "_log_out.txt " + "-l h_vmem=70G " + "-l h_rt=60:00:00 " + "-pe threaded 4 " + \
             "/hpc/cog_bioinf/ridder/users/dstoker/scripts/run_script.sh " + "python " + \
             "/hpc/cog_bioinf/ridder/users/dstoker/scripts/TrainRandomForestHPC_IndicesTrainTestInput_27_05_2019.py " + \
             trainTestSplitFilePath + " " + repCrossValInfoToParse + " " + \
             inputFeatureDictPath + " " + outputFileDir + " " + anchorToAppend)
            
    print("--------------")
    print("All jobs started. Please check " + logDir + " for progress logs.")
    print("Note that THESE ARE RERUNS")
                
                






































if __name__ == "__main__":
   main(sys.argv[1:])



