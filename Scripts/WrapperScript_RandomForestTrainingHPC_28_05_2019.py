#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:21:09 2019

@author: dstoker

#update 27-07-2019
Comments below are already pretty self-explanatory. Note that I updated the
run times and amount of cores requested so no rerunning should be necessary if all is well.
If an anchor size is not supplied, classifiers are trained on feature sets of all anchor
sizes in the total feature set.

Wrapper script that reads in a file containing train-test splits for a number
of repeats and a number of cross-validations, and then starts a script
that trains random forest classifiers on these subsets of the data.

#arguments

1: input .pkl file containing the calculated features, their names, class labels, and loopID
2: inpute .pkl file containing the train-test splits to give to instances of the Random Forest Training script
3: output file directory, where Random Forest Training scripts can deposit their outputs
4 and on: (optional) anchorSizes for which to calculate:
    so (arg1, arg2, arg3, 2500, 10000, ...)
    Note that these anchor sizes need to have features calculated for them in the 
    file supplied as arg1.
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
    
    if len(argv) >= 3:
        pass
    else:
        sys.exit("Insufficient arguments given./nThis programme needs at least a feature dictionary .pkl (arg1) /n a train-test .pkl /n and an output folder. /n Optionally, anchor sizes for which to train separate RF classifiers can be specified after those three as 2500, 10000, ... ")
    
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
        
    if len(argv) > 3:
        anchorSizesToSelect = argv[3:]
    else:
        #if none, then no argument needs to be supplied, so an empty string will be appended.
        anchorSizesToSelect = [""]
    
    #create a log folder if it doesn't exist
    if os.path.isdir(outputFileDir + "logs/"):
        pass
    else:
        os.makedirs(outputFileDir + "logs/")
    logDir = outputFileDir + "logs/"
    
    
    inputFileNameToAppendToJobName = os.path.basename(os.path.splitext(inputFeatureDictPath)[0])
    #load the test-train file, loop over anchorSizesToSelect, reps and inputs
    print("Starting HPC jobs")
    print("-----------------")
    for anchorToCalcFor in anchorSizesToSelect:
        #print(anchorToCalcFor)
        for repCount, repData in enumerate(trainTestSplitInfo):
            #print(repCount)
            #print(repData)
            for crossValCount, crossValData in enumerate(repData):
                #print(crossValCount)
                #print(crossValData)
                
                
#THIS is old, better to just have the random Forest process read from the pickle 
#                trainData = [str(item) for item in list(trainTestSplitInfo[repCount][crossValCount][0])]
#                testData  = [str(item) for item in list(trainTestSplitInfo[repCount][crossValCount][1])]
#                
#                trainData2 = "[" + ",".join(trainData) + "]"
#                testData2  = "[" + ",".join(testData)  + "]"
                
                
                
                repCrossValInfoToParse        = "rep_" + str(repCount + 1) + "_of" +\
                                                str(len(trainTestSplitInfo)) + "_fold_" + \
                                                str(crossValCount + 1) + "_of" + \
                                                str(len(repData))
                
                
                #start the jobs
                
                print("Running job for data of: ")
                print(repCrossValInfoToParse)
                os.system("qsub -cwd -N DieterRFCalc_" + inputFileNameToAppendToJobName + \
                 "_" + anchorToCalcFor + " " +
                 "-o " + logDir + repCrossValInfoToParse + "_" + anchorToCalcFor + \
                 "_log_out.txt " + "-l h_vmem=65G " + "-l h_rt=56:00:00 " + "-pe threaded 3 " + \
                 "/hpc/cog_bioinf/ridder/users/dstoker/scripts/run_script.sh " + "python " + \
                 "/hpc/cog_bioinf/ridder/users/dstoker/scripts/TrainRandomForestHPC_IndicesTrainTestInput_27_05_2019.py " + \
                 trainTestSplitFilePath + " " + repCrossValInfoToParse + " " + \
                 inputFeatureDictPath + " " + outputFileDir + " " + anchorToCalcFor)
                
    print("--------------")
    print("All jobs started. Please check " + logDir + " for progress logs.")
                
                






































if __name__ == "__main__":
   main(sys.argv[1:])



