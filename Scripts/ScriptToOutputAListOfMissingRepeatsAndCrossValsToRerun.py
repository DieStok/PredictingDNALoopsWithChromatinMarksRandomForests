#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:55:27 2019

@author: dstoker

#update 27-07-2019
This script was made because some jobs did not finish in time on the HPC. It 
must be run via command line on the HPC, and outputs a .csv of the repeats and
cross-validation folds that should be rerun because they did not finish.
Another script (WrapperScript_RandomForestTrainingHPC_RerunningSpecificRepsAndCrossVals_14_06_2019.py)
then takes that file as an input and starts jobs only for those files.

###
###NOTE THAT I HAVE SINCE CHANGED THE ORIGINAL SCRIPT TO RESERVE MORE TIME SO THIS SCRIPT SHOULD NOT BE NECESSARY ANY LONGER
###


Script to run to restart jobs that haven't finished with:
    1 more core (3 instead of 2)
    24 hours more time
    Should:
        a)  read in fileNames from inputDir
        b ) read in fileNames from inputDir/logs
        -compare what fileNames are present in b but not in a
        -output a .csv with two columns: rep, crossVal
            which contain the rep and crossVal for this classifier that need to be redone
    I can then feed this to a modified training script, which only starts those jobs
"""
import sys, getopt
import pandas as pd, numpy as np
import os, time, errno, glob
import datetime
import subprocess

import re
import gzip
import shutil
import pybedtools as pbed
import pickle 
import seaborn as sb
import scipy as scp
#import plotnine as pltn
import sklearn as sk
import math, random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import RobustScaler


def main(argv):
    if os.path.isdir(argv[0]):
        inputDir = argv[0]
        if inputDir.endswith("/"):
            pass
        else:
            inputDir += "/"
    else:
        sys.exit("Input should be classifier output directory!")
    
    if os.path.isdir(inputDir + "logs"):
        logsFolder = inputDir + "logs/"
    else: 
        sys.exit("Folder contains no logs folder. This is necessary for comparison. Check this!")
    
    
    #match files
    inputFileList = glob.glob(inputDir + "*.pkl")
    logsFileList  = glob.glob(logsFolder + "*.txt")
    
    #get what is in the logs
    anchorSizesPresentLogs = [re.match(".*_(\d+)_log_out.txt|.*(__)log_out.txt", entry) for entry in \
                          logsFileList]
    numbersPresent = set([entry[1]  for entry in anchorSizesPresentLogs if entry is not None and entry[1] is not None])
    togetherPresent = set([entry[2]  for entry in anchorSizesPresentLogs if entry is not None and entry[2] is not None])
    anchorSizesPresentLogs = set([entry[1]  for entry in anchorSizesPresentLogs if entry is not None])
    totalLogFileVarietiesPresent = numbersPresent.union(togetherPresent)
    
    #Now, go through each of these anchor types, capture reps and crossVals in the  logs,
    #match to reps and CrossVals in outputFiles
    toRerunTotalDF = None
    for anchorType in totalLogFileVarietiesPresent:
        if anchorType == "__":
            toMatchPkls = ".*_AllAnchorsTogether_Data.*"
            anchorName = "AllAnchorsTogether"
        else:
            toMatchPkls = ".*_AnchorSize_" + anchorType + "_Data.*"
            anchorName = anchorType
        
        subsetPklFilesThisAnchor = np.where([re.match(toMatchPkls, entry) for entry in inputFileList])
        outputFilesToTakeThisAnchor = [inputFileList[i] for i in subsetPklFilesThisAnchor[0]]
        
        subsetLogFilesThisAnchor = np.where([re.match(".*_" + anchorType + "_log_out.txt|.*" + anchorType + "log_out.txt", entry) is not None for entry in \
                          logsFileList])
        logFilesToTakeThisAnchor = [logsFileList[i] for i in subsetLogFilesThisAnchor[0]]
        
        listReps = []
        listCrossVals = []
        for logFileName in logFilesToTakeThisAnchor:
            regexMatchRepFold = re.match(".*rep_(\d+)_of\d+_fold_(\d+)_of\d+.*",
                                         logFileName)
            listReps.append(regexMatchRepFold[1])
            listCrossVals.append(regexMatchRepFold[2])
            
            
        #now check for each of these pairings whether they are present in the pkl files.
        #if not, add them to a dataFrame (together with the anchorType)
        dataFrameThisAnchorRepsAndCrossValsToRun = None
        for rep, crossVal in zip(listReps, listCrossVals):
            areAnyOfTheFilesForThisRepAndCrossVal = [re.match(toMatchPkls + "_rep_" + str(rep) + "_of\d+_fold_" + str(crossVal) +"_of\d+.*", entry) for entry in outputFilesToTakeThisAnchor]
            if not all(entry is None for entry in areAnyOfTheFilesForThisRepAndCrossVal):
                pass #it is present
            else:
                rowToAdd = pd.DataFrame({"rep": [rep],
                                         "fold" : [crossVal],
                                         "anchor" : [anchorName]})
                dataFrameThisAnchorRepsAndCrossValsToRun = pd.concat([dataFrameThisAnchorRepsAndCrossValsToRun,
                                                                     rowToAdd])
        toRerunTotalDF = pd.concat([toRerunTotalDF, dataFrameThisAnchorRepsAndCrossValsToRun])
        
    toRerunTotalDF.to_csv(inputDir + "repsAndCrossValsToRerun.csv")
    

if __name__ == "__main__":
   main(sys.argv[1:])

