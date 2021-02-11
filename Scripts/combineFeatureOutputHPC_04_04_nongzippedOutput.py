#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:55:32 2019

@author: dstoker

#update 27-07-2019
This script is run to combine the features calculated per loop into one file and generate
the final feature table.

Combines the features on the HPC from their concatenated format.
Also gives them headers.
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

#make this function HPC format compatible. All output files from loop calculations have been 
#concatenated to a file using command line: find . -name "*.csv" -print0 | xargs -I {} -0 cat {} | grep -v ',loopID,anchorType' | gzip > NAMEOFRESULT.csv.gz
#I now want to load this file, add headers, pivot, and save the output as a dictionary in a pickle.
#assumes that argument one is the input file 
#argument two is the output file path (with final /)
def main(argv):
    
    if len(argv) is not 2:
        print("Wrong arguments supplied. First argument should be input file")
        print("Second argument should be output file path.")
        sys.exit("Error: wrong number of arguments.")
    inputFilePath = argv[0]
    overwriteExistingFiles = False
    #inputFilePathWithoutTrailingSlash = inputFilePath.rstrip("/")
    now = str(datetime.datetime.today()).split()[0]
    #concatenate the per-loop feature files:
    folderNameToAddToFileName = os.path.split(os.path.dirname(inputFilePath))[1]
    nameConcatenatedInputFile = "allLoopFeaturesConcatenated_" + folderNameToAddToFileName + ".csv.gz"
    fullFilePathConcatenatedInputFile = inputFilePath + nameConcatenatedInputFile
    
    if os.path.isfile(fullFilePathConcatenatedInputFile):
        print("Data was already combined. Skipping step to concatenate individual loop files.")
        print("If this behaviour is undesired, change overwriteExistingFiles to True within the script file.")
        if overwriteExistingFiles == False:
            pass
        else:
            freek = subprocess.Popen("find " + inputFilePath +  " -name \"*.csv\" -print0 | xargs -I {} -0 cat {} | grep -v ',loopID,anchorType' | awk 'BEGIN {FS = \",\"; OFS = \",\"} {print $2,$3\"_\"$4,$5}' | gzip > " + fullFilePathConcatenatedInputFile,
                             shell = True)
            freek.wait()
    else:
        freek = subprocess.Popen("find " + inputFilePath +  " -name \"*.csv\" -print0 | xargs -I {} -0 cat {} | grep -v ',loopID,anchorType' | awk 'BEGIN {FS = \",\"; OFS = \",\"} {print $2,$3\"_\"$4,$5}' | gzip > " + fullFilePathConcatenatedInputFile,
                             shell = True)
        freek.wait()
    
    
    #subprocess.run("find " + inputFilePath +  " -name \"*.csv\" -print0 | xargs -I {} -0 cat {} | grep -v ',loopID,anchorType' | gzip > " + nameConcatenatedInputFile,
    #               check = True, shell = True)
    
    outputFilePath = argv[1]
    
    pd.set_option("display.max_columns", 10)
    print("Starting file read-in")
    
    
    headerNames = ["loopID","feature","value"]
    dataTypes   = ["category", "object", "float32"]
    freek = dict(zip(headerNames, dataTypes))
    dataFrameTotalFeatures = pd.read_csv(fullFilePathConcatenatedInputFile, header = None,
                                         names = headerNames, dtype = freek)
    print("Done. Head of file:")
    print(dataFrameTotalFeatures.head())
    
    def makeFeatureArray(dataFrameLoops: pd.DataFrame) -> dict:
            
            featuresDataFrame = dataFrameLoops.pivot(index = "loopID", columns = "feature", values = "value")
            
            loopID    = featuresDataFrame.index.values
            loopLabels   = np.array([loop.split("_")[1] for loop in loopID])
            featureNames = featuresDataFrame.columns.values
            arrayData    = featuresDataFrame.values
            
            return({"featureArray" : arrayData, "classLabelArray" : loopLabels,
                "namesFeatureArrayColumns" : featureNames, "loopID" : loopID})
        
    arrayDataDict = makeFeatureArray(dataFrameTotalFeatures)
    
    outputFileName = outputFilePath  + "finalFeatureDictionary_" + folderNameToAddToFileName + "_" + now + ".pkl"
    with open(outputFileName, "wb") as f:
        pickle.dump(arrayDataDict, f)
    
    print("done combining. Outputfile = " + outputFileName)



if __name__ == "__main__":
   main(sys.argv[1:])