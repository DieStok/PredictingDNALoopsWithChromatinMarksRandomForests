#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:00:46 2019

@author: dstoker

#update 27-07-2019
I run this locally to generate, per feature .pkl, 5 sets of cross-folds for which
to calculate classifiers (50 classifiers are thus trained per anchor size feature set in total)

make 5 repeats of 10 folds of my data, to be saved and fed into the random forest script 
on the HPC

Command-line. argument one: the path of the feature .pkl
argument two: the path to output a matrix of the results (or something)

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


def main(argv):
    
    if len(argv) == 2:
        pass
    else:
        sys.exit("Supplied the wrong number of arguments. Supply input file path (featureDict.pkl) and output file folder.")
    if os.path.isfile(argv[0]):
        inputFilePath = argv[0]
    else:
        sys.exit("Input file path is incorrect.")
    if (os.path.isdir(argv[1])):
        outputFileDir = argv[1]
    else:
        try: 
            os.makedirs(argv[1])
        except:
            sys.exit("Supplied output path does not match an existing directory and directory could not be made.")
        outputFileDir = argv[1]
    
    #read in
    
    with open(inputFilePath, "rb") as f:
        featureDict = pickle.load(f)
    
    #rename and remove p-value columns which are useless
    featuresNotNormalised = featureDict["featureArray"]
    columnsNotPValue = [k for k in featureDict["namesFeatureArrayColumns"] if not k.endswith("avgPValueNegativeLog10") and not k.endswith("medPValueNegativeLog10")]
    indicesFeatureArrayToTake = [featureDict["namesFeatureArrayColumns"].tolist().index(k) for k in featureDict["namesFeatureArrayColumns"] if \
                                 k in columnsNotPValue]
    indicesFeatureArrayToTake
    
    featuresNotNormalised = featureDict["featureArray"][:, indicesFeatureArrayToTake]

    

    #get labels and reshape etc.
    numericLabels = sk.preprocessing.LabelBinarizer()
    #input to labelBinarizer should be reshaped
    reshapedLabels = featureDict["classLabelArray"].reshape(-1, 1)
    numericLabels.fit(reshapedLabels)
    transformedClassLabels = numericLabels.transform(reshapedLabels)
    transformedClassLabels = np.ravel(transformedClassLabels)    
    
    xInput, yInput = featuresNotNormalised, transformedClassLabels                                                                        


    def generate_indices(xInput, yInput, repeats: int = 5, folds: int = 10):
        
        totalList = []
        
        for rep in range(0, repeats):
            perRepList = []
            splitter = sk.model_selection.StratifiedKFold(n_splits = folds, shuffle = True)
            for trainIndices, testIndices in splitter.split(xInput, yInput):
                perFoldList = [trainIndices, testIndices]
                perRepList.append(perFoldList)
            totalList.append(perRepList)
        return (totalList)
            
    
    generatedIndices = generate_indices(xInput, yInput, 5, 10)    
    generatedIndices            
    len(generatedIndices)        
    len(generatedIndices[0])    
    len(generatedIndices[0][0])    
    len(generatedIndices[0][0][1])  
    
    #if I for-loop over the first and second list I can get the indices I need for
    #the random forest in a wrapper script.
    if outputFileDir.endswith("/"):
        pass
    else:
        outputFileDir += "/"
    now = str(datetime.datetime.today()).split()[0]
    saveFileTotalPath = outputFileDir + "crossValRepIndicesFor_" + \
              os.path.splitext(os.path.basename(inputFilePath))[0] + "_" + \
              now + ".pkl"
    with open(saveFileTotalPath, "wb") as pickleFile:
        pickle.dump(generatedIndices, pickleFile)
    print("Done generating. Files saved in: " + saveFileTotalPath + " .")
    

if __name__ == "__main__":
   main(sys.argv[1:])



   
   

