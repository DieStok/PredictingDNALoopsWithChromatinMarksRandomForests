#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:34:12 2019

@author: dstoker

#update 27-07-2019
This script takes random anchor locations and real anchor locations and combines
the generated labels and features into one file for classification.

Script to load in the loose anchors that I made (Naive and Mimick). The question:
can random anchors be discerned from true ones on the basis of their chromatin mark signature?

This script pairs the features for the random data with that of the real data (without
in-between features) and saves it with class labels (real/fake)


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

normalFeatureDictPath = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/SavedFeatureDicts/finalFeatureDictionary_2019-04-08.pkl"
mimickFeatureDictPath = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/SavedFeatureDicts/finalFeatureDictionary_FeatureOutput_17_05_2019_MimickFakeAnchors_2019-05-22.pkl"
uniformFeatureDictPath = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/SavedFeatureDicts/finalFeatureDictionary_FeatureOutput_17_05_2019_UniformFakeAnchors_2019-05-22.pkl"

outputFilePathNormalFeaturesWithoutInbetween = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/SavedFeatureDicts/finalFeatureDictionary_2019-04-08_NoInBetween_28_05_2019.pkl"
outputFilePathSingleAnchorsMimickRandomVersusGM12878 = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/SavedFeatureDicts/finalFeatureDictionaryMimickVersusGM_40000SingleAnchorsEach_28_05_2019.pkl"
outputFilePathSingleAnchorsUniformRandomVersusGM12878 = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/SavedFeatureDicts/finalFeatureDictionaryUniformRandomVersusGM_40000SingleAnchorsEach_28_05_2019.pkl"

paths = [normalFeatureDictPath, mimickFeatureDictPath, uniformFeatureDictPath]

featList = []
for path in paths:
    with open(path, "rb") as f:
        featList.append(pickle.load(f))
        
        
featList[0]

inbetweenCols = np.where( ["inbetween" in item for item in featList[0]["namesFeatureArrayColumns"]])

#remove these from names and featureArray

newFeatNames = np.delete(featList[0]["namesFeatureArrayColumns"], inbetweenCols)
newFeats     = np.delete(featList[0]["featureArray"], inbetweenCols, axis = 1)

featList[0]["namesFeatureArrayColumns"] = newFeatNames
featList[0]["featureArray"] = newFeats

#save this separately for classification without in-between features
with open(outputFilePathNormalFeaturesWithoutInbetween, "wb") as f:
    pickle.dump(featList[0], f)
    
    
    
######################################Mimick and Uniform##########################
    
    
featList[1].keys()
featList[1]["featureArray"].shape
featList[1]["namesFeatureArrayColumns"]

#split and stack: split on anchors, stack together

rightAnchorIndices = np.where( ["rightAnchor" in item for item in featList[1]["namesFeatureArrayColumns"]])
leftAnchorIndices  = np.where( ["leftAnchor" in item for item in featList[1]["namesFeatureArrayColumns"]])
leftAnchorFeatNamesIsolated = np.delete(featList[1]["namesFeatureArrayColumns"], rightAnchorIndices)
rightAnchorFeatNamesIsolated = np.delete(featList[1]["namesFeatureArrayColumns"], leftAnchorIndices)

leftAnchorFeatsIsolated = np.delete(featList[1]["featureArray"], rightAnchorIndices, axis = 1)
rightAnchorFeatsIsolated = np.delete(featList[1]["featureArray"], leftAnchorIndices, axis = 1)

rightAnchorFeatNamesIsolated
rightAnchorFeatsIsolated.shape

leftAnchorFeatNamesIsolated
#now: remove right and left from the names and vstack the feature arrays. Then change class labels
#to reflect that they are all 0 (not true anchors)

newFeatNames = np.array([item.replace("left", "") for item in leftAnchorFeatNamesIsolated])

stackedSingleAnchorFeaturesMimick = np.vstack((leftAnchorFeatsIsolated, rightAnchorFeatsIsolated))

#give classLabel and then stack with the whole set of true loop anchor sites

classLabelsMimick = np.array(["MimickRandom"] * 40000)
loopIDsMimick     = ["Mimick_SingleAnchor_" + str(i) for i in range (0,40000)]


#get true anchor sites:

nonGM = np.where(featList[0]["classLabelArray"] == "Heart")
GMLoopsOnlyFeatures = np.delete(featList[0]["featureArray"], nonGM, axis = 0)
print(GMLoopsOnlyFeatures.shape)

rightAnchorIndicesGM = np.where( ["rightAnchor" in item for item in featList[0]["namesFeatureArrayColumns"]])
leftAnchorIndicesGM  = np.where( ["leftAnchor" in item for item in featList[0]["namesFeatureArrayColumns"]])
leftAnchorFeatsIsolatedGM = np.delete(GMLoopsOnlyFeatures, rightAnchorIndicesGM, axis = 1)
rightAnchorFeatsIsolatedGM = np.delete(GMLoopsOnlyFeatures, leftAnchorIndicesGM, axis = 1)

GMStackedFeaturesPerAnchor = np.vstack((leftAnchorFeatsIsolatedGM, rightAnchorFeatsIsolatedGM))
classLabelsGM = np.array(["GM12878"] * 40000)
loopIDsGM     = ["GM12878_SingleAnchor_" + str(i) for i in range (0,40000)]


#combine and save
mimickRandomAndRealGMAnchorFeatsOnlyTotal = np.vstack((stackedSingleAnchorFeaturesMimick,
                                                       GMStackedFeaturesPerAnchor))

totalFeatureDictMimickGM = {"featureArray" : mimickRandomAndRealGMAnchorFeatsOnlyTotal,
                            "classLabelArray" : np.vstack((classLabelsMimick, classLabelsGM)),
                            "namesFeatureArrayColumns" : newFeatNames,
                            "loopID" : np.vstack((loopIDsMimick, loopIDsGM))}

with open(outputFilePathSingleAnchorsMimickRandomVersusGM12878, "wb") as f:
    pickle.dump(totalFeatureDictMimickGM, f)
    
    
    
    
#now do the same thing for uniform fake anchors
    
    
rightAnchorIndicesUniformRandom = np.where( ["rightAnchor" in item for item in featList[2]["namesFeatureArrayColumns"]])
leftAnchorIndicesUniformRandom  = np.where( ["leftAnchor" in item for item in featList[2]["namesFeatureArrayColumns"]])
#leftAnchorFeatNamesIsolatedUniformRandom = np.delete(featList[2]["namesFeatureArrayColumns"], rightAnchorIndicesUniformRandom)
#rightAnchorFeatNamesIsolatedUniformRandom = np.delete(featList[2]["namesFeatureArrayColumns"], leftAnchorIndicesUniformRandom)

leftAnchorFeatsIsolatedUniformRandom = np.delete(featList[2]["featureArray"], rightAnchorIndicesUniformRandom, axis = 1)
rightAnchorFeatsIsolatedUniformRandom = np.delete(featList[2]["featureArray"], leftAnchorIndicesUniformRandom, axis = 1)

stackedSingleAnchorFeaturesUniformRandom = np.vstack((leftAnchorFeatsIsolatedUniformRandom, rightAnchorFeatsIsolatedUniformRandom))

#give classLabeland then stack with the whole set of true loop anchor sites

classLabelsUniformRandom = np.array(["UniformRandom"] * 40000)
loopIDsUniformRandom     = ["UniformRandom_SingleAnchor_" + str(i) for i in range (0,40000)]



totalFeatureDictUniformRandomGM = {"featureArray" : np.vstack((stackedSingleAnchorFeaturesUniformRandom, GMStackedFeaturesPerAnchor)),
                            "classLabelArray" : np.vstack((classLabelsUniformRandom, classLabelsGM)),
                            "namesFeatureArrayColumns" : newFeatNames,
                            "loopID" : np.vstack((loopIDsUniformRandom, loopIDsGM))}



with open(outputFilePathSingleAnchorsUniformRandomVersusGM12878, "wb") as f:
    pickle.dump(totalFeatureDictUniformRandomGM, f)


