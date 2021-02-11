#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:21:11 2019

@author: dstoker

#update 27-07-2019:
This script tallies the number of peaks of RAD21 and CTCF in anchor sites in my data.
I ran it interactively and copied data to a table (bad practice, yes).
Goal 2 was not further pursued.

Goals:
1. Open the bedfiles for the mimick anchors and the true anchors (filter for GM), and see how many CTCF are in them in total
2. Open the bedfiles for the GM loops with in-between and the heart loops with in-between, and see how many ATF-2, HDGF
are in-between. Perhaps with a sort of heatmap
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
import sys


folderForBedFilesMimickAnchors = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/FilesForHPCProcessing/BedIntersectDataFrames_RandomAnchorsMimick_Unpaired_17_05_2019"

folderForBedFilesUniformAnchors = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/FilesForHPCProcessing/BedIntersectDataFrames_RandomAnchorsUniform_Unpaired_17_05_2019"
#note that, for these files, I will need to filter out the intersects for heart loops
folderForBedFilesNormalGMLoops =  "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/FilesForHPCProcessing/BEDIntersectDataFrames"

listOfMimickFiles = glob.glob(folderForBedFilesMimickAnchors + "/" + "*.csv")
listOfUniformFiles = glob.glob(folderForBedFilesUniformAnchors + "/" + "*.csv")
listOfNormalLoopFiles = glob.glob(folderForBedFilesNormalGMLoops + "/" + "*.csv")
listOfNormalLoopFiles = [entry for entry in listOfNormalLoopFiles if not re.match("inbetween", entry)]

separateSelectionsToMake = ["2500", "10000"]
separateDataSets = ["Mimick", "Uniform", "NormalData"]
totalDictAnchorsAndFactors = {}
for dataSet in separateDataSets:
    print(dataSet)
    if dataSet == "Mimick":
        currentList = listOfMimickFiles
        subsetGM = False
    elif dataSet == "NormalData":
        currentList = listOfNormalLoopFiles
        subsetGM = True
    elif dataSet == "Uniform":
        currentList = listOfUniformFiles
        subsetGM = False
    for anchorSize in separateSelectionsToMake:
        totalCTCFIntersectsInAnchorsOfThisSize = 0
        totalCTCFIntersectsDFThisAnchorSize = None
        
        totalCohesinIntersectsInAnchorsOfThisSize = 0
        totalCohesinIntersectsDFThisAnchorSize = None
        
        for entry in currentList:
            print(entry)
            if "inbetween" in entry : continue
            if re.search(anchorSize, entry):
                print("matching")
                dataFrame = pd.read_csv(entry)
                if subsetGM == True:
                    dataFrame = dataFrame[dataFrame["areaType"].str.contains("GM12878", regex = False)]
                
                #now add the amount of CTCF peaks and cohesin peaks to them, along with some info
                CTCFOnly = dataFrame[dataFrame["factorName"] == "CTCF-human"]
                totalCTCFIntersectsInAnchorsOfThisSize += len(CTCFOnly)
                CTCFOnly["anchorSize"] = int(anchorSize)
                #divide by anchor Size divided by two because I have right and left sections of each anchor
                CTCFOnly["fractionalOverlap"] = CTCFOnly["overlapBasePairs"] / (float(anchorSize)/2)
                CTCFOnly["loopID"]     = [re.search(".*_(GM12878_\d+)", entry)[1] for entry in CTCFOnly["areaType"]]
                try:
                    CTCFOnly["anchorSide"] = [re.search("^(\w+Anchor).*", entry)[1] for entry in CTCFOnly["areaType"]]
                except:
                    print(dataSet)
                    print(anchorSize)
                    print(CTCFOnly["areaType"])
                
                CTCFOnly["anchorSideLoopID"] = CTCFOnly["anchorSide"] + "_" + CTCFOnly["loopID"]
                
                totalCTCFIntersectsDFThisAnchorSize = pd.concat([totalCTCFIntersectsDFThisAnchorSize,
                                                                 CTCFOnly])
                #same for cohesin
                cohesinOnly = dataFrame[dataFrame["factorName"] == "RAD21-human"]
                totalCohesinIntersectsInAnchorsOfThisSize += len(cohesinOnly)
                cohesinOnly["anchorSize"] = int(anchorSize)
                cohesinOnly["fractionalOverlap"] = cohesinOnly["overlapBasePairs"] / (float(anchorSize)/2)
                
                cohesinOnly["loopID"]     = [re.search(".*_(GM12878_\d+)", entry)[1] for entry in cohesinOnly["areaType"]]
                cohesinOnly["anchorSide"] = [re.search("(^\w+Anchor).*", entry)[1] for entry in cohesinOnly["areaType"]]
                
                cohesinOnly["anchorSideLoopID"] = cohesinOnly["anchorSide"] + "_" + cohesinOnly["loopID"]
                
                totalCohesinIntersectsDFThisAnchorSize = pd.concat([totalCohesinIntersectsDFThisAnchorSize,
                                                                    cohesinOnly])
            else:
                continue
                
                
        if dataSet not in totalDictAnchorsAndFactors.keys():
            totalDictAnchorsAndFactors[dataSet] = {}
        else:
            pass
        if anchorSize not in totalDictAnchorsAndFactors[dataSet]:
            totalDictAnchorsAndFactors[dataSet][anchorSize] = {}
        else:
            pass
        totalDictAnchorsAndFactors[dataSet][anchorSize]["counts"]    = {"cohesin" : totalCohesinIntersectsInAnchorsOfThisSize,
                                                                        "CTCF"    : totalCTCFIntersectsInAnchorsOfThisSize}
        totalDictAnchorsAndFactors[dataSet][anchorSize]["DFCTCF"]    = totalCTCFIntersectsDFThisAnchorSize
        totalDictAnchorsAndFactors[dataSet][anchorSize]["DFCohesin"] = totalCohesinIntersectsDFThisAnchorSize
                
totalDictAnchorsAndFactors


#combine leftAnchorLeft and leftAnchorRIght into just leftAnchor, same for right anchor:

totalDictAnchorsAndFactors["NormalData"]["2500"]["DFCTCF"].groupby(["loopID", "anchorSide"])["fractionalOverlap"].mean()
#Name: fractionalOverlap, Length: 2776, dtype: float64
#so only 2776 anchors overlap with CTCF peaks as deemed to be significant by IDR for 2500
totalDictAnchorsAndFactors["NormalData"]["10000"]["DFCTCF"].groupby(["loopID", "anchorSide"])["fractionalOverlap"].mean()
#Name: fractionalOverlap, Length: 8505, dtype: float64
#And 8505 anchors out of 40.000 anchors for 10.000 anchor size. Understandable that it's not a major feature then.

#if we look at mimick:
totalDictAnchorsAndFactors["Mimick"]["2500"]["DFCTCF"].groupby(["loopID", "anchorSide"])["fractionalOverlap"].mean()
#Name: fractionalOverlap, Length: 2146, dtype: float64
#2146 anchors total for mimick

totalDictAnchorsAndFactors["Mimick"]["10000"]["DFCTCF"].groupby(["loopID", "anchorSide"])["fractionalOverlap"].mean()
#Name: fractionalOverlap, Length: 7103, dtype: float64
#so there is a difference, but it is very small relative to the amount of anchors. ~1400 less anchors are covered in this
#category, out of the 40.000 anchors. 

#if we look at uniform:
totalDictAnchorsAndFactors["Uniform"]["2500"]["DFCTCF"].groupby(["loopID", "anchorSide"])["fractionalOverlap"].mean()
#Name: fractionalOverlap, Length: 1632, dtype: float64
#so very little, only 1632 out of 40.000 anchors. This goes to show that there seems to be some enrichment of CTCF in loop-heavy areas

totalDictAnchorsAndFactors["Uniform"]["10000"]["DFCTCF"].groupby(["loopID", "anchorSide"])["fractionalOverlap"].mean()
#Name: fractionalOverlap, Length: 5473, dtype: float64
#5473 out of 40.000 single anchors. Less than Mimick and than the real deal.

#guess we can conclude that CTCF signal is not great. 



totalDictAnchorsAndFactors["NormalData"]["2500"]["DFCohesin"].groupby(["loopID", "anchorSide"])["fractionalOverlap"].mean()
#Name: fractionalOverlap, Length: 2718, dtype: float64
#so only 2718 anchors overlap with cohesin peaks as deemed to be significant by IDR for 2500
totalDictAnchorsAndFactors["NormalData"]["10000"]["DFCohesin"].groupby(["loopID", "anchorSide"])["fractionalOverlap"].mean()
#Name: fractionalOverlap, Length: 8450, dtype: float64
#And 8450 anchors out of 40.000 anchors for 10.000 anchor size. Understandable that it's not a major feature then.
#Do note that this is nearly the same amount as CTCF. I have not expressly checked whether they are the same anchors,
#but this does imply that anchor coverage is stable for CTCF and cohesin in this data.

#if we look at mimick:
totalDictAnchorsAndFactors["Mimick"]["2500"]["DFCohesin"].groupby(["loopID", "anchorSide"])["fractionalOverlap"].mean()
#Name: fractionalOverlap, Length: 2030, dtype: float64
#2030 anchors total for mimick

totalDictAnchorsAndFactors["Mimick"]["10000"]["DFCohesin"].groupby(["loopID", "anchorSide"])["fractionalOverlap"].mean()
#Name: fractionalOverlap, Length: 6828, dtype: float64
#so there is a difference, but it is very small relative to the amount of anchors. ~1650 anchors amongst 40.000 anchors
#per category.

#if we look at uniform:
totalDictAnchorsAndFactors["Uniform"]["2500"]["DFCohesin"].groupby(["loopID", "anchorSide"])["fractionalOverlap"].mean()
#Name: fractionalOverlap, Length: 1574, dtype: float64
#so very little, only 1574 out of 40.000 anchors. This goes to show that there seems to be some enrichment of cohesin
# in loop-heavy areas (if we compare with mimick)

totalDictAnchorsAndFactors["Uniform"]["10000"]["DFCohesin"].groupby(["loopID", "anchorSide"])["fractionalOverlap"].mean()
#Name: fractionalOverlap, Length: 5329, dtype: float64
#less than Mimick, which in turn is less than NormalData. But still not the huge difference you would expect.


#now for both together

sum(totalDictAnchorsAndFactors["NormalData"]["2500"]["DFCohesin"].drop_duplicates(
        subset = "anchorSideLoopID", keep = "first", inplace = False)["anchorSideLoopID"].isin(
                totalDictAnchorsAndFactors["NormalData"]["2500"]["DFCTCF"].drop_duplicates(
        subset = "anchorSideLoopID", keep = "first", inplace = False)["anchorSideLoopID"]))


sum(totalDictAnchorsAndFactors["NormalData"]["10000"]["DFCohesin"].drop_duplicates(
        subset = "anchorSideLoopID", keep = "first", inplace = False)["anchorSideLoopID"].isin(
                totalDictAnchorsAndFactors["NormalData"]["10000"]["DFCTCF"].drop_duplicates(
        subset = "anchorSideLoopID", keep = "first", inplace = False)["anchorSideLoopID"]))


#if we look at mimick:
sum(totalDictAnchorsAndFactors["Mimick"]["2500"]["DFCohesin"].drop_duplicates(
        subset = "anchorSideLoopID", keep = "first", inplace = False)["anchorSideLoopID"].isin(
                totalDictAnchorsAndFactors["Mimick"]["2500"]["DFCTCF"].drop_duplicates(
        subset = "anchorSideLoopID", keep = "first", inplace = False)["anchorSideLoopID"]))


sum(totalDictAnchorsAndFactors["Mimick"]["10000"]["DFCohesin"].drop_duplicates(
        subset = "anchorSideLoopID", keep = "first", inplace = False)["anchorSideLoopID"].isin(
                totalDictAnchorsAndFactors["Mimick"]["10000"]["DFCTCF"].drop_duplicates(
        subset = "anchorSideLoopID", keep = "first", inplace = False)["anchorSideLoopID"]))

#Uniform

sum(totalDictAnchorsAndFactors["Uniform"]["2500"]["DFCohesin"].drop_duplicates(
        subset = "anchorSideLoopID", keep = "first", inplace = False)["anchorSideLoopID"].isin(
                totalDictAnchorsAndFactors["Uniform"]["2500"]["DFCTCF"].drop_duplicates(
        subset = "anchorSideLoopID", keep = "first", inplace = False)["anchorSideLoopID"]))


sum(totalDictAnchorsAndFactors["Uniform"]["10000"]["DFCohesin"].drop_duplicates(
        subset = "anchorSideLoopID", keep = "first", inplace = False)["anchorSideLoopID"].isin(
                totalDictAnchorsAndFactors["Uniform"]["10000"]["DFCTCF"].drop_duplicates(
        subset = "anchorSideLoopID", keep = "first", inplace = False)["anchorSideLoopID"]))


####################Heatmap or something can be done below