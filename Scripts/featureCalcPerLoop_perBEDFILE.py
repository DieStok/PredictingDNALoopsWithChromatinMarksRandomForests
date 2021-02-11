#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:09:33 2019

@author: dstoker

#update 27-07-2019
This script is started by the wrapper script. It calculates features for 
the .bed files that it is passed, i.e. for all sequence stretches for that loop.
#

changes the random forest feature calculation to be command-line based, taking
in any amount of .csv files from which to calculate features and save these to a file.

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


def main(argv):
    
    #assumes that first argument received is the output folder (without trailing /)
    #rest of the arguments are input files
    #chromatin marks are assumed to be in the same input folder.
    
    pd.set_option('expand_frame_repr', True)
    pd.set_option("display.max_columns", 40)
    
    print("start programme at: " + str(datetime.datetime.today()).split()[0])
    outputFilePath      = argv[0] #note that main is passed argv[1:] upon actual execution
    filesToCalculateFor = argv[1:-1]
    loopToCalculateFor  = argv[-1]
    fileNameList        = [os.path.splitext(os.path.split(file)[1])[0] for file in filesToCalculateFor]
    loopID = loopToCalculateFor
    chromatinMarkList = []
    #assumes chromatinMark.txt is in the same folder as all other inputs
    if(os.path.isfile(os.path.split(argv[1])[0] + "/chromatinMarks.txt")) :
        with open(os.path.split(argv[1])[0] + "/chromatinMarks.txt") as f:
            chromatinMarkList = [line.rstrip() for line in f.readlines()]
    else:
        print("--- ERROR: chromatinMarks.txt not in input folder! ---")
        sys.exit("Rerun with the correct file in the folder")
    
    
    #get the lines per csv file to read. Always read the first line (header)
    startLineAndLinesToReadPerFile = []
    for fileToRead in filesToCalculateFor:
        linesToReadAwk = subprocess.check_output("awk 'BEGIN {FS = \",\";} $5 ~ /" +
                                                 loopID + "$/ {print NR;}'" + " " +
                                                 fileToRead + "",
                                                 stderr=subprocess.STDOUT,
                                                 shell = True)
        linesToReadList = linesToReadAwk.decode().rstrip().split("\n")
        linesToReadList = [int(value) for value in linesToReadList]
        
        startLine = linesToReadList[0] -1 #-1 because zero-indexed
        if(len(linesToReadList) == 1): 
            endLine   = linesToReadList[0]
            linesToRead = 1
        else:
            endLine = linesToReadList[-1]
            linesToRead = endLine - startLine
        startLineAndLinesToReadPerFile.append([startLine, linesToRead]) 
    
        
    
    #with this information, calculate features

    def make_random_forest_features_command_line(listIntersectBEDFiles: list,
                                                 chromatinMarkList    : list,
                                                 loopToCalc           : str ,
                                                 lineInformation      : list) -> pd.DataFrame:
        
        listChromatinFactors = chromatinMarkList
        dictFeatures = {}
        counter = 0
        counterLoops = 0
        lenListFiles = len(listIntersectBEDFiles)
        
        listFeatureNames = ["intersectsPerFactor", "totalOverlap", "totalOverlapFraction",
                                           "avgSignalValue", "medianSignalValue",
                                           "avgPValueNegativeLog10",
                                           "medPValueNegativeLog10",
                                           "avgQValueNegativeLog10",
                                           "medQValueNegativeLog10",
                                           "sumPointPeaksInInterval"]
        
        
        namesFeatureColumns = []
        for nr, bedFile in enumerate(listIntersectBEDFiles):
            
            #get the names only once by  using this bool. Since every Bed file is of a specific
            #anchor position, those variables only need to be set on the first iteration
            #through all loops.
            first = True
            
            print("working on file: " + str(counter + 1) + " out of " + str(lenListFiles))
            print("---")
            counter += 1
            
            columnNames  = pd.read_csv(bedFile, skiprows = 0, nrows = 0, index_col = 0).columns
            
            DataFrameBED = pd.read_csv(bedFile,
                                       skiprows = lineInformation[nr][0],
                                       nrows    = lineInformation[nr][1],
                                       index_col = 0, header = None)
            DataFrameBED.columns = columnNames
            #get only the parts pertaining to the loop this script calculates
            
            
            DataFrameBEDLoopSpecific = DataFrameBED[DataFrameBED.loc[:, "areaType"].str.endswith(
                    loopToCalc)]
            
            groupedByLoop = DataFrameBEDLoopSpecific.groupby("areaType")
            
            #initialise feature names only once per BED file
            if(first == True):
                for name, group in groupedByLoop:
                
                    widthArea = int(group["areaType"].tolist()[0].split("_")[1])
                    nameType = group["areaType"].tolist()[0].split("_")[0] + "_" + str(widthArea)
                
                    #make the name for all features:
                    chromatinMarkFeatureNames = [nameType + "_" + chroMark + "_" + featName for chroMark in 
                                         listChromatinFactors for featName in listFeatureNames]
                    namesFeatureColumns.append(chromatinMarkFeatureNames)
                    first = False
                    break
            
            #filter into loops with no intersections (on current anchor site) and those that do have that.
            groupedByLoopNoIntersects = groupedByLoop.filter(lambda groups: groups.shape[0] == 1 and \
                                                             groups["overlapBasePairs"] == 0)
            groupedByLoopWithIntersects = groupedByLoop.filter(lambda groups: groups.shape[0] >= 1 and \
                                                               groups[groups["chromosomeFactor"].isin([0])].shape[0] == 0)
            groupedByLoopWithIntersects = groupedByLoopWithIntersects.groupby("areaType")
            
            
            #for those that do not have it, add values for all factors at once:
            #check first if this loop indeed doesn't have intersects
            if(groupedByLoopNoIntersects.size == 0):
                toAddForNonIntersectingAnchors = [0] * len(chromatinMarkFeatureNames) 
                for index, row in groupedByLoopNoIntersects.iterrows():
                    
                    if(index % 250 == 0):
                        print("Working on anchors without intersection, nr " + str(index) + " out of " + 
                              str(len(groupedByLoopNoIntersects)) + " for file" + str(counter + 1) + ".")
                    loopID              = row["areaType"].split("_")[3]
                    loopType            = row["areaType"].split("_")[2]
                    nameToGiveDictEntry = loopID + "_" + loopType
                    
                    if nameToGiveDictEntry in dictFeatures: 
                        dictFeatures[nameToGiveDictEntry][nameType] = toAddForNonIntersectingAnchors
                    else:
                        dictFeatures[nameToGiveDictEntry] = {nameType : toAddForNonIntersectingAnchors}
                    
            #for those that do intersect with factors, calculate these intersections    
            for name, group in groupedByLoopWithIntersects:
                
                if(counterLoops % 250 == 0):
                    print("Calculating loop: " + str(counterLoops) + ".")
                counterLoops += 1
                
                
                #get the loop ID and type
                print(type(group["areaType"]))
                print(group["areaType"])
                print(type(groupedByLoopWithIntersects))
                splitNameForID = group["areaType"].iloc[0].split("_")
                loopID   = splitNameForID[3]
                loopType = splitNameForID[2]
                   
                #if there is no intersect with any factor at all (LEGACY COMMENT, SEE BELOW)
                #this should not occur in this new code. Print if in here.
                if((group.shape[0] == 1) & (int(group["overlapBasePairs"].tolist()[0]) == 0)):
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("Somehow this still happens!")
                    print(group)
                    print(name)
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    time.sleep(3)
                    groupedByChromFactor = group.groupby("factorName")      
       
                else:
                    #group the subset by chromatin factor, sum overlaps, while averaging the rest
                    #also get the total amount of overlaps within this range with each individual chromatin factor
    
                    group["factorScore"]         = pd.to_numeric(group["factorScore"])
                    group["factorSignalValue"]   = pd.to_numeric(group["factorSignalValue"])
                    group["pValueNegativeLog10"] = pd.to_numeric(group["pValueNegativeLog10"])
                    group["qValueNegativeLog10"] = pd.to_numeric(group["qValueNegativeLog10"])
                    group["pointPeak"]           = pd.to_numeric(group["pointPeak"])
                    group["pointPeakInInterval"] = [0 if row["pointPeak"] + row["factorStart"] < \
                                                         row["areaStart"] or 
                                                         row["pointPeak"] + row["factorStart"] >=  \
                                                         row["areaEnd"  ] else 1 for index,row in group.iterrows()  
                                               ]
                    
                    groupedByChromFactor = group.groupby("factorName")
                    
                    intersectsPerFactor     = groupedByChromFactor["factorName"].count()
                    totalOverlap            = groupedByChromFactor["overlapBasePairs"].sum()
                    if(re.search("inbetween", loopType)):
                        totalOverlapFraction = groupedByChromFactor["overlapBasePairs"].sum() / \
                        (groupedByChromFactor["AreaEnd"] - groupedByChromFactor["AreaStart"])
                    else:
                        #widthArea = interval. Leftanchorleft and leftanchorright together = interval,
                        #so the sum / widthArea /2. Same goes for rightanchor.
                        totalOverlapFraction    = groupedByChromFactor["overlapBasePairs"].sum() / widthArea / 2
                    avgSignalValue          = groupedByChromFactor["factorSignalValue"].mean()
                    medianSignalValue       = groupedByChromFactor["factorSignalValue"].median()
                    avgPValueNegativeLog10  = groupedByChromFactor["pValueNegativeLog10"].mean()
                    medPValueNegativeLog10  = groupedByChromFactor["pValueNegativeLog10"].median()
                    avgQValueNegativeLog10  = groupedByChromFactor["qValueNegativeLog10"].mean()
                    medQValueNegativeLog10  = groupedByChromFactor["qValueNegativeLog10"].median()
                    sumPointPeaksInInterval = groupedByChromFactor["pointPeakInInterval"].sum()
                    
                    dfValues = pd.concat([intersectsPerFactor, totalOverlap, totalOverlapFraction,
                                           avgSignalValue, medianSignalValue,
                                           avgPValueNegativeLog10,
                                           medPValueNegativeLog10,
                                           avgQValueNegativeLog10,
                                           medQValueNegativeLog10,
                                           sumPointPeaksInInterval], axis = 1)
                    
                    dfValues.columns = listFeatureNames
    
                    #print(medQValueNegativeLog10)    
            
                    #now for every chromatin factor in the data, go through it
                    #see if it is in this loop. If so, add it to data for this loop.   
                
                #print(intersectsPerFactor)
                
                #featureNameList = []
                valueList       = []
                for factor in listChromatinFactors:
                    #listFeatureNames should be the order of the columns in the dataframe as well
                    for feat in listFeatureNames:
                        #finalFeaturesColName  = nameType + "_" + factor  + "_" + feat
                        if factor in groupedByChromFactor.groups:
                            #print("In here man")
                            finalFeaturesColValue = dfValues.loc[factor, feat]
                        else:
                            finalFeaturesColValue = 0
                            
                        #featureNameList.append(finalFeaturesColName)
                        valueList.append(finalFeaturesColValue)
                
                #print(featureNameList)
                #print(valueList)
                nameToGiveDictEntry = loopID + "_" + loopType
                
                #print(len(chromatinMarkFeatureNames) == len(valueList))
                
                #if the loopID exists, add the data for this interval/location
                if nameToGiveDictEntry in dictFeatures: 
                    dictFeatures[nameToGiveDictEntry][nameType] = valueList
                else:
                    dictFeatures[nameToGiveDictEntry] = {nameType : valueList}
                    
        #change the final dictionary to a pd.DataFrame (loopID, anchorType, features)
                
        listDFEntries = []
        for loopEntry, values in dictFeatures.items():
            for anchor, features in values.items():
                dataFrameEntry = [loopEntry, anchor] + features
                listDFEntries.append(dataFrameEntry)
        
        
        #print(namesFeatureColumns); print("length featureColumns: " + str(len(namesFeatureColumns)))                
        resultDataFrame = pd.DataFrame(listDFEntries)
        
        nonAnchorSpecificNames = [chroMark + "_" + featName for chroMark in 
                                         listChromatinFactors for featName in listFeatureNames]
        
        #totalNameList = [item for sublist in namesFeatureColumns for item in sublist]
        print(resultDataFrame.head())
        resultDataFrame.columns = ["loopID", "anchorType"] + nonAnchorSpecificNames
                
                #I am making a terrible mess of this code because I am changing formats for
                #easier downstream processing. Now I will split the names and give them as 
                #columns ["featName", "chromatinFact"] for easier pivoting downstream.
        meltedResult = resultDataFrame.melt(id_vars = ["loopID", "anchorType"])
    
        return(meltedResult)
        
    calculateStuff = make_random_forest_features_command_line(filesToCalculateFor,
                                                              chromatinMarkList,
                                                              loopID,
                                                              startLineAndLinesToReadPerFile)
    
    #now write the resultant files in the output folder
    if(os.path.exists(outputFilePath)):
        pass
    else:
        os.makedirs(outputFilePath)
    calculateStuff.to_csv(outputFilePath + "/" + loopID + ".csv")
    
#    #featureNames need to be saved only once (if that)
#    if(os.path.isfile(outputFilePath + "/" + "featureNames.txt")):
#        pass
#    else:
#        with open(outputFilePath + "/" + "featureNames.txt", "w") as featFile:
#            for name in calculateStuff[1]:
#                featFile.write(name + "\n")
        
    print("done running at: " + str(datetime.datetime.today()).split()[0])
        
        

if __name__ == "__main__":
   main(sys.argv[1:])


 
