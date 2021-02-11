#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:58:16 2019

#update 27-07-2019:
This script was run interactively (i.e. in the Spyder IDE). It contains code
for reading in the 153 ChIP-Seq marks, getting the peak data out of them,
formatting that data into bed format so I can use bedIntersect to
find overlaps between sequence sites (i.e. the anchors and in-between) and these peaks,
 and a first version of the random forest feature calculation function (later incorporated
into featureCalcPerLoop_perBEDFILE.py)
 
Later, also incorporated code to make some figures (distribution of deltaCovQ,
loop size distribution), and code to make random anchors and mismatched anchor pairs.
#

Write functions to get chromatin marks for specific genomic intervals.
Also contains code to generate mismatched anchor pairs.

@author: dstoker
"""

import pandas as pd, numpy as np
import os, time, errno
import datetime
from collections import Counter
import subprocess

import re
import gzip
import shutil
import pybedtools as pbed
import pickle 
import sys
import seaborn as sb
import matplotlib.pyplot as plt


def load_obj_fullpath(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
def save_obj(obj, name, directory):
    with open(directory + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


pd.set_option('expand_frame_repr', True)
pd.set_option("display.max_columns", 34)
pbed.set_tempdir("/2TB-disk/Project/Geert Geeven loops data/Data/PyBedTools_tmp")

loopFilePathGM = "/2TB-disk/Project/Geert Geeven loops data/Data/CTCF_LOOPS_GM12878_peakHiC.txt"


loopDataFrameGM = pd.read_csv(loopFilePathGM, sep = "\t")
loopDataFrameGM.head()
loopDataFrameGM.columns.values
loopDataFrameGM.head(2)
loopDataFrameGM["delta"].describe()
loopDataFrameGM["redundantID"].describe() ; len(loopDataFrameGM["redundantID"].unique())
sum(loopDataFrameGM["nr.binID"].duplicated())
loopDataFrameGM.shape
loopDataFrameGM[loopDataFrameGM["maxV4CscorePos"] == 201328130]
k = loopDataFrameGM["chr"].unique().tolist()
k.sort()
k
#is the anchor left always higher than on the right?
loopDataFrameGM[loopDataFrameGM["vp_X1"] < loopDataFrameGM["anchor_X1"]]
loopDataFrameGM

loopDataDescriptionGM = loopDataFrameGM.describe()
loopDataDescriptionGM

loopDataFrameGM[(loopDataFrameGM["totTags.heart"] == 0) & (loopDataFrameGM["totTags.GM12878"] > 0)].shape
loopDataFrameGM.shape

loopDataFrameGM[loopDataFrameGM["heart.vs.GM.signif"] == 1]


loopFilePathHeart  = "/2TB-disk/Project/Geert Geeven loops data/Data/HEART_LOOPS.txt"
loopDataFrameHeart = pd.read_csv(loopFilePathHeart, sep = "\t")
loopDataFrameHeart.head(2)
loopDataFrameHeart[(loopDataFrameHeart["totTags.heart"] == 0) & (loopDataFrameHeart["totTags.GM12878"] > 0)]
#okay so the dataframes don't have quite the same content.
loopDataFrameHeart[(loopDataFrameHeart["totTags.heart"] > 0) & (loopDataFrameHeart["totTags.GM12878"] == 0)].shape

loopDataFrameHeart[loopDataFrameHeart["heart.vs.GM.signif"] == 1]

p = loopDataFrameHeart["chr"].unique().tolist()
p.sort()
p

#
len(loopDataFrameHeart[loopDataFrameHeart["heart.vs.GM.signif"] == 1])
sum(loopDataFrameHeart[loopDataFrameHeart["heart.vs.GM.signif"] == 1]["loopID"].isin(
        loopDataFrameGM[loopDataFrameGM["heart.vs.GM.signif"] == 1]["loopID"]))
sum(loopDataFrameHeart[loopDataFrameHeart["heart.vs.GM.signif"] == 1]["loopID"].isin(
        loopDataFrameGM[loopDataFrameGM["heart.vs.GM.signif"] == 0]["loopID"]))

len(loopDataFrameGM[loopDataFrameGM["heart.vs.GM.signif"] == 1])
sum(loopDataFrameGM[loopDataFrameGM["heart.vs.GM.signif"] == 1]["loopID"].isin(
        loopDataFrameHeart[loopDataFrameHeart["heart.vs.GM.signif"] == 1]["loopID"]))


#Input a dataframe with loops (needs to have column names: chrom, anchorOneStart, anchorOneEnd, anchorTwoStart, anchorTwoEnd)
#also a list of chromatin mark files
                    #per file, get peaks, coverage fractions, etc. for this interval
                     

def listFiles(path : str) -> list:
    r = []
    for root, dirs, files in os.walk(path):
        for name in files:
            r.append(os.path.join(root, name))
    return r

def mkdir_p(path):
    """Make directory if it doesn't exist.
    """
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

#%% functions to get the ChIP-Seq data such that there is one data file per factor.
def find_unique_ChIP_datasets_df(fullTable        : pd.DataFrame                           ,
                                 customAccessions : dict = None                            ,
                                 outputTypeColumn : str  = "Output type"                   ,
                                 assemblyColumn   : str  = "Assembly"                      ,
                                 dateColumn       : str  = "Experiment date released"      ,
                                 targetColumn     : str  = "Experiment target"             ,
                                 accessionColumn  : str  = "Experiment accession"          ,
                                 dataType         : str  = "optimal idr thresholded peaks" 
                                 ) -> pd.DataFrame :
    """
    Selects data files from ENCODE metadata for further processing based on the date of 
    release of the dataset. Allows custom accessions for factors that still have >1 file
    associated even if the latest release is selected. 
    """
    
    addCustomAccessions = False
    addMult             = False
    #Files with different peaks
    subsetTable = fullTable[(fullTable[outputTypeColumn]       == dataType) &
                                 (fullTable[assemblyColumn]    == "GRCh38")]
    
    
    #are there duplicates?
    countsPerFactor           = Counter(subsetTable[targetColumn])
    factorMoreThanOnce        = [factor for factor, num in countsPerFactor.items() if num > 1]
    
    #yes. 
    if(len(factorMoreThanOnce) >= 1):
        
        addMult                     = True
        subsetTableMult             = subsetTable[subsetTable[targetColumn].isin(factorMoreThanOnce)]
        subsetTableMult[dateColumn] = pd.to_datetime(subsetTableMult[dateColumn])
        selectorMaxDateRows         = subsetTableMult.groupby([targetColumn])[dateColumn].transform(
                                                                 max) == subsetTableMult[dateColumn]
        
    #get file names for those factors that are deduplicated by taking the max dates
        uniqueAfterMaxDate         = subsetTableMult[selectorMaxDateRows].drop_duplicates(
                                              subset = targetColumn, keep = False)
    
    #if there are factors that are not made unique by taking the latest datasets, use custom accessions
        notUniqueAfterMaxDate = subsetTableMult[selectorMaxDateRows][subsetTableMult[selectorMaxDateRows].duplicated(
                                        subset = targetColumn, keep = False)]
        
        if(len(notUniqueAfterMaxDate) >= 1):
            addCustomAccessions  = True
            notUniqueMaxDateCust = notUniqueAfterMaxDate[notUniqueAfterMaxDate[accessionColumn].isin(
                                        customAccessions.values())]

    #for those that were not duplicate get the filenames
    
    factorUnique      = list(set(countsPerFactor.keys()).difference(set(factorMoreThanOnce)))
    uniqueDataSets    = subsetTable[subsetTable[targetColumn].isin(factorUnique)]
    
    returnData        = uniqueDataSets
    #concatenate this
    if(addMult == True):
        returnData = pd.concat([returnData, uniqueAfterMaxDate])
        if(addCustomAccessions == True):
            returnData = pd.concat([returnData, notUniqueMaxDateCust])
    
    return(returnData)
    
def get_metadata_chromatinmarks(metaDataFilePath : str, verbose : bool = False,
                                particularAccessions : dict = {"ETV6-human" : "ENCSR626VUC",
                                                               "PAX5-human" : "ENCSR000BHD"}
                                ) -> tuple:
    """
    Open the metadata. Get filenames for files to open, along with the chromatin mark measured.
    Return as dictionary {chromatin mark : file name}. Since many ChIP-Seq marks are measured multiple
    times, we take the latest entry to get a unique file. For ETV6 and PAX5, there are still duplicates in this case. 
    Manual selection was made based on antibody type for PAX5, for ETV6, the dataset that was ´under investigation
    for possible contamination' was not used.
    """
    
    containingDir     = re.match(".+?(?=.metadata.tsv)", metaDataFilePath)[0] + "/"
    metaData          = pd.read_csv(metaDataFilePath, sep = "\t")
    
    
    #sort these files to get one unique dataset (file) per ChIP factor measured
    normalDataUnique  = find_unique_ChIP_datasets_df(metaData, customAccessions = particularAccessions)
    
    #process and catalogue files that are only mapped to hg19, and not to GRCh38
    subsetOptimalData = metaData[(metaData["Output type"] == "optimal idr thresholded peaks") &
                                 (metaData["Assembly"]    == "GRCh38")]
    
    subsetHg19Only    = metaData[(metaData["Output type"] == "optimal idr thresholded peaks") &
                                 (metaData["Assembly"]    == "hg19")][~metaData["Experiment target"].isin(
                                         subsetOptimalData["Experiment target"])]
                                                              
    #Okay so there are six of those. Convert them to Grch38/hg38 using CrossMap if the files do not already exist.
    #to do that, unzip, write to file, open those files, liftover.
    for file in subsetHg19Only["File accession"]:
        if not(os.path.isfile(containingDir + file + "_CrossMapGRCh38.bed")):
            with gzip.open(containingDir + file + ".bed.gz", 'rb') as f_in:
                with open(containingDir + file + ".bed", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
            print(file)
            subprocess.run(["CrossMap.py", "bed", containingDir + "CrossOverChainFile/hg19ToHg38.over.chain.gz",
                            containingDir + file + ".bed", containingDir + file + "_CrossMapGRCh38.bed"])
    #Done. add them to dataframe.
    subsetHg19Only["Assembly"] = "GRCh38_fromHg19_CrossMapped"
    subsetHg19Only["File accession"] = subsetHg19Only["File accession"] + "_CrossMapGRCh38.bed"
    
    #Files with different peak calling files (these files only have ´replicated peaks' , not idr-thresholded peaks)
    differentPeakCallingFiles = find_unique_ChIP_datasets_df(metaData, dataType = "replicated peaks")
    
    #add data together
    totalData                   = pd.concat([normalDataUnique,
                                             differentPeakCallingFiles], axis = 0)
    #set file name to include extension
    totalData["File accession"] = totalData["File accession"] + ".bed.gz"
    #add in hg19_to_grch38data

    totalData = pd.concat([totalData, subsetHg19Only], axis = 0)
    
    if(verbose == True):
        print("Metadata all factors to read in:")
        print(totalData.head())
        print(totalData.tail())
    
    fileNameDictionary = {}    
    for index, row in totalData.iterrows():
        fileNameDictionary[row["Experiment target"]] = (row["File accession"])
    
    return (fileNameDictionary, totalData)

#%%
#test
metaDataFilePath = "/2TB-disk/Project/Geert Geeven loops data/Data/EncodeChipSeqDataGrCH38_bedNarrowPeak_and_bedBroadPeak_209datasets_withisogenicreplicates_downloaded24_01_2019/metadata.tsv"

fileNameDictionary, totalMetaDataPerFactor = get_metadata_chromatinmarks(metaDataFilePath,
                                                               verbose = False)

fileNameDictionary
totalMetaDataPerFactor
totalMetaDataPerFactor[totalMetaDataPerFactor["Output type"] == "replicated peaks"]

#get the accessions used for a supp. table:
mooieLijst = [re.search("^[A-Za-z0-9]+\d+[^_.]+", entry)[0] for entry in fileNameDictionary.values()]
pd.DataFrame({"factorNames" : list(fileNameDictionary.keys()),
              "accessionsUsed" : mooieLijst}).to_csv("/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/figuresFinalClassifiers_16_06_2019/fileAccessionsEnsembl.csv")


#I see in openoffice calc that there are more than 137 unique factors (157 in fact). I am thus missing some. Why?
#compare with some of them whether they are present.
"E2F4-human" in fileNameDictionary.keys()
"PML-human" in fileNameDictionary.keys()


#okay so the thing is that these files are not in GrCH38. Still
#see whether I can resolve all data by loading the most recent Chip Seq files

#Nope, that only yields 6 more. 137 + 6 = 143 /= 157. Where is the rest?
totalNamesNow = list(fileNameDictionary.keys())
totalNamesNow
len(totalNamesNow)

#what is missing
allUniqueNames = pd.read_csv("/2TB-disk/Project/Geert Geeven loops data/Data/EncodeChipSeqDataGrCH38_bedNarrowPeak_and_bedBroadPeak_209datasets_withisogenicreplicates_downloaded24_01_2019/Unique_factor_names.txt",
                             header = None)
allUniqueNames[~allUniqueNames[0].isin(totalNamesNow)][0].tolist()

#Okay so these are marks that either have only one or two files, or marks that have a different file structure.
#will add the marks in the function. 


def get_chromatin_factor_dataframe(chromatinMarkFiles : dict,
                                   dataDir            : str  = "/2TB-disk/Project/Geert Geeven loops data/Data/EncodeChipSeqDataGrCH38_bedNarrowPeak_and_bedBroadPeak_209datasets_withisogenicreplicates_downloaded24_01_2019/",
                                   verbose            : bool = False) -> pd.DataFrame:  
    """
    Reads in all chromatin factor files. Outputs one dataframe with all peaks for all factors.
    """
    peaksDict = {}
    for factor, filePath in chromatinMarkFiles.items():
        peaksList = []
        if(filePath.endswith(".gz")):
            with gzip.open(dataDir + filePath, "r") as file:
              for line in file:
                  line = line.decode()
                  #print(line)
                  peaksList.append(line.rstrip().split("\t"))
        else:
            with open(dataDir + filePath, "r") as file:
                for line in file:
                    #print(line)
                    peaksList.append(line.rstrip().split("\t"))
        
        dataFrame = pd.DataFrame(peaksList)
        if(verbose):
            print(dataFrame.head(10)); print(factor)
        dataFrame.set_axis(["chrom", "start", "stop", "name", "score", "strand", "signalValue",
                                "pValueNegativeLog10", "qValueNegativeLog10", "pointPeak"],
                                   axis='columns', inplace=True)
        dataFrame["Factor name"] = factor 
        peaksDict[factor] = dataFrame    
        
    #now concatenate this into one large dataFrame for searching in.
    totalDataFrameFactors = pd.concat(peaksDict.values())
    totalDataFrameFactors = totalDataFrameFactors[["Factor name", "chrom", "start", "stop",
                                                   "name", "score", "strand", "signalValue",
                                                   "pValueNegativeLog10", "qValueNegativeLog10", "pointPeak"]]
    
    #make sure data types are correct
    columnsNumeric = ["start", "stop", "score", "signalValue",
                      "pValueNegativeLog10", "qValueNegativeLog10", "pointPeak"]
    for col in columnsNumeric:
        print("making a number out of column: " + col)
        totalDataFrameFactors[col] = pd.to_numeric(totalDataFrameFactors[col])
    
    
    totalDataFrameFactors["pointPeakGenomicCoordinate"] = totalDataFrameFactors["start"] + \
                                                          totalDataFrameFactors["pointPeak"]
    totalDataFrameFactors.sort_values(by = ["chrom", "start", "stop"], ascending = True, inplace = True)
    
    return(totalDataFrameFactors)


allPeaksInData = get_chromatin_factor_dataframe(fileNameDictionary)        

allPeaksInData.head(2)
      
def check_column_uniques(dataFrame : pd.DataFrame) -> dict:
    """
    Saves list of uniques per column of a pandas dataframe to a dictionary with column names as keys
    """
    colDict = {}
    for column in dataFrame.columns:
        colDict[column] = dataFrame[column].unique().tolist()
    return(colDict)
      
uniquesPerColumnPeakData = check_column_uniques(allPeaksInData)        
uniquesPerColumnPeakData["chrom"]

uniquesPerColumnLoopDataGM    = check_column_uniques(loopDataFrameGM)
uniquesPerColumnLoopDataGM["loopCall"]
uniquesPerColumnLoopDataHeart = check_column_uniques(loopDataFrameHeart)
uniquesPerColumnLoopDataHeart["loopCall"]

def load_and_filter_loops(loopsFilePath : str) -> pd.DataFrame:
    #what this needs to do:
    #-read in data
    #-Remove top and bottom 2.5% of data based on maxVirtual4C to remove outliers
    #that occur because of reads mapping to anchors next to removed other anchors.
    #-sort on maxV4Cscore, descending
    #-filter out duplicates in nr.binID. Because higher maxV4Cscore means stronger loop,
    #this results in a list of the strongest loops between any pair of bins.
    #calculate delta: CovQNORMGM-CovQNORMHeart. top of this last: strongest GM12878 loops. Bottom:
    #strongest heart loops.
    
    
    loopsDF = pd.read_csv(loopsFilePath, sep = "\t")
    
    #filter top and bottom of data
    
    filterBound     = loopsDF["maxV4Cscore"].quantile([0.025, 0.975]).tolist()
    loopsDFFiltered = loopsDF[(loopsDF["maxV4Cscore"] >= filterBound[0]) &
                              (loopsDF["maxV4Cscore"] <= filterBound[1])]
    
    loopsDFFiltered.sort_values(by = "maxV4Cscore", ascending = False, inplace = True)
    loopsDFFiltered.drop_duplicates(subset = "nr.binID", keep = "first", inplace = True)
    
    loopsDFFiltered["calculatedDeltaCovQ"] =  loopsDFFiltered["covQ.GM12878.norm"] - loopsDFFiltered["covQ.heart.norm"]
    loopsDFFiltered.sort_values(by = "calculatedDeltaCovQ", ascending = False, inplace = True)
    
    return loopsDFFiltered

def get_loops(filteredLoopDataFrame: pd.DataFrame,
              amountOfLoops: int = 20000) -> tuple:
    
    #get an X amount of loops from each set, based on the dataframe from load_and_filter_loops()
    nRows = len(filteredLoopDataFrame)
    if(amountOfLoops > 1/3* nRows):
        print("Warning, you are selecting more than 1/3rd of loops for each case")
        if(amountOfLoops >= 1/2* nRows):
            print("There will now be overlap in loops, as the amount requested > 0.5 nr of rows")
    loopsGM    = filteredLoopDataFrame[0:amountOfLoops]
    
    loopsHeart = filteredLoopDataFrame[nRows-1-amountOfLoops:nRows-1]
    
    return (loopsGM, loopsHeart)


#Now: take per set the position of the V4CmaxScorePos and the middle of the vp_X1 and vp_X2:
    #these represent the middle of the found loop anchors. Also take the chr, and label with
    #heart or GM.
    
def get_loop_data_for_chromatin_marks(tupleOfGMAndHeart: tuple) -> tuple:
    if(len(tupleOfGMAndHeart) > 2):
        print("incorrect input. Use the tuple generated by get_loops()")
    
    loopsGM, loopsHeart = tupleOfGMAndHeart[0], tupleOfGMAndHeart[1]
    
    def make_sub_df(loopsDF: pd.DataFrame, tissueType: str) -> pd.DataFrame:
        
        columnsToSelect = ["chr", "vp_X1", "vp_X2", "maxV4CscorePos"]
        tempDF = loopsDF[columnsToSelect]
        #get all the data into a separate DF if it is ever needed
        metaData = loopsDF
        metaData["loopTissue"] = tissueType
        metaData = metaData[columnsToSelect + ["loopTissue"] +
                            [col for col in metaData.columns.values.tolist() if col not in
                             columnsToSelect]]
        metaData = metaData.loc[:, ~metaData.columns.duplicated()]
        tempDF["vp_middle"]  = tempDF["vp_X1"] + 5000
        tempDF["loopTissue"] = tissueType
        dF     = tempDF[["chr", "vp_middle", "maxV4CscorePos", "loopTissue"]]
        return(dF, metaData)
    
    finalLoopsGM, finalLoopsHeart = make_sub_df(loopsGM, "GM12878"), make_sub_df(loopsHeart, "Heart")
    returnDFLoops                 = pd.concat([finalLoopsGM[0], finalLoopsHeart[0]])
    returnDFLoops.sort_values(inplace = True, by = "chr", ascending = True)
    returnDFMetaData              = pd.concat([finalLoopsGM[1], finalLoopsHeart[1]])
    returnDFMetaData.sort_values(inplace = True, by = "chr", ascending = True)

    return(returnDFLoops, returnDFMetaData)


#getLoops
    
loopsFilePath = "/2TB-disk/Project/Geert Geeven loops data/Data/CTCF_LOOPS_GM12878_peakHiC.txt"

filteredLoops     = load_and_filter_loops(loopsFilePath)
loopsGMAndHeart   = get_loops(filteredLoops)
positionDataFrame = get_loop_data_for_chromatin_marks(loopsGMAndHeart)
metaDataPositionDataFrame = positionDataFrame[1]
positionDataFrame = positionDataFrame[0]
  
    

chromatinMarkDataFrame = allPeaksInData
chromatinMarkDataFrame.head()
otherIntervals         = [10000, 2000, 300]

positionDataFrame.head()

#output for HPC
#save_obj(positionDataFrame,
#         "positionDataFrame",
#         "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/FilesForHPCProcessing/ChromatinAndAnchorDFPickles_14_03_2019/")
#
#save_obj(chromatinMarkDataFrame,
#         "chromatinMarkDataFrame",
#         "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/FilesForHPCProcessing/ChromatinAndAnchorDFPickles_14_03_2019/")

#want a function that returns, for different intervals around loop anchors,
#several measures of ChIP-Seq. Such as: amount of point peaks in the region,
#distance of nearest point peak, enrichment value, etc.
#note intervals should be divisible by 2
#def make_features_random_forest(positionDataFrame      : pd.DataFrame,
#                                chromatinMarkDataFrame : pd.DataFrame,
#                                *otherIntervals) -> pd.DataFrame:
#    #get only data on chromosomes that are in the loopDataFrame
#    chromatinMarkDataFrame = chromatinMarkDataFrame[chromatinMarkDataFram["chrom"].isin(
#                                                        positionDataFrame["chr"].unique().tolist())]
#    
#    #sort by chromosome for subsetting
#    groupedLoopsDF = positionDataFrame.groupby(by = "chr")
#    groupedChromDF = chromatinMarkDataFrame.groupby(by = "chrom")
#    #groupedChromDF.count()
#    groupedLoopsDF.count()
#
#    for chromosome, allCorrespRows in groupedLoopsDF:
#        print(allCorrespRows)
#        #get correct chrom, make groups for all chrom features
#        currentChrChromMarkDF = groupedChromDF.get_group("chromosome").groupby("Factor name")
#        
#        for index, row in allCorrespRows.iterrows():
#            
#            leftLoopAnchor, rightLoopAnchor = row["vp_middle"], row["maxV4CscorePos"]
#            chrom, tissue = row["chr"], row["loopTissue"]
#            
#            #right part of anchor starts on the anchor position, inclusive that position. Left part ends 1 before it.
#            #Left part anchor starts on anchorPos - interval/2, inclusive. Right part of anchor ends on 
#            #anchorPos + interval/2 -1 (i.e. anchorPos + interval/2, exclusive)
#            subsetLeftLeft  = currentChrChromMarkDF[((currentChrChromMarkDF["start"] >= (leftLoopAnchor - otherIntervals[0]/2)) &
#                                                     (currentChrChromMarkDF["start"] <  leftLoopAnchor)) |
#                                                    ((currentChrChromMarkDF["end"]    >= (leftLoopAnchor - otherIntervals[0]/2)) & 
#                                                     (currentChrChromMarkDF["end"]    < leftLoopAnchor))
#                                                   ]
#            #-1 because of the indexing, anchors start from the left and are inclusive of that position, exclusive of the last pos
#            subsetLeftRight = currentChrChromMarkDF[(currentChrChromMarkDF["start"] >= (leftLoopAnchor + otherIntervals[0]/2 - 1)) |
#                                                   (currentChrChromMarkDF["end"]    >= (leftLoopAnchor + otherIntervals[0]/2 - 1))]
#        
            
##While typing the above I figured out that working with BedTools is probably much better      
##I can easily find intersections there and report them.
            
#plan now:
#make bed file of the anchor positions: 
            #-encode each feature as chrom start end, with name being one of:
            #left anchor left, left anchor right, in-between, right anchor left, right anchor right
            #save as .bed

#compare that file with either a. the original bed files for chromatin marks or
            #b. save the total bed file for all chromatin marks. 

#Might need to include checks that left anchor + 0.5* interval does not overlap with right anchor - 0.5 * interval? 
            
intervals = [2500, 10000]
def make_bed_format_anchors(positionDataFrame  : pd.DataFrame,
                            calculateInBetween : bool = True,
                            *intervals) -> list:
    
    if(len(intervals) < 1):
        print("You need at least one interval within which to compute features")
        return(None)
    
    combinedBEDFileList = []
    
    for interval in list(intervals):
        
        #get every position inclusive for bed files.
        #every anchor is divided into left and right. There is also data in between.
        positionDFCopy = positionDataFrame.copy()
        
        #flip values such that the lower genomic coordinate is always on the left.
        positionDFCopy["leftmost_coordinate" ]  = positionDFCopy[["vp_middle", "maxV4CscorePos"]].min(axis = 1)
        positionDFCopy["rightmost_coordinate"]  = positionDFCopy[["vp_middle", "maxV4CscorePos"]].max(axis = 1)
        
        positionDFCopy["interval"             ] = interval
        positionDFCopy["leftAnchorLeftStart"  ] = positionDFCopy["leftmost_coordinate" ] - interval/2
        positionDFCopy["leftAnchorLeftStop"   ] = positionDFCopy["leftmost_coordinate" ] - 1
        positionDFCopy["leftAnchorRightStart" ] = positionDFCopy["leftmost_coordinate" ]
        positionDFCopy["leftAnchorRightStop"  ] = positionDFCopy["leftmost_coordinate" ] + interval/2 -1
        positionDFCopy["rightAnchorLeftStart" ] = positionDFCopy["rightmost_coordinate"] - interval/2
        positionDFCopy["rightAnchorLeftStop"  ] = positionDFCopy["rightmost_coordinate"] - 1
        positionDFCopy["rightAnchorRightStart"] = positionDFCopy["rightmost_coordinate"]
        positionDFCopy["rightAnchorRightStop" ] = positionDFCopy["rightmost_coordinate"] + interval/2 -1
        if(calculateInBetween == True):
            positionDFCopy["inbetweenStart"   ] = positionDFCopy["leftAnchorRightStop" ] + 1
            positionDFCopy["inbetweenStop"    ] = positionDFCopy["rightAnchorLeftStart"] - 1 
        positionDFCopy["uniqueLoopIDPerType"  ] = positionDFCopy.groupby("loopTissue").cumcount()

        tissueTypeLoopID    = [str(row["loopTissue"]) + "_" + str(row["uniqueLoopIDPerType"]) for index, row in positionDFCopy.iterrows()]

        #now make for each feature a bed-file format which is chr start end name
        
        #first: define whether inbetween features exist or not:
        if(calculateInBetween == True):
            listFeatureLocations = ["leftAnchorLeft", "leftAnchorRight", "inbetween", "rightAnchorLeft", "rightAnchorRight"]
        else:
            listFeatureLocations = ["leftAnchorLeft", "leftAnchorRight", "rightAnchorLeft", "rightAnchorRight"]
        
        for featurePosType in listFeatureLocations:
            columnsToGet        = [col for col in positionDFCopy.columns.values.tolist() if featurePosType in col]
            separateDF          = positionDFCopy[["chr"] + columnsToGet]
            #format: featurePosType_interval_tissueType_loopID
            nameList            = [featurePosType + "_" + str(interval) + "_" + elmnt for elmnt in tissueTypeLoopID]
            separateDF["name"]  = nameList
            separateDF.columns  = ["chr", "start", "stop", "name"]
            separateDF["start"] = pd.to_numeric(separateDF["start"], downcast = "integer")
            separateDF["stop" ] = pd.to_numeric(separateDF["stop"], downcast = "integer")
            separateDF.sort_values(by = ["chr", "start", "stop"], inplace = True)
            combinedBEDFileList.append(separateDF)
            
#okay now I have a list of dataframes in BED format. I could pass each through bedtools, or make one huge bed file.
            #I will start with the huge bed file and see how it goes.
            
    #finalDataFrameForBED = pd.concat(combinedBEDFileList)
    
    return(combinedBEDFileList)
    
chicken = make_bed_format_anchors(positionDataFrame, True, 2500,10000)

len(chicken)
chicken[0].head()
chicken[1].head()
chicken[2].head()
chicken[3].head()
chicken[4].head()
chicken[0].head().iloc[0,3]
     


def generate_intersect_BEDs(listOfBEDFormatAnchors: list,
                            chromatinMarkDataFrame: pd.DataFrame,
                            parentDirectory       : str = "/2TB-disk/Project/Geert Geeven loops data/Data/EncodeChipSeqDataGrCH38_bedNarrowPeak_and_bedBroadPeak_209datasets_withisogenicreplicates_downloaded24_01_2019/"
                            ) -> list:
    
    listIntersectBEDs = []
    chromatinMarkDF = chromatinMarkDataFrame.copy()
    
    dateFile  = str(datetime.datetime.today()).split()[0]
    directoryName = parentDirectory + "BedFilesForIntersectChromatinMarksLoops_" + dateFile
    #make directory if necessary, otherwise write to it
    mkdir_p(directoryName)   
    
    if("Factor name" in chromatinMarkDF.columns.values.tolist()):
        chromatinMarkDF["name"] = chromatinMarkDF["Factor name"] 
        chromatinMarkDF.drop(["Factor name", "pointPeakGenomicCoordinate"], axis = 1, inplace = True) 
    chromatinMarkDF.sort_values(by = ["chrom", "start", "stop"], inplace = True)
    bedChromatinMark = pbed.BedTool.from_dataframe(chromatinMarkDF)
    
    for element in listOfBEDFormatAnchors:
        anchorBED = pbed.BedTool.from_dataframe(element)
        nameFile  = element.iloc[0,3].split("_")[0] + "_" + element.iloc[0,3].split("_")[1]
        
            
        intersectBED = anchorBED.intersect(bedChromatinMark, wao = True,
                                           sorted = True,
                                           output = directoryName + "/pybedtoolsIntersect_wao_" + \
                                           nameFile + ".bed")
            
        listIntersectBEDs.append(intersectBED)
            
    return(listIntersectBEDs)

kaas = generate_intersect_BEDs(chicken, chromatinMarkDataFrame)
kaas[0].head() #leftanchorleft 2500


kaas[1].head() #leftanchorright 2500
kaas[2].head() #inbetween 2500
kaas[3].head() #right anchor left 2500
kaas[4].head() #right anchor right 2500

kaas[5].head()
kaas[6].head()
kaas[7].head()
len(kaas)
chicken[0]

len(kaas)
freek = kaas[0].to_dataframe(
        names = [
                "chromosomeArea"     , "areaStart"          , "areaEnd"            , "areaType",
                "chromosomeFactor"   , "factorStart"        , "factorEnd"          ,
                "factorName"         , "factorScore"        , "factorStrand_Unused",
                "factorSignalValue"  , "pValueNegativeLog10",
                "qValueNegativeLog10", "pointPeak"          , "overlapBasePairs"
                ]
                            )

freek

berend = kaas[5].to_dataframe(
        names = [
                "chromosomeArea"     , "areaStart"          , "areaEnd"            , "areaType",
                "chromosomeFactor"   , "factorStart"        , "factorEnd"          ,
                "factorName"         , "factorScore"        , "factorStrand_Unused",
                "factorSignalValue"  , "pValueNegativeLog10",
                "qValueNegativeLog10", "pointPeak"          , "overlapBasePairs"
                ]
                            )
berend.head(10)

#now combine all this data into features for a random forest

listChromatinFactors = chromatinMarkDataFrame["Factor name"].unique().tolist()

testing = berend.head(10).melt(id_vars = ["areaType"]).groupby("areaType")
testing.head()

test2 = berend.head(100).groupby("areaType")

test2.head()



#def make_dataframe_BED(BEDobject) -> pd.DataFrame:
#    """
#    Changes pybedtools like object to pd.DataFrame.
#    """
#    
#    dataFrame = BEDobject.to_dataframe(
#    names = [
#                "chromosomeArea"     , "areaStart"          , "areaEnd"            , "areaType",
#                "chromosomeFactor"   , "factorStart"        , "factorEnd"          ,
#                "factorName"         , "factorScore"        , "factorStrand_Unused",
#                "factorSignalValue"  , "pValueNegativeLog10",
#                "qValueNegativeLog10", "pointPeak"          , "overlapBasePairs"
#                ]
#                          )
#
#        
#    return(dataFrame)    


#comment 17-05-2019: this is what I finally used. The per-loop saving produces far too
#many files. 
def make_dataframe_BED_save(BEDObjectList, directoryToSaveTo, fileNameAddition = None):
    
    for thingy in BEDObjectList:
        
        dataFrame = thingy.to_dataframe(
                names = [
                "chromosomeArea"     , "areaStart"          , "areaEnd"            , "areaType",
                "chromosomeFactor"   , "factorStart"        , "factorEnd"          ,
                "factorName"         , "factorScore"        , "factorStrand_Unused",
                "factorSignalValue"  , "pValueNegativeLog10",
                "qValueNegativeLog10", "pointPeak"          , "overlapBasePairs"
                ]
                          )
        
        firstRow       = dataFrame.iloc[0, :]
        split = firstRow["areaType"].split("_")
        now = str(datetime.datetime.today()).split()[0]
        nameToGiveFile = split[0] + "_" + split[1] + "_" + now
        if fileNameAddition is None:
            pass
        else:
            nameToGiveFile += ("_" + fileNameAddition) 
        if (os.path.exists(directoryToSaveTo)):
            pass
        else:
            os.makedirs(directoryToSaveTo)
        
        dataFrame.to_csv(directoryToSaveTo + nameToGiveFile + ".csv")
        
 


#def make_dataframe_BED_save_per_loop_csv(BEDobject, directoryToSaveTo):
#    
#    dataFrame = BEDobject.to_dataframe(
#    names = [
#                "chromosomeArea"     , "areaStart"          , "areaEnd"            , "areaType",
#                "chromosomeFactor"   , "factorStart"        , "factorEnd"          ,
#                "factorName"         , "factorScore"        , "factorStrand_Unused",
#                "factorSignalValue"  , "pValueNegativeLog10",
#                "qValueNegativeLog10", "pointPeak"          , "overlapBasePairs"
#                ]
#                          )
#    for name, group in dataFrame.groupby("areaType"):
#        #print(name)
#        fileName = name + ".csv"
#        group.to_csv(directoryToSaveTo + fileName)
#        

#testing the functionality of this function

#directoryToSaveToFeatureCalc =  "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/FilesForHPCProcessing/BEDIntersectDataFramesPerLoop/"       
directoryToSaveToFullBEDFiles =   "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/FilesForHPCProcessing/BEDIntersectDataFrames/"   
      
#for bedIntersectFile in kaas:
#    
#    make_dataframe_BED_save_per_loop_csv(bedIntersectFile, directoryToSaveToFeatureCalc)


make_dataframe_BED_save(kaas, directoryToSaveToFullBEDFiles)

#also output the list of chromatin factors (necessary for feature calc)

def create_chromatinmarktxt_in_bedfiledir(BEDFileDir, chromatinMarkDataFrame):
    with open(BEDFileDir + "chromatinMarks.txt", "w+") as f:
        for entry in chromatinMarkDataFrame["Factor name"].unique().tolist():    
            f.write(entry + "\n")

with open(directoryToSaveToFeatureCalc + "chromatinMarks.txt", "w+") as f:
    for entry in chromatinMarkDataFrame["Factor name"].unique().tolist():    
        f.write(entry + "\n")

with open(directoryToSaveToFullBEDFiles + "chromatinMarks.txt", "w+") as f:
    for entry in chromatinMarkDataFrame["Factor name"].unique().tolist():    
        f.write(entry + "\n")        
        
    

        




#how many values are we dealing with here?
    kaas[0].count()
    kaas[1].count()

######################
    ##################
    ################## BELOW IS THE UPDATED FUNCTION THAT SHOULD SPEED UP
    ##################  Note that this function is now incorporated into featureCalcPerLoop_perBEDFILE.py for running on the HPC.
######################

def make_random_forest_features_2(listIntersectBEDFiles: list,
                               chromatinMarkDataFrame: pd.DataFrame) -> pd.DataFrame:
    
    listChromatinFactors = chromatinMarkDataFrame["Factor name"].unique().tolist()
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
    for bedFile in listIntersectBEDFiles:
        
        #get the names only once by  using this bool. Since every Bed file is of a specific
        #anchor position, those variables only need to be set on the first iteration
        #through all loops.
        first = True
        
        print("working on file: " + str(counter + 1) + " out of " + str(lenListFiles))
        print("---")
        counter += 1
        
        DataFrameBED = make_dataframe_BED(bedFile)
        
        groupedByLoop = DataFrameBED.groupby("areaType")
        
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
                #Note, the below is wrong for in-between features!
                if(re.search("inbetween", loopType)):
                    totalOverlapFraction = groupedByChromFactor["overlapBasePairs"].sum() / \
                    (groupedByChromFactor["AreaEnd"] - groupedByChromFactor["AreaStart"])
                else:
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

    return([dictFeatures, namesFeatureColumns])

        
        
ingekorteLijstIntersectsZonderInBetween = kaas.copy()
ingekorteLijstIntersectsZonderInBetween.pop(2)
ingekorteLijstIntersectsZonderInBetween.pop(6)
len(ingekorteLijstIntersectsZonderInBetween)
hoeveelheidInterSects = [entry.count() for entry in ingekorteLijstIntersectsZonderInBetween]
print(hoeveelheidInterSects)
print(sum(hoeveelheidInterSects))

cheesyPizza = make_random_forest_features_2(ingekorteLijstIntersectsZonderInBetween, chromatinMarkDataFrame)


#### save the run with intervals around anchors of either 2500 or 10000 to a pickle
pickleFeaturesOut = open("/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/SavedFeatureDicts/14_03_2019_firstTryFeatures.pickle", "wb")
pickle.dump(cheesyPizza, pickleFeaturesOut)
pickleFeaturesOut.close()


with open("/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/SavedFeatureDicts/14_03_2019_firstTryFeatures.pickle", "rb") as file:
    cheesyPizza = pickle.load(file)


cheesyPizza[1][0]
    
#okay so:
        #get all the different keys (anchor sites) for one loop ID
        #for each anchor site, make a list of lists with values for that site.
        #then, loop through all the loops with this, adding to each list, also keeping a list
        #of IDs
        #make a dataframe for each anchor site, with ID as the index. Join the separate dataframes
        #on index.


#get an arbitrary element from the dictionary, get the values, and there is another dictionary,
        #whose keys I want, as they denote for what anchor the feature values are
        
        
def format_feature_data(featureGenerationOutput : tuple) -> dict:
    
    anchors = list(next(iter(featureGenerationOutput[0].values())).keys())
    loopIDs = list(featureGenerationOutput[0].keys())
    dictionaryAnchorsForDataFrame = {anchor: [] for anchor in anchors}
    
    for entry in featureGenerationOutput[0].values():
        for anchor, features in entry.items():
            dictionaryAnchorsForDataFrame[anchor].append(features)
    
    numpyArrayPerAnchor = {anchor: np.array(featureList) for anchor, featureList in dictionaryAnchorsForDataFrame.items()}
    #numpyArrayPerAnchor.keys()
    namesRightOrder = [ nameList for nameList in featureGenerationOutput[1] for
                       anchorName in numpyArrayPerAnchor.keys() if nameList[0].startswith(anchorName)]    
    namesRightOrder  = [names for sublist in namesRightOrder for names in sublist]  
    #namesRightOrder
    #len(namesRightOrder)
    
    totalNumpyArray = np.hstack((numpyArrayPerAnchor.values()))
    #totalNumpyArray.shape
    loopLabels = np.array([loop.split("_")[1] for loop in loopIDs])
    
    return({"featureArray" : totalNumpyArray, "classLabelArray" : loopLabels,
            "namesFeatureArrayColumns" : namesRightOrder})

formattedFeatureData =   format_feature_data(cheesyPizza)  







##############################################################################################
##  show the distribution of delta-covQ (the GM-specificity or heart-specificity of a loop) ##
##############################################################################################

deltaToPlot = filteredLoops["calculatedDeltaCovQ"]

import seaborn as sb

distributionPlotDelta = sb.distplot(deltaToPlot)
len(deltaToPlot)

#also as a scatter?
xValuesScatterCovQ = range(0, len(deltaToPlot), 1)
usageInClassifier = ["top20000GMLoops"] * 20000 + ["notUsedInClassifierForNow"] * (len(deltaToPlot) - 40000) + ["top20000HeartLoops"] * 20000

a4_dims2 = (15, 7.5)
fig2, ax2 = plt.subplots(figsize=a4_dims2)

scatterPlotDelta = sb.scatterplot(xValuesScatterCovQ, deltaToPlot, hue = usageInClassifier, style = usageInClassifier,
               alpha = 0.2, markers = ["*", ".", "*"], ax = ax2, linewidth = 0)



#make plots of size distribution of loops, per tissue (heart, GM) and per chromosome

GMSpecificLoops = loopsGMAndHeart[0]
GMSpecificLoops["loopType"] = "GM12878"
HeartSpecificLoops = loopsGMAndHeart[1]
HeartSpecificLoops["loopType"] = "Heart"
GMSpecificLoops.head()

combinedLoopsDataFrameForLoopSizePlotting = pd.concat([GMSpecificLoops, HeartSpecificLoops])
combinedLoopsDataFrameForLoopSizePlotting.shape
combinedLoopsDataFrameForLoopSizePlotting.sort_values(["loopType", "chr"], ascending = [True, True], inplace = True)
combinedLoopsDataFrameForLoopSizePlotting.set_index(combinedLoopsDataFrameForLoopSizePlotting["loopType"], inplace = True)
heartSubset = combinedLoopsDataFrameForLoopSizePlotting.filter(
            like = "Heart", axis = "index")
heartSubset["numberingForIndex"] = range(0, len(heartSubset))
heartSubset.set_index("numberingForIndex", inplace = True)

GMSubset = combinedLoopsDataFrameForLoopSizePlotting.filter(
            like = "GM12878", axis = "index")

GMSubset["numberingForIndex"] = range(0, len(GMSubset)) #this is to have numeric indexes for deletion later
#for equalising the numbr of loops per loop size bin.
#Also, this allows me to remove those indices from the feature array easily, by simply deleting
#feature rows with corresponding loopIDs (number from 0-19999) + _CellType (Heart/GM12878)
GMSubset.set_index("numberingForIndex", inplace = True)
f3, axes3 = plt.subplots(6, 4, figsize=(90, 30), sharex=True, sharey=True)
widthPlot = 5000
for i, feature in enumerate(combinedLoopsDataFrameForLoopSizePlotting["chr"].unique()):
    print(feature)
    #if needs to be zoomed in, do that
#    if ("intersectsPerFactor" in feature or "Overlap" in feature or "sumPointPeaksInInterval" in feature):
#        axes2[i%8, i//8].set_xlim(-2, 3)
#        axes2[i%8, i//8].set_ylim(0, 2000)
    
    heartChromSubset = heartSubset[heartSubset["chr"] == feature]
    
    GMChromSubset = GMSubset[GMSubset["chr"] == feature]
    
    binsGM = np.histogram(GMChromSubset["dist"], bins = 80, range = (0, 1000000))
    binsHeart = np.histogram(heartChromSubset["dist"], bins = 80, range = (0, 1000000))
    
    #binLabels
    binLabels = [str(int(value)) + "-" + str(int(binsGM[1][index+1])) for index, value in enumerate(binsGM[1]) if index != 80]
    #white bg
    axes3[i%6, i//6].set_facecolor("white")
    
    rects1 = axes3[i%6, i//6].bar(binsGM[1][1:], binsGM[0], width = widthPlot, color = "red", edgecolor = "gray")
    rects2 = axes3[i%6, i//6].bar(binsHeart[1][1:] + widthPlot, binsHeart[0], width = widthPlot, color = "skyblue", edgecolor = "gray")
    axes3[i%6, i//6].set_xticks(binsGM[1][1:] + widthPlot / 2)
    axes3[i%6, i//6].set_xticklabels(binLabels, rotation = 60)
    axes3[i%6, i//6].legend( (rects1[0], rects2[0]), ('GM', 'Heart') )
    axes3[i%6, i//6].tick_params(axis='both', which='major', pad=0, grid_alpha = 0.7, color = "black", grid_color = "black")
#    sb.distplot(heartChromSubset["dist"] ,
#            color="skyblue", ax=axes3[i%6, i//6], bins = 80, kde = False,
#            hist_kws=dict(alpha=0.4), rug = False)
#    sb.distplot(GMChromSubset["dist"] ,
#            color="red", ax=axes3[i%6, i//6], bins = 80, kde = False,
#            hist_kws=dict(alpha=0.4), rug = False)
    axes3[i%6, i//6].set_xlabel("Loop size distribution " + feature)

f3.savefig("/2TB-disk/Project/Documentation/Meetings/06-05-2019/loopSizeDistributionPerChromosomePerLoopType_06-05-2019_80bins.png")

f4, axes4 = plt.subplots(6, 4, figsize=(90, 30), sharex=True, sharey=True)
widthPlot = 5000
for i, feature in enumerate(combinedLoopsDataFrameForLoopSizePlotting["chr"].unique()):
    print(feature)
    #if needs to be zoomed in, do that
#    if ("intersectsPerFactor" in feature or "Overlap" in feature or "sumPointPeaksInInterval" in feature):
#        axes2[i%8, i//8].set_xlim(-2, 3)
#        axes2[i%8, i//8].set_ylim(0, 2000)
    
    heartChromSubset = heartSubset[heartSubset["chr"] == feature]
    
    GMChromSubset = GMSubset[GMSubset["chr"] == feature]
    
    binsGM    = np.histogram(GMChromSubset["dist"], bins = 40, range = (0, 1000000))
    binsHeart = np.histogram(heartChromSubset["dist"], bins = 40, range = (0, 1000000))
    
    #binLabels
    binLabels = [str(int(value)) + "-" + str(int(binsGM[1][index+1])) for index, value in enumerate(binsGM[1]) if index != 40]
    #white bg
    axes4[i%6, i//6].set_facecolor("white")
    
    rects1 = axes4[i%6, i//6].bar(binsGM[1][1:], binsGM[0], width = widthPlot, color = "red", edgecolor = "gray")
    rects2 = axes4[i%6, i//6].bar(binsHeart[1][1:] + widthPlot, binsHeart[0], width = widthPlot, color = "skyblue", edgecolor = "gray")
    axes4[i%6, i//6].set_xticks(binsGM[1][1:] + widthPlot / 2)
    axes4[i%6, i//6].set_xticklabels(binLabels, rotation = 60)
    axes4[i%6, i//6].legend( (rects1[0], rects2[0]), ('GM', 'Heart') )
    axes4[i%6, i//6].tick_params(axis='both', which='major', pad=0, grid_alpha = 0.7, color = "black", grid_color = "black")
#    sb.distplot(heartChromSubset["dist"] ,
#            color="skyblue", ax=axes3[i%6, i//6], bins = 80, kde = False,
#            hist_kws=dict(alpha=0.4), rug = False)
#    sb.distplot(GMChromSubset["dist"] ,
#            color="red", ax=axes3[i%6, i//6], bins = 80, kde = False,
#            hist_kws=dict(alpha=0.4), rug = False)
    axes4[i%6, i//6].set_xlabel("Loop size distribution " + feature)

f4.savefig("/2TB-disk/Project/Documentation/Meetings/06-05-2019/loopSizeDistributionPerChromosomePerLoopType_06-05-2019_40bins.png")



minValue = np.min(GMSubset[GMSubset["chr"] == "chr1"]["dist"])
maxValue = np.max(GMSubset[GMSubset["chr"] == "chr1"]["dist"])
desiredBins = 80
binStep  = (maxValue-minValue)/desiredBins
binValues = np.arange(start = 0, stop = 1000000 + 1, step = 1000000/80)   
binsGM = np.histogram(GMSubset[GMSubset["chr"] == "chr1"]["dist"], bins = 80, range = (0, 1000000))
bins2 = np.histogram_bin_edges(GMSubset[GMSubset["chr"] == "chr1"]["dist"], bins = 80, range = (0, 1000000))

binsHeart = np.histogram(heartSubset[heartSubset["chr"] == "chr1"]["dist"], bins = 80, range = (0, 1000000))

figureTest = plt.figure(figsize = (20,5))
axTest     = figureTest.add_subplot(111)

rects1 = axTest.bar(binsGM[1], np.append(binsGM[0], 0), width = widthPlot, color = "red")
rects2 = axTest.bar(binsHeart[1] + 5000, np.append(binsHeart[0], 0), width = widthPlot, color = "skyblue")
axTest.set_xticks(binsGM[1] + widthPlot / 2)
plt.xticks(rotation = 60)
axTest.legend( (rects1[0], rects2[0]), ('GM', 'Heart') )
locs, labels = plt.xticks()
plt.show()

axTest



#loop size distribution not per chromosome, but only per tissue type
figLoopSizeDist, axLoopSizeDist = plt.subplots(1,1, figsize = (10,9))
coloursPlot = ["red", "skyblue"]
axLoopSizeDist.hist([GMSubset["dist"], heartSubset["dist"]], bins = 80, color = coloursPlot, label = ["GM", "Heart"],
                    rwidth = 0.8, edgecolor = "gray")
axLoopSizeDist.legend(prop={'size': 14})
axLoopSizeDist.tick_params(axis='both', which='major', pad=0, grid_alpha = 0.7, color = "black", grid_color = "black")
axLoopSizeDist.set_facecolor("white")
axLoopSizeDist.set_xlabel("Loop size distribution (aggregated over 23 chromosomes)")

axLoopSizeDist.figure.savefig("/2TB-disk/Project/Documentation/Meetings/06-05-2019/loopSizeDistributionAggregated_06-05-2019_80bins.png")

#get the bins from this plot and iteratively remove too large Heart loops and random GM loops

counts, bins, bars = axLoopSizeDist.hist([GMSubset["dist"], heartSubset["dist"]], bins = 80, color = coloursPlot, label = ["GM", "Heart"],
                    rwidth = 0.8, edgecolor = "gray")


######
#####
#####   TOWARDS THE END OF THE PROJECT, IT WAS DETERMINED THAT WE SHOULD EQUALISE LOOP SIZE DISTRIBUTION QUICKLY
#####   BY SIMPLY DELETING EQUAL AMOUNTS OF HEART AND GM LOOPS TO GET AN EQUAL DISTRIBUTION OF LOOP SIZES 
#####   THIS IS DONE WITH THE CODE BELOW, WHICH RETURNS INDICES THAT CAN BE USED TO ALTER THE FEATURE ARRAYS
#####
######



def equalise_loop_size_distributions(GMSubset, heartSubset, maxLoopsToRemoveInEach = 2500):
    figLoopSizeDist, axLoopSizeDist = plt.subplots(1,1, figsize = (10,9))
    
    
    counts, bins, bars = axLoopSizeDist.hist([GMSubset["dist"], heartSubset["dist"]], bins = 80, color = coloursPlot, label = ["GM", "Heart"],
                    rwidth = 0.8, edgecolor = "gray")
    
    binsWhereHeartLoopsMoreThanGM = np.ravel(np.where(counts[1] > counts[0]))
    totalToRemoveHeart = np.sum(np.abs(counts[1] - counts[0])[binsWhereHeartLoopsMoreThanGM])
    print(totalToRemoveHeart)
    binsWhereGMLoopsMoreThanHeart = np.ravel(np.where(counts[0] > counts[1]))
    totalToRemoveGM = np.sum(np.abs(counts[1] - counts[0])[binsWhereGMLoopsMoreThanHeart])
    print(totalToRemoveGM)
    HeartIndicesToRemove = []
    GMIndicesToRemove = []
    
    totalLoopsRemovableForHeart = 0
    for count, currentBin in enumerate(binsWhereHeartLoopsMoreThanGM):
        print("Working on Heart bin " + str(count + 1) + "out of: " + str(len(binsWhereHeartLoopsMoreThanGM)) + ".")
        differenceBetweenHeartAndGMCurrentBin = np.int(abs(counts[0] - counts[1])[currentBin])
        totalLoopsRemovableForHeart += differenceBetweenHeartAndGMCurrentBin
        lowerEndDist, higherEndDist = bins[currentBin], bins[currentBin + 1]
        indicesToChooseFromToRemoveHeart = heartSubset[(heartSubset["dist"] >= lowerEndDist) & \
                                           (heartSubset["dist"] < higherEndDist)].index
        indicesToRemoveHeart = np.random.choice(indicesToChooseFromToRemoveHeart,
                         size = differenceBetweenHeartAndGMCurrentBin,
                         replace = False)
        HeartIndicesToRemove.append(indicesToRemoveHeart)
        
        #Do the same thing, but for a GM bins that have more loops than those bins in heart
    
    totalLoopsRemovableForGM = 0
    for count, currentBin in enumerate(binsWhereGMLoopsMoreThanHeart):
        print("Working on GM bin " + str(count + 1) + "out of: " + str(len(binsWhereGMLoopsMoreThanHeart)) + ".")
        differenceBetweenHeartAndGMCurrentBin = np.int(abs(counts[0] - counts[1])[currentBin])
        totalLoopsRemovableForGM += differenceBetweenHeartAndGMCurrentBin
        lowerEndDist, higherEndDist = bins[currentBin], bins[currentBin + 1]
        indicesToChooseFromToRemoveGM = GMSubset[(GMSubset["dist"] >= lowerEndDist) & \
                                           (GMSubset["dist"] < higherEndDist)].index
        indicesToRemoveGM = np.random.choice(indicesToChooseFromToRemoveGM,
                         size = differenceBetweenHeartAndGMCurrentBin,
                         replace = False)
        GMIndicesToRemove.append(indicesToRemoveGM)
        
    
    print("Possible to remove Heart: " + str(totalLoopsRemovableForHeart))
    print("Possible to remove GM : " + str(totalLoopsRemovableForGM))
    
    #these are now still in arrays. Get the elements of the arrays into one list
    HeartIndicesToRemove = [index for entry in HeartIndicesToRemove for index in entry]
    GMIndicesToRemove    = [index for entry in GMIndicesToRemove for index in entry]
    #Now you have two sets of indices you would want to remove. Probably, there are more heart to remove than
    #GM. Also, I wish to impose a maximum amount of loops to be removed from the data.
    
    #To do this:
        #-subsample  the max amount from the smallest one (or the total amount if there are less loops
        #to delete for that sample than half the max amount)
        #do the same for the larger sample
        #return these indices as the ones to be deleted from the feature Array. Also return adapted pd.DataFrame
        #to visualise the new distributions
        
    print("Combining...")
    smallestSet = min(len(HeartIndicesToRemove), len(GMIndicesToRemove))
    print(str(smallestSet))
    print("HeartAmount: " + str(len(HeartIndicesToRemove)))
    print("GMAmount: " + str(len(GMIndicesToRemove)))
    
    if smallestSet < maxLoopsToRemoveInEach:
        loopsToSamplePerCellType = smallestSet
    else:
        loopsToSamplePerCellType = maxLoopsToRemoveInEach
    
    print("Removing " + str(loopsToSamplePerCellType) + "loops per cell type. " + str(2* loopsToSamplePerCellType) + " loops total removed.")
        
        #not good...one should go without sampling but oh well.
    heartFinalToRemove = np.random.choice(HeartIndicesToRemove, size = loopsToSamplePerCellType,
                                          replace = False).tolist()
    print(heartFinalToRemove)
    print(type(heartFinalToRemove[0]))
    
    GMFinalToRemove = np.random.choice(GMIndicesToRemove, size = loopsToSamplePerCellType,
                                       replace = False).tolist()
    
    GMSubsetChanged = GMSubset.drop(index = GMFinalToRemove)
    print(GMSubsetChanged.shape)
    HeartSubsetChanged = heartSubset.drop(index = heartFinalToRemove)
    
    IDsToRemoveFromFeatureArrayHeart = [str(index) + "_Heart" for index in heartFinalToRemove]
    IDsToRemoveFromFeatureArrayGM    = [str(index) + "_GM12878" for index in GMFinalToRemove]
    
    combinedList = IDsToRemoveFromFeatureArrayHeart
    for element in IDsToRemoveFromFeatureArrayGM:
        combinedList.append(element)
    
    
    return(GMSubsetChanged, HeartSubsetChanged, combinedList)
    

GMSubsetModifiedLoopSizeDist , heartSubsetModifiedLoopSizeDist, IDsToRemoveFromFeatArray = \
    equalise_loop_size_distributions(GMSubset, heartSubset, 2500)

IDsToRemoveFromFeatArray
len(IDsToRemoveFromFeatArray)
GMSubsetModifiedLoopSizeDist


#now draw the modified number of loops

figLoopSizeDist, axLoopSizeDist = plt.subplots(1,1, figsize = (10,9))
coloursPlot = ["red", "skyblue"]
axLoopSizeDist.hist([GMSubsetModifiedLoopSizeDist["dist"], heartSubsetModifiedLoopSizeDist["dist"]], bins = 80, color = coloursPlot, label = ["GM", "Heart"],
                    rwidth = 0.8, edgecolor = "gray")
axLoopSizeDist.legend(prop={'size': 14})
axLoopSizeDist.tick_params(axis='both', which='major', pad=0, grid_alpha = 0.7, color = "black", grid_color = "black")
axLoopSizeDist.set_facecolor("white")
axLoopSizeDist.set_xlabel("Loop size distribution (aggregated over 23 chromosomes). Corrected for Loop size disparity.")

figLoopSizeDist, axLoopSizeDist = plt.subplots(1,1, figsize = (10,9))
coloursPlot = ["red", "skyblue"]
axLoopSizeDist.hist([GMSubsetModifiedLoopSizeDist["dist"], heartSubsetModifiedLoopSizeDist["dist"]], bins = 40, color = coloursPlot, label = ["GM", "Heart"],
                    rwidth = 0.8, edgecolor = "gray")
axLoopSizeDist.legend(prop={'size': 14})
axLoopSizeDist.tick_params(axis='both', which='major', pad=0, grid_alpha = 0.7, color = "black", grid_color = "black")
axLoopSizeDist.set_facecolor("white")
axLoopSizeDist.set_xlabel("Loop size distribution (aggregated over 23 chromosomes). Corrected for Loop size disparity.")

sb.distplot(GMSubset["dist"])
sb.distplot(GMSubsetModifiedLoopSizeDist["dist"])
sb.distplot(heartSubset["dist"])
sb.distplot(heartSubsetModifiedLoopSizeDist["dist"])
#####
####
#####


##############################################
####
#### Load the normal .pkl file for classification into GM and non-GM, and scrap the IDs so that the loop size dist is more equal.
####
###############################################
featureDictPath = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/SavedFeatureDicts/finalFeatureDictionary_2019-04-08.pkl"

with open(featureDictPath, "rb") as f:
    featureDictToEdit = pickle.load(f)


featureDictToEdit["featureArray"]
isItThere = [np.ravel(np.where(element == featureDictToEdit["loopID"])) for element in IDsToRemoveFromFeatArray]
isItThereUnpacked = [element for array in isItThere for element in array]
#np.where(featureDictToEdit == IDsToRemoveFromFeatArray)
#featureDictToEdit["loopID"].isin(IDsToRemoveFromFeatArray)


newFeatArray = np.delete(featureDictToEdit["featureArray"], isItThereUnpacked, axis = 0)
newFeatArray.shape

newClassLabelArray = np.delete(featureDictToEdit["classLabelArray"], isItThereUnpacked, axis = 0)
newClassLabelArray.shape

newLoopIDs = np.delete(featureDictToEdit["loopID"], isItThereUnpacked, axis = 0)
newLoopIDs.shape

featureDictToEdit["featureArray"], featureDictToEdit["classLabelArray"], featureDictToEdit["loopID"] = \
    newFeatArray, newClassLabelArray, newLoopIDs

featureDictToEdit["featureArray"].shape

editedFeatureDictPathToWriteTo = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/SavedFeatureDicts/finalFeatureDictionary_2019-04-08_editedToBringLoopSizeDistributionsCloser_4954loopsRemoved_08_06_2019.pkl"

with open(editedFeatureDictPathToWriteTo, "wb") as pickleFile:
    pickle.dump(featureDictToEdit, pickleFile)
    
    
    
###Do the same thing, but now removing in-between features as well

featureDictToEditNoInbetween = featureDictToEdit

inbetweenCols = np.where( ["inbetween" in item for item in featureDictToEditNoInbetween["namesFeatureArrayColumns"]])

#remove these from names and featureArray

newFeatNames = np.delete(featureDictToEditNoInbetween["namesFeatureArrayColumns"], inbetweenCols)
newFeats     = np.delete(featureDictToEditNoInbetween["featureArray"], inbetweenCols, axis = 1)

featureDictToEditNoInbetween["namesFeatureArrayColumns"] = newFeatNames
featureDictToEditNoInbetween["featureArray"] = newFeats

editedFeatureDictPathToWriteToNoInbetween = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/SavedFeatureDicts/finalFeatureDictionary_2019-04-08_editedToBringLoopSizeDistributionsCloser_4954loopsRemoved_08_06_2019_NoInbetween.pkl"


with open(editedFeatureDictPathToWriteToNoInbetween, "wb") as pickleFile:
    pickle.dump(featureDictToEditNoInbetween, pickleFile)


###
###
###

figLoopSizeDist, axLoopSizeDist = plt.subplots(1,1, figsize = (10,9))
coloursPlot = ["red", "skyblue"]
axLoopSizeDist.hist([GMSubset["dist"], heartSubset["dist"]], bins = 40, color = coloursPlot, label = ["GM", "Heart"],
                    rwidth = 0.8, edgecolor = "gray")
axLoopSizeDist.legend(prop={'size': 14})
axLoopSizeDist.tick_params(axis='both', which='major', pad=0, grid_alpha = 0.7, color = "black", grid_color = "black")
axLoopSizeDist.set_facecolor("white")
axLoopSizeDist.set_xlabel("Loop size distribution (aggregated over 23 chromosomes)")

axLoopSizeDist.figure.savefig("/2TB-disk/Project/Documentation/Meetings/06-05-2019/loopSizeDistributionAggregated_06-05-2019_40bins.png")






sb.distplot(heartSubset["dist"] ,
            color="skyblue", ax=axLoopSizeDist, bins = 80, kde = False,
            hist_kws=dict(alpha=0.2, ), rug = False)
sb.distplot(GMSubset["dist"] ,
            color="red", ax=axLoopSizeDist, bins = 80, kde = False,
            hist_kws=dict(alpha=0.2), rug = False)





#get a plot of the amount of loops per chromosome
GMSpecificLoops = loopsGMAndHeart[0]
GMSpecificLoops["loopType"] = "GM12878"
HeartSpecificLoops = loopsGMAndHeart[1]
HeartSpecificLoops["loopType"] = "Heart"
GMSpecificLoops.head()

countsPerChrom = GMSpecificLoops["chr"].value_counts()
countsPerChrom
countsPerChrom.index
barPlotLoopsPerChrom = sb.barplot(countsPerChrom.index, countsPerChrom)
barPlotLoopsPerChrom.set_xticklabels(countsPerChrom.index, rotation = 60)
barPlotLoopsPerChrom.set_ylabel("Number of loops")
barPlotLoopsPerChrom.figure


chromSizes = pd.read_csv("/2TB-disk/Project/Geert Geeven loops data/Data/hg19ChromSizes/hg19.chrom.sizes.txt", sep = "\t", header = None)
chromSizes.columns = ["chromName", "basePairs"]

relevantChromSizes = chromSizes[chromSizes["chromName"].isin(countsPerChrom.index)]

#correct for chromosome size

countsPerChrom.sort_index(inplace = True)
relevantChromSizes.sort_values("chromName", inplace = True)

sizeCorrectedLoopCounts = countsPerChrom.values/relevantChromSizes["basePairs"].values
countsPerChrom.index

sizeCorrectedLoopCounts

orderedCollectionChromosomesHighToLow = np.flip(np.argsort(sizeCorrectedLoopCounts))
sizeCorrectedLoopCounts[np.flip(np.argsort(sizeCorrectedLoopCounts))]

countsPerChrom.index

barPlotLoopsPerChromCorrected = sb.barplot(countsPerChrom.index[orderedCollectionChromosomesHighToLow], sizeCorrectedLoopCounts[orderedCollectionChromosomesHighToLow])
barPlotLoopsPerChromCorrected.set_xticklabels(countsPerChrom.index, rotation = 60)
barPlotLoopsPerChromCorrected.set_ylabel("Number of loops per basepair (normalised to chrom. size)")
barPlotLoopsPerChromCorrected.figure




##############################################################
##                                                          ##
##                  get random anchors                      ##
##                                                          ##
##############################################################

# X = number of loops in real data for each chromsome
#for every chromosome, pick X locations to put anchors

#for anchor designation of 10.000 bp, I need 5000 bp on either side to exist
#to calculate features. That is what this does. 

bufferEndOfChrom = 5005
dictRandomAnchorSites = {}
for chrom in relevantChromSizes["chromName"]:
    amountOfLoopAnchorsToMake = 2 * countsPerChrom[chrom]
    
    #get random coordinates. Will not work at the outsides of the chromosome because
    #I cannot calculate features there.
    basePairsEndChrom = relevantChromSizes[relevantChromSizes["chromName"] == chrom]["basePairs"]
    #+1 to make it end-inclusive
    randomValues = np.random.randint(0 + bufferEndOfChrom, high = basePairsEndChrom - bufferEndOfChrom + 1, size = amountOfLoopAnchorsToMake)
    dictRandomAnchorSites[chrom] = randomValues
    


#check that I now have 40.000 loop anchors:
lenSum = 0
for entry, value in dictRandomAnchorSites.items():
    lenSum += len(value)
print(lenSum)    
    
#draw a distribution to see how it is distributed over the genome

uniformRandomAnchorSitesChromOne = sb.distplot(dictRandomAnchorSites["chr1"], bins = 30)
#how does that compare to the actual distribution?
chromOneActualLoops = GMSpecificLoops[GMSpecificLoops["chr"] == "chr1"][["vp_X1", "maxV4CscorePos"]] 
chromOneActualLoops["vp_X1"] = chromOneActualLoops["vp_X1"] + 5000
chromOneActualLoopAnchorsOnly = pd.concat([chromOneActualLoops["vp_X1"],
                                           chromOneActualLoops["maxV4CscorePos"]]).values


realDistAnchorSitesChromOne = sb.distplot(chromOneActualLoopAnchorsOnly, bins = 30)


#that's quite different
binsForEachChrom = 30
binWidthChromOne = 249250621/binsForEachChrom
binsForRealDist = np.histogram(chromOneActualLoopAnchorsOnly, bins = binsForEachChrom, range = (0, 249250621),
                               density = True)
#binsForRealDistEdges = np.histogram_bin_edges(chromOneActualLoopAnchorsOnly, bins = 30)
probDensityPerBin = binsForRealDist[0] * binWidthChromOne
binLabelsChromOne = [str(int(value)) + "-" + str(int(binsForRealDist[1][index+1])) for index, value in enumerate(binsForRealDist[1]) if index != 30]
binLabelsChromOne
probDensityPerBin
sum(probDensityPerBin)

#now sample loop anchors from this real distribution

floerg = np.random.choice(np.arange(0, binsForEachChrom),
                          size =  len(chromOneActualLoopAnchorsOnly),
                          p = probDensityPerBin)


#binLabelsChromNumbers = [int(num) for num in binLabelsChromOne[0].split("-")]

binLabelsChromNumbers = [[int(bins.split("-")[0]), int(bins.split("-")[1])]  for bins in binLabelsChromOne ]

anchorListSimilarlyDistributed = []
#[x for y in collection for x in y]
for num in floerg:
    currentBinEdges = binLabelsChromNumbers[num]
    if num == floerg.min():
        #+1 to make it inclusive of the last base
        anchorToAdd = np.random.randint(currentBinEdges[0] + bufferEndOfChrom,
                                        currentBinEdges[1] + 1)
    elif num == floerg.max():
        anchorToAdd = np.random.randint(currentBinEdges[0],
                                        currentBinEdges[1] - bufferEndOfChrom + 1)
    else:
        anchorToAdd = np.random.randint(currentBinEdges[0], currentBinEdges[1] + 1)
    
    anchorListSimilarlyDistributed.append(anchorToAdd)
        
sb.distplot(anchorListSimilarlyDistributed, bins = 30)





#function to do this for all chromosomes automatically
#make both uniform (naive) random anchors and random anchor sites that follow (mimick) the
#distribution of real anchor sites


def generate_random_anchor_sites(loopCountsPerChromosome     = countsPerChrom,
                                 chromosomeSizes         = relevantChromSizes,
                                 GMLoopDataFrame         = GMSpecificLoops,
                                 binsChromosome          : int          = 30,
                                 bufferBpEdgeChromosome  : int          = 5005) -> dict:
    
    
    
    dictResult = {"naive"  : {},
                  "mimick" : {},
                  "actual" : {}}
    
    for chrom in chromosomeSizes["chromName"]:
        #each loop has two anchors.
        amountOfLoopAnchorsToMake = 2 * loopCountsPerChromosome[chrom]
        
        #get random coordinates. Will not work at the outsides of the chromosome because
        #I cannot calculate features there.
        basePairsEndChrom = chromosomeSizes[chromosomeSizes["chromName"] == chrom]["basePairs"]
        #+1 to make it end-inclusive
        randomValuesUniform = np.random.randint(0 + bufferBpEdgeChromosome,
                                         high = basePairsEndChrom - bufferBpEdgeChromosome + 1,
                                         size = amountOfLoopAnchorsToMake)
        dictResult["naive"][chrom] = randomValuesUniform
        
        #add in actual anchor sites for this chrom
        
        actualLoops = GMLoopDataFrame[GMLoopDataFrame["chr"] == chrom][["vp_X1", "maxV4CscorePos"]] 
        #anchor intervals are 10000 wide, so the start of the anchor + 5000 = middle
        actualLoops["vp_X1"] = actualLoops["vp_X1"] + 5000
        actualLoopAnchorsOnly = pd.concat([actualLoops["vp_X1"],
                                           actualLoops["maxV4CscorePos"]]).values
        
        dictResult["actual"][chrom] = actualLoopAnchorsOnly
        
        
        #add in random coordinates that follow a distribution over the genome
        #that is (very) similar to that of the real anchor coordinates
        #two-tiered: bin actual values in 30 bins and get the probabilities
        #of being in any of those bins.
        #sample from those 30 bins
        #then, within each of the 30 bins, uniformly sample random loop anchors
        
        binWidthChrom = basePairsEndChrom.values[0]/binsChromosome
        
        binsForRealDist = np.histogram(actualLoopAnchorsOnly,
                                       bins = binsChromosome,
                                       range = (0, basePairsEndChrom.values[0]),
                                       density = True)
        
        probDensityPerBin = binsForRealDist[0] * binWidthChrom
        binLabelsChrom = [str(int(value)) + "-" + str(int(binsForRealDist[1][index+1])) for index, value in enumerate(binsForRealDist[1]) if index != binsChromosome]
        binLabelsChromNumbers = [[int(bins.split("-")[0]), int(bins.split("-")[1])]  for bins in binLabelsChrom ]
        
        print("total probabilities in histogram: " + str(sum(probDensityPerBin)))
        
        #now sample from this distribution
        
        histogramGuidedSampling = np.random.choice(np.arange(0, binsChromosome),
                                                   size =  len(actualLoopAnchorsOnly),
                                                   p = probDensityPerBin)
        
        anchorListMimick = []
        for num in histogramGuidedSampling:
            currentBinEdges = binLabelsChromNumbers[num]
            if num == histogramGuidedSampling.min():
                #+1 to make it inclusive of the last base
                anchorToAdd = np.random.randint(currentBinEdges[0] + bufferBpEdgeChromosome,
                                                currentBinEdges[1] + 1)
            elif num == histogramGuidedSampling.max():
                anchorToAdd = np.random.randint(currentBinEdges[0],
                                                currentBinEdges[1] - bufferBpEdgeChromosome + 1)
            else:
                anchorToAdd = np.random.randint(currentBinEdges[0], currentBinEdges[1] + 1)
            
            anchorListMimick.append(anchorToAdd)
            
        dictResult["mimick"][chrom] = np.asarray(anchorListMimick)
        
    return dictResult

randomLoopAnchorDictionaryAllChroms = generate_random_anchor_sites()
    
randomLoopAnchorDictionaryAllChroms["actual"]["chr1"]  / 1e8  
sb.distplot(randomLoopAnchorDictionaryAllChroms["actual"]["chr1"]  / 1e8 )

def plot_distplots_chromosome(chrom          : str,
                              loopAnchorDict : dict = randomLoopAnchorDictionaryAllChroms) -> list:
    figure, ax = plt.subplots(1,3, sharey = True, figsize = (22,7))
    actual  = sb.distplot(loopAnchorDict['actual'][chrom] / 1e8, bins = 30, ax = ax[0])
    actual.set_title("Actual loop anchors")
    actual.set_xlabel("basepairs ( * 1e8)")
    uniform = sb.distplot(loopAnchorDict['naive'][chrom]  / 1e8, bins = 30, ax = ax[1])
    uniform.set_title("Uniformly sampled random loop anchors")
    uniform.set_xlabel("basepairs ( * 1e8)")
    mimick  = sb.distplot(loopAnchorDict['mimick'][chrom] / 1e8, bins = 30, ax = ax[2])
    mimick.set_title("Random loop anchors sampled according to true anchor distribution (mimick)")
    mimick.set_xlabel("basepairs ( * 1e8)")
    figure.suptitle(chrom, fontsize = 14)
    
    result = figure
    
    return result

chrom1 = plot_distplots_chromosome("chr1")  

chromX = plot_distplots_chromosome("chrX")

chrom12 = plot_distplots_chromosome("chr12")  
                            
    
sum(randomLoopAnchorDictionaryAllChroms["actual"]["chr1"] == randomLoopAnchorDictionaryAllChroms["mimick"]["chr1"])



#okay nice. Now, I need to get this into a format that my feature calculation script will accept

#first test:
#are there even numbers of loop anchors? (if not, I can not easily merge them into the existing format, I think)

for chrom, values in randomLoopAnchorDictionaryAllChroms["mimick"].items(): print(len(values)/2)
for chrom, values in randomLoopAnchorDictionaryAllChroms["naive"].items(): print(len(values)/2)

#yes there are. Okay then.


#for i, (chrom, values) in enumerate(randomLoopAnchorDictionaryAllChroms["mimick"].items()):
#    chromRep   = [chrom] * int(0.5 * len(values))
#    loopTissue = ["GM12878"] * int(0.5 * len(values))
#    vp_middle  = values[0:int(len(values)*0.5)]
#    maxV4CscorePos = values[int(len(values)*0.5):len(values)]
#    
#    dataFramePart = pd.concat([pd.Series(chromRep),
#                               pd.Series(vp_middle),
#                               pd.Series(maxV4CscorePos),
#                               pd.Series(loopTissue)], axis = 1)
#
#    dataFramePart.columns = ["chr", "vp_middle", "maxV4CscorePos", "loopTissue"]
#    
#    if i == 0:
#        uniformRandomAnchorDF = dataFramePart
#    else:
#        uniformRandomAnchorDF = pd.concat([uniformRandomAnchorDF,
#                                           dataFramePart], axis = 0)


def format_random_anchors_for_processing(loopAnchorAllChromsDictionaryEntry : dict = randomLoopAnchorDictionaryAllChroms["mimick"]):
    
    for i, (chrom, values) in enumerate(loopAnchorAllChromsDictionaryEntry.items()):
        chromRep   = [chrom] * int(0.5 * len(values))
        loopTissue = ["GM12878"] * int(0.5 * len(values))
        vp_middle  = values[0:int(len(values)*0.5)]
        maxV4CscorePos = values[int(len(values)*0.5):len(values)]
        
        dataFramePart = pd.concat([pd.Series(chromRep),
                                   pd.Series(vp_middle),
                                   pd.Series(maxV4CscorePos),
                                   pd.Series(loopTissue)], axis = 1)
    
        dataFramePart.columns = ["chr", "vp_middle", "maxV4CscorePos", "loopTissue"]
        
        if i == 0:
            resultDF = dataFramePart
        else:
            resultDF = pd.concat([resultDF,
                                  dataFramePart], axis = 0)
    resultDF.sort_values(inplace = True, by = "chr", ascending = True)
    
    return resultDF

fakeAnchorsFormattedForFurtherProcessingMimick = format_random_anchors_for_processing(randomLoopAnchorDictionaryAllChroms["mimick"])
fakeAnchorsFormattedForFurtherProcessingNaive  = format_random_anchors_for_processing(randomLoopAnchorDictionaryAllChroms["naive"])

print(fakeAnchorsFormattedForFurtherProcessingMimick.head()); print(fakeAnchorsFormattedForFurtherProcessingMimick.tail())
len(fakeAnchorsFormattedForFurtherProcessingMimick)


##this output is now the same as the output of
##get_loop_data_for_chromatin_marks (bar the fact that it doesn't give a tuple of (GMloops, Heartloops)), but just a DF with fake GM loops.
#For further processing: 
#change function that formats anchors in BED format to have a switch whether to include in-between or not (don't need it here)

#intervals = [2500, 10000] #same as before
bedFormatUniformRandomAnchors = make_bed_format_anchors(fakeAnchorsFormattedForFurtherProcessingNaive,
                                                        False,
                                                        2500, 10000)


intersectBEDsUniformRandomAnchors = generate_intersect_BEDs(bedFormatUniformRandomAnchors, chromatinMarkDataFrame)

directoryToSaveUniformRandomAnchors = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/FilesForHPCProcessing/BedIntersectDataFrames_RandomAnchorsUniform_Unpaired_17_05_2019"
directoryToSaveUniformRandomAnchors += "/"

make_dataframe_BED_save(intersectBEDsUniformRandomAnchors, directoryToSaveUniformRandomAnchors,
                        "UniformSingleRandomAnchors")


#do the same for mimicked distribution

bedFormatMimickAnchors = make_bed_format_anchors(fakeAnchorsFormattedForFurtherProcessingMimick,
                                                        False,
                                                        2500, 10000)
intersectBedFormatMimickAnchors = generate_intersect_BEDs(bedFormatMimickAnchors, chromatinMarkDataFrame)

directoryToSaveMimickAnchors = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/FilesForHPCProcessing/BedIntersectDataFrames_RandomAnchorsMimick_Unpaired_17_05_2019"
directoryToSaveMimickAnchors += "/"

make_dataframe_BED_save(intersectBedFormatMimickAnchors, directoryToSaveMimickAnchors,
                        "MimickSingleRandomAnchors")


#add chromatinmark.txt for processing on HPC
create_chromatinmarktxt_in_bedfiledir(directoryToSaveUniformRandomAnchors, chromatinMarkDataFrame)
create_chromatinmarktxt_in_bedfiledir(directoryToSaveMimickAnchors, chromatinMarkDataFrame)





##############################################################
##                                                          ##
##                get random linked anchors                 ##
##                                                          ##
##############################################################




BEDFormatFeaturesForFalseLinkedAnchors = make_bed_format_anchors(positionDataFrame[positionDataFrame["loopTissue"] == "GM12878"],
                                                                 True, 2500,10000)

BEDFormatFeaturesForFalseLinkedAnchors[0].head()
BEDFormatFeaturesForFalseLinkedAnchors[1].head()
#BEDFormatFeaturesForFalseLinkedAnchors[2].head()




BEDFormatFeaturesForFalseLinkedAnchors[0].columns = BEDFormatFeaturesForFalseLinkedAnchors[0].columns + "leftLeft"
BEDFormatFeaturesForFalseLinkedAnchors[1].columns = BEDFormatFeaturesForFalseLinkedAnchors[1].columns + "leftRight"


leftAnchorCombined = pd.concat([BEDFormatFeaturesForFalseLinkedAnchors[0], BEDFormatFeaturesForFalseLinkedAnchors[1]],
                          axis = 1)


#get anchor middle
leftAnchorCombined["leftAnchorMiddlePos"] = (leftAnchorCombined["stopleftRight"] + leftAnchorCombined["startleftLeft"] + 1)/2
leftAnchorCombined["chr"] = leftAnchorCombined["chrleftLeft"]
regexForIDOnly = "\w*_\d*_([\w\d]*)"
leftAnchorCombined["loopID"] = [re.match(regexForIDOnly, entry)[1] for entry in leftAnchorCombined["nameleftLeft"]]

leftAnchorCombined.head()

#same for the right anchor

BEDFormatFeaturesForFalseLinkedAnchors[3].head()
BEDFormatFeaturesForFalseLinkedAnchors[4].head()


BEDFormatFeaturesForFalseLinkedAnchors[3].columns = BEDFormatFeaturesForFalseLinkedAnchors[3].columns + "rightLeft"
BEDFormatFeaturesForFalseLinkedAnchors[4].columns = BEDFormatFeaturesForFalseLinkedAnchors[4].columns + "rightRight"


rightAnchorCombined = pd.concat([BEDFormatFeaturesForFalseLinkedAnchors[3], BEDFormatFeaturesForFalseLinkedAnchors[4]],
                          axis = 1)


#get anchor middle
rightAnchorCombined["rightAnchorMiddlePos"] = (rightAnchorCombined["stoprightRight"] + rightAnchorCombined["startrightRight"] + 1)/2
rightAnchorCombined["chr"] = rightAnchorCombined["chrrightRight"]
regexForIDOnly = "\w*_\d*_([\w\d]*)"
rightAnchorCombined["loopID"] = [re.match(regexForIDOnly, entry)[1] for entry in rightAnchorCombined["namerightRight"]]

rightAnchorCombined.head()

#now combine and iterate

rightAnchorCombined.drop(axis = 1, labels = ["chr", "loopID"], inplace = True)


combinedAnchorDF = pd.concat([leftAnchorCombined, rightAnchorCombined], axis = 1)
leftAnchorCombined.head()
combinedAnchorDF.head()
combinedAnchorDF.sort_values("chr", inplace = True)
combinedAnchorDF["loopSize"] = combinedAnchorDF["rightAnchorMiddlePos"] - combinedAnchorDF["leftAnchorMiddlePos"]

#on to iteration
#import package
from heapq import nsmallest
s = [1,2,3,4,5,6,7]
nsmallest(3, s, key=lambda x: abs(x-6.5))
[6, 7, 5]


def make_false_anchor_pairs(dataWithLeftAndRightAnchorTruePositions = combinedAnchorDF):

    totalDFFalseLinkedAnchors = None
    GMGroupedByChrom = dataWithLeftAndRightAnchorTruePositions.groupby("chr")
    
    for name, group in GMGroupedByChrom:
        
        uniqueRightAnchorPositions = np.unique(group["rightAnchorMiddlePos"])
        uniqueLeftAnchorPositions  = np.unique(group["leftAnchorMiddlePos"])
        currentChrom = np.unique(group["chr"])[0]
        
        print("Working on data for: " + currentChrom + ".")
        for index, row in group.iterrows():
            
            #get the closest values 
            #https://stackoverflow.com/questions/24112259/finding-k-closest-numbers-to-a-given-number 
            leftAnchorPos, rightAnchorPos = row["leftAnchorMiddlePos"], row["rightAnchorMiddlePos"]
            loopID = row["loopID"]
            
            #list of closest to  right anchor. Will find itself. Therefore, take 6, remove the first.
            closestToRightAnchor = nsmallest(6, uniqueRightAnchorPositions, key = lambda x: abs(x-rightAnchorPos))[1:6]
            
            #for these, get loop sizes, and the linked IDs for the left and the right part
            #sometimes, the same anchor position appears in multple loops, hence drop_duplicates
            subsetToTake = group[group["rightAnchorMiddlePos"].isin(closestToRightAnchor)].drop_duplicates(subset = "rightAnchorMiddlePos")
            
            loopSizes = subsetToTake["rightAnchorMiddlePos"] - subsetToTake["leftAnchorMiddlePos"]
            loopIDLeft = [loopID] * len(closestToRightAnchor)
            #get matching loopID part for later retrieval from feature array
            loopIDRight = [re.match("\w*_\d*_([\w\d]*)", entry)[1] for entry in subsetToTake["namerightRight"]]
            leftAnchorMiddlePositions = [leftAnchorPos] * len(closestToRightAnchor)
            rightAnchorMiddlePositions = subsetToTake["rightAnchorMiddlePos"]
            chromosomeInfo = [currentChrom] * len(closestToRightAnchor) 
            
            rightDataFrame = pd.DataFrame({"chr": chromosomeInfo,
                                           "loopIDLeft" : loopIDLeft,
                                           "loopIDRight" : loopIDRight,
                                           "leftAnchorMiddlePos" : leftAnchorMiddlePositions,
                                           "rightAnchorMiddlePos" : rightAnchorMiddlePositions,
                                           "loopSize" : loopSizes})
            #print(rightDataFrame)
            #Now do the exact same for the left anchors
            
            
            closestToLeftAnchor = nsmallest(6, uniqueLeftAnchorPositions, key = lambda x: abs(x-leftAnchorPos))[1:6]
            
            #for these, get loop sizes, and the linked IDs for the left and the right part
            #sometimes, the same anchor position appears in multple loops, hence drop_duplicates
            subsetToTake = group[group["leftAnchorMiddlePos"].isin(closestToLeftAnchor)].drop_duplicates(subset = "leftAnchorMiddlePos")
            
            #this stays the same, size is always right-left
            loopSizes = subsetToTake["rightAnchorMiddlePos"] - subsetToTake["leftAnchorMiddlePos"]
            loopIDLeft = [re.match("\w*_\d*_([\w\d]*)", entry)[1] for entry in subsetToTake["nameleftLeft"]]
            #get matching loopID part for later retrieval from feature array
            loopIDRight = [loopID] * len(closestToLeftAnchor)
            leftAnchorMiddlePositions = subsetToTake["leftAnchorMiddlePos"]   
            rightAnchorMiddlePositions = [rightAnchorPos] * len(closestToLeftAnchor)
            chromosomeInfo = [currentChrom] * len(closestToLeftAnchor) 
            
            leftDataFrame = pd.DataFrame({"chr": chromosomeInfo,
                                           "loopIDLeft" : loopIDLeft,
                                           "loopIDRight" : loopIDRight,
                                           "leftAnchorMiddlePos" : leftAnchorMiddlePositions,
                                           "rightAnchorMiddlePos" : rightAnchorMiddlePositions,
                                           "loopSize" : loopSizes})
            #print(leftDataFrame)
            
            #now we have 10 alternative loops for one true loop. concat to outputDF
            totalDFFalseLinkedAnchors = pd.concat([totalDFFalseLinkedAnchors, rightDataFrame, leftDataFrame], axis = 0)
            
            #print(group[group["rightAnchorMiddlePos"].isin(closestToRightAnchor)])
    return totalDFFalseLinkedAnchors


totalDFFalseLinkedAnchors = make_false_anchor_pairs()



#save these loops to a file for future reference
now = str(datetime.datetime.today()).split()[0]
saveDirectory = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/FakeCoupledAnchors_10_06_2019/200000FakeCoupledAnchorDataFrame_" + now + ".csv"

pd.DataFrame.to_csv(totalDFFalseLinkedAnchors, saveDirectory)

#Okay great, now bin the distributions and get picking

orgLoopDistDF = positionDataFrame[positionDataFrame["loopTissue"] == "GM12878"]
orgLoopDistDF["loopSize"] = abs(orgLoopDistDF["vp_middle"]-orgLoopDistDF["maxV4CscorePos"])
orgLoopDistDF.head()
orgLoopDistDF.shape


sb.distplot(orgLoopDistDF["loopSize"], bins = 40, kde = False)
sb.distplot(totalDFFalseLinkedAnchors["loopSize"], bins = 40, kde = False, color = "r")

nBins = 40
maxLoopDist = 1000000
binsForRealDist = np.histogram(orgLoopDistDF["loopSize"],
                                       bins = nBins,
                                       range = (0, maxLoopDist),
                                       density = True)

binWidthEqualBins = 1000000/40

probDensityPerBin = binsForRealDist[0] * binWidthEqualBins
#binLabelsChrom = [str(int(value)) + "-" + str(int(binsForRealDist[1][index+1])) for index, value in enumerate(binsForRealDist[1]) if index != binsChromosome]
#binLabelsChromNumbers = [[int(bins.split("-")[0]), int(bins.split("-")[1])]  for bins in binLabelsChrom ]

print("total probabilities in histogram: " + str(sum(probDensityPerBin)))

#now sample from this distribution

histogramGuidedSamplingOfFakeAnchors = np.random.choice(np.arange(0, nBins),
                                           size =  len(orgLoopDistDF),
                                           p = probDensityPerBin)

#nu weet ik uit welke bins ik de zaken moet samplen. Nu dat nog doen.
binsAndFakeLoopNumbersToSampleFromThem = np.unique(histogramGuidedSamplingOfFakeAnchors, return_counts = True)

totalDFFalseLinkedAnchors["binnedAccordingToRealData"] = pd.cut(totalDFFalseLinkedAnchors["loopSize"], bins = binsForRealDist[1], labels = range(0,nBins))

finalDataFrameFakeAnchors = None
for num, bin_ in enumerate(binsAndFakeLoopNumbersToSampleFromThem[0]):
    toSampleCount = binsAndFakeLoopNumbersToSampleFromThem[1][num]
    toSampleFrom = totalDFFalseLinkedAnchors[totalDFFalseLinkedAnchors["binnedAccordingToRealData"] == bin_]
    sample = toSampleFrom.sample(n = toSampleCount)
    finalDataFrameFakeAnchors = pd.concat([finalDataFrameFakeAnchors, sample], axis = 0)
    
    

def helper_function_swap_loopID(loopID):
    regexResult = re.match("([\w\d]*)_([\w\d]*)", loopID)
    swapped = regexResult[2] + "_" + regexResult[1]
    return swapped

#####################################################
#I now have a final dataFrame with 20.000 fake loops. Their loop size distribution mimicks that of the real loops.
    
#Now: load a featureArray, remove Heart and inbetween, and splice together the left and right anchor features of
    #the fake loops
    

#I need to swap the number and celltype for the loopID, I apparently did that in my scripts somwhere
finalDataFrameFakeAnchors["loopIDLeft"] = [helper_function_swap_loopID(entry) for entry in finalDataFrameFakeAnchors["loopIDLeft"]]
finalDataFrameFakeAnchors["loopIDRight"] = [helper_function_swap_loopID(entry) for entry in finalDataFrameFakeAnchors["loopIDRight"]]

featureDictToLoadForFakeLoopsPath = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/SavedFeatureDicts/finalFeatureDictionary_2019-04-08_NoInBetween_28_05_2019.pkl"
with open(featureDictToLoadForFakeLoopsPath, "rb") as f:
    featureDictFakeLoops = pickle.load(f)

featureDictFakeLoops["classLabelArray"]
featureDictFakeLoops["featureArray"]    = np.delete(featureDictFakeLoops["featureArray"],
                                          np.ravel(np.where(featureDictFakeLoops["classLabelArray"] == "Heart")),
                                          axis = 0)
featureDictFakeLoops["loopID"]          = np.delete(featureDictFakeLoops["loopID"], obj = np.ravel(np.where(featureDictFakeLoops["classLabelArray"] == "Heart")), axis = 0)
featureDictFakeLoops["classLabelArray"] = np.delete(featureDictFakeLoops["classLabelArray"], obj = np.ravel(np.where(featureDictFakeLoops["classLabelArray"] == "Heart")), axis = 0)
featureDictFakeLoops["featureArray"].shape

#now, for every entry of the fake loops, get the associated features and give it an id

featureDictFakeLoops["namesFeatureArrayColumns"]
leftFeatures = np.where([("leftAnchor") in entry for entry in featureDictFakeLoops["namesFeatureArrayColumns"]] )
rightFeatures = np.where([("rightAnchor") in entry for entry in featureDictFakeLoops["namesFeatureArrayColumns"]] )


listOfFeats = []
listOfIDs   = []
listOfClass = []

for index, row in finalDataFrameFakeAnchors.iterrows():
    IDToGive = str(index) + "_" + "fakeGMLoop"
    leftArrayForThisFakeLoop  = np.ravel(featureDictFakeLoops["featureArray"][featureDictFakeLoops["loopID"] == row["loopIDLeft"]])[leftFeatures]
    rightArrayForThisFakeLoop = np.ravel(featureDictFakeLoops["featureArray"][featureDictFakeLoops["loopID"] == row["loopIDRight"]])[rightFeatures]
    #add this
    featsToAdd = np.concatenate((leftArrayForThisFakeLoop, rightArrayForThisFakeLoop), axis = 0)
    listOfFeats.append(featsToAdd) 
    listOfIDs.append(IDToGive)
    listOfClass.append("FakeLoopGM12878")
    
    
    
    

featsToAdd = np.vstack(listOfFeats)
IDsToAdd = np.ravel(np.vstack(listOfIDs))
classes = np.ravel(np.vstack(listOfClass))
##

featureDictFakeLoops["featureArray"] = np.vstack((featureDictFakeLoops["featureArray"], featsToAdd))
featureDictFakeLoops["loopID"] = np.append(featureDictFakeLoops["loopID"], IDsToAdd)
featureDictFakeLoops["classLabelArray"] = np.append(featureDictFakeLoops["classLabelArray"], classes)

##okay done. Save this altered feature dict. Then, make indices


for key, entry in featureDictFakeLoops.items():
    print(key)
    print("shape:")
    print(entry.shape)
    print("--")

featureDictFakeLoopsSavePath = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/SavedFeatureDicts/finalFeatureDictionary_10_06_2019_FakeLoops_NoInBetween.pkl"

with open(featureDictFakeLoopsSavePath, "wb") as f:
    pickle.dump(featureDictFakeLoops, f)

#okay nice!
    


###################################################################################################
####
####
#### Now make real loops and fake loops WITH in-between features (seeing that these seem to be most indicative of occuring vs. possible loops)
####
####
###################################################################################################
    
    


finalDataFrameFakeAnchorsWithInbetween = None
for num, bin_ in enumerate(binsAndFakeLoopNumbersToSampleFromThem[0]):
    toSampleCount = binsAndFakeLoopNumbersToSampleFromThem[1][num]
    toSampleFrom = totalDFFalseLinkedAnchors[totalDFFalseLinkedAnchors["binnedAccordingToRealData"] == bin_]
    sample = toSampleFrom.sample(n = toSampleCount)
    finalDataFrameFakeAnchorsWithInbetween = pd.concat([finalDataFrameFakeAnchors, sample], axis = 0)

finalDataFrameFakeAnchorsWithInbetween["loopIDLeft"] = [helper_function_swap_loopID(entry) for entry in finalDataFrameFakeAnchorsWithInbetween["loopIDLeft"]]
finalDataFrameFakeAnchorsWithInbetween["loopIDRight"] = [helper_function_swap_loopID(entry) for entry in finalDataFrameFakeAnchorsWithInbetween["loopIDRight"]]



#I just need to recalculate features for this case completely. Remodel the dataframe to have columns:
#chr, vp_middle, maxV4CscorePos, loopTissue. Then input that to make bed_anchors and bedIntersect functions

fakeAnchorsWithInBetweenDataFrameForBedFunction = finalDataFrameFakeAnchorsWithInbetween[["chr", "leftAnchorMiddlePos", "rightAnchorMiddlePos"]]
fakeAnchorsWithInBetweenDataFrameForBedFunction.columns = ["chr", "vp_middle", "maxV4CscorePos"]
fakeAnchorsWithInBetweenDataFrameForBedFunction["loopTissue"] = "WronglyConnectedAnchorsFakeLoop"

#filter out anchors that are too close together (closer than the minimum anchor size apart)
anchorsTooClose = fakeAnchorsWithInBetweenDataFrameForBedFunction[
        abs(fakeAnchorsWithInBetweenDataFrameForBedFunction["vp_middle"] - \
            fakeAnchorsWithInBetweenDataFrameForBedFunction["maxV4CscorePos"]) <= 10000]

#that's ~600 loops. Not a huge amount, but better to filter this from the 200000 loops beforehand.

#redo


totalDFFakeAnchorsToSampleFromWhereDistanceBetweenAnchorsMoreThan10000 = totalDFFalseLinkedAnchors[
        abs(totalDFFalseLinkedAnchors["leftAnchorMiddlePos"] - \
            totalDFFalseLinkedAnchors["rightAnchorMiddlePos"]) > 10001]

#removes about 5000 loops. That's really fine.


finalDataFrameFakeAnchorsWithInbetween = None
for num, bin_ in enumerate(binsAndFakeLoopNumbersToSampleFromThem[0]):
    toSampleCount = binsAndFakeLoopNumbersToSampleFromThem[1][num]
    toSampleFrom = totalDFFakeAnchorsToSampleFromWhereDistanceBetweenAnchorsMoreThan10000[
            totalDFFakeAnchorsToSampleFromWhereDistanceBetweenAnchorsMoreThan10000[
                    "binnedAccordingToRealData"] == bin_]
    print(toSampleFrom[toSampleFrom.loopSize <= 10000])
    sample = toSampleFrom.sample(n = toSampleCount)
    finalDataFrameFakeAnchorsWithInbetween = pd.concat([finalDataFrameFakeAnchorsWithInbetween, sample], axis = 0)

finalDataFrameFakeAnchorsWithInbetween["loopIDLeft"] = [helper_function_swap_loopID(entry) for entry in finalDataFrameFakeAnchorsWithInbetween["loopIDLeft"]]
finalDataFrameFakeAnchorsWithInbetween["loopIDRight"] = [helper_function_swap_loopID(entry) for entry in finalDataFrameFakeAnchorsWithInbetween["loopIDRight"]]

finalDataFrameFakeAnchorsWithInbetween[finalDataFrameFakeAnchorsWithInbetween.loopSize <= 10000]
#empty, as it should be

#I just need to recalculate features for this case completely. Remodel the dataframe to have columns:
#chr, vp_middle, maxV4CscorePos, loopTissue. Then input that to make bed_anchors and bedIntersect functions

fakeAnchorsWithInBetweenDataFrameForBedFunction = finalDataFrameFakeAnchorsWithInbetween[["chr", "leftAnchorMiddlePos", "rightAnchorMiddlePos"]]
fakeAnchorsWithInBetweenDataFrameForBedFunction.columns = ["chr", "vp_middle", "maxV4CscorePos"]
fakeAnchorsWithInBetweenDataFrameForBedFunction["loopTissue"] = "WronglyConnectedAnchorsFakeLoop"



finalDataFrameFakeAnchorsWithInbetween[
        abs(finalDataFrameFakeAnchorsWithInbetween["leftAnchorMiddlePos"] - finalDataFrameFakeAnchorsWithInbetween["rightAnchorMiddlePos"]) > 10000]



bedFormatFakeAnchorsWithInbetween = make_bed_format_anchors(fakeAnchorsWithInBetweenDataFrameForBedFunction,
                                                            True,
                                                            2500, 10000)

#calculate the intersects

intersectsWithChromatinMarksFakeLoopsWithInBetween = generate_intersect_BEDs(bedFormatFakeAnchorsWithInbetween,
                                                                             chromatinMarkDataFrame)

pathToSaveFakeLoopsWithInbetweenTo = "/2TB-disk/Project/Geert Geeven loops data/Data/ScriptOutput/FilesForHPCProcessing/BEDIntersectDataFrames_FakeLoops_WronglyPairedAnchors_ForFeatureCalculationWithInbetween_24_06_2019/"


make_dataframe_BED_save(intersectsWithChromatinMarksFakeLoopsWithInBetween,
                        pathToSaveFakeLoopsWithInbetweenTo)












