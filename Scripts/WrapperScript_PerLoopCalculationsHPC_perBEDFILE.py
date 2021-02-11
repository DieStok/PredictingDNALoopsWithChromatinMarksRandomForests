#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:03:17 2019

@author: dstoker

#update 27-07-2019

This script starts jobs to calculate features. A separate job
is started for each loop, and every started job calculates features
in all loop areas (LeftAnchor Left and Right, RightAnchor Left and Right, Inbetween if present)
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



#new wrapper script.



#the below is deprecated. In actuality, this script reads in all the BED files and assigns those,
#a csv per loop did not work.

######DEPRECATED
#idea
#step 1. generate dataframes from each bed intersect file.
#step 2. for each dataframe, save each group (loop) to a separate csv file 
    #bearing the name of said loop, and the anchor.
#step 3. make a wrapper script that runs the feature calculations
    #for each loop separately, and saves the output to a file with the loopname
#step 4. at the end, have the wrapper script read every file, and combine
    #into one final numpy feature array.
    
    
#for step 3: have the wrapper read from a specified folder all the files
    #
#######


def main(argv):
   outputFilesPath = ""
   inputFilesPath  = ""
   try:
      opts, args = getopt.getopt(argv,"h",["outputFilesPath=", "inputFilesPath="])
   except getopt.GetoptError:
      print('test.py --inputFilesPath <path to all separate loop .csv files> --outputFilesPath' + 
            "<path to store output files of features per loop>")
      #print("Also, add a trailing slash to the outputFilesPath")
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py --inputFilesPath <path to all separate loop .csv files> --outputFilesPath' + 
            "<path to store output files of features per loop>")
         #print("Also, add a trailing slash to the outputFilesPath")
         sys.exit()
      elif opt == "--outputFilesPath":
          outputFilesPath =  arg
          if outputFilesPath.endswith("/"):
              pass
          else:
              outputFilesPath = outputFilesPath + "/"
      elif opt == "--inputFilesPath": 
          inputFilesPath = arg
   fileList = []
   fileNameList = []      
   for file in glob.glob(inputFilesPath + "/" + "*.csv"):
        fileList.append(file)
        fileNameList.append(os.path.splitext(os.path.split(file)[1])[0])
    
    #read a csv to get an IDList
   dataFrame = pd.read_csv(fileList[0])
   columnWithIDs = dataFrame.loc[:, "areaType"].tolist()
   splittedIDList = [entry.split("_") for entry in columnWithIDs]
   IDList = set([splitted[2] + "_" + splitted[3] for splitted in splittedIDList])
    
   print("Sending out loop calculation jobs to queue.")
   for loop in IDList:
       fileNamesToGiveFeatureCalc = fileList
       os.system("qsub -cwd -N DieterLoopCalc" + " " +
                 #"-o " + "/home/cog/dstoker/logs/" + loop + "_log.out " +
                 "/hpc/cog_bioinf/ridder/users/dstoker/scripts/run_script.sh " + "/hpc/cog_bioinf/ridder/users/dstoker/scripts/featureCalcPerLoop_perBEDFILE.py " +
                 outputFilesPath + " " + " ".join(fileNamesToGiveFeatureCalc) + " " + loop)
   print("All loops have been sent for calculation.")
    #or perhaps use os.system("")
    
    #then, start a job that combines all those outputs. For that job, i can just write a script
    #that reads all these csv files, and then use
    #qsub with hold_jid DieterLoopCalc, so that it only happens when all those jobs are done.
    
    #outputFilesPath is passed twice because the script wants both an input and an output
    #directory. It is best to cluster all data together, so I keep that the same directory
    #for now. Could change this with options in the future.

   os.system("qsub -hold_jid DieterLoopCalc -N DieterLoopCombine  -l h_rt=02:00:00 -l h_vmem=100G " + #"-o " + "loopCombine_log" +
              "/hpc/cog_bioinf/ridder/users/dstoker/scripts/run_script.sh " + "python " +
              "/hpc/cog_bioinf/ridder/users/dstoker/scripts/combineFeatureOutputHPC_04_04_nongzippedOutput.py " +
              outputFilesPath + " " + outputFilesPath)

    #"find . -path " + outputFilesPath  -name \"*.csv\" -print0 | xargs -I {} -0 cat {} | grep -v ',loopID,anchorType' | gzip > NAMEOFRESULT.csv.gz"


if __name__ == "__main__":
   main(sys.argv[1:])          
         
          
          
    