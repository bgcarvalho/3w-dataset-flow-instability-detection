#!/bin/bash
date

# ============ Folds ==== Samples
# 0 => 597   -     - 
# 1 =>   5 114     5 
# 2 =>  22  16     5 
# 3 =>  32  74     4 
# 4 => 344   -    10 
# 5 =>  12 439     6 
# 6 =>   6 215     6 
# 7 =>   4   -     4 
# 8 =>   3  81     3 

# ============ ===== 
# 0 => timestamp
# 1 "P-PDG",
# 2 "P-TPT",
# 3 "T-TPT",
# 4 "P-MON-CKP",
# 5 "T-JUS-CKP",
# 6 "P-JUS-CKGL",
# 7 "T-JUS-CKGL",
# 8 "QGL",
# 9 => class label

# ==================
# Features
# 0 - MAX
# 1 - Mean
# 2 - Median
# 3 - Min
# 4 - Std
# 5 - Var

PYTHON="python3"
CLS="1NN,RF,QDA,LDA,GNB,ZERORULE"
NR=1
NJ=5  # compute folds in parallel
UC="1,2,3,4,5"
SW=900
FOLDER="results"
GRID_SEARCH=0

mkdir $FOLDER

# this takes about 15 minutes in my machine
$PYTHON experiments.py csv2hdf --path="3w_dataset/"

# this goes fast (a few seconds)
$PYTHON experiments.py cleandataseth5

# this takes a few min
$PYTHON experiment1a.py splitfolds --windowsize=$SW --nrounds=$NR
$PYTHON experiment1b.py splitfolds --windowsize=$SW --nrounds=$NR
$PYTHON experiment2a.py splitfolds --windowsize=$SW --nrounds=$NR
$PYTHON experiment2b.py splitfolds --windowsize=$SW --nrounds=$NR

$PYTHON experiment1a.py runexperiment --njobs=$NJ --classifierstr=$CLS --nrounds=$NR --usecolsstr=$UC --windowsize=$SW --gridsearch=$GRID_SEARCH
mv -f experiment1a_final.md   $FOLDER/experiment1a_final.md
mv -f experiment1a_final.xlsx $FOLDER/experiment1a_final.xlsx

$PYTHON experiment1b.py runexperiment --njobs=$NJ --classifierstr=$CLS --nrounds=$NR --usecolsstr=$UC --windowsize=$SW --gridsearch=$GRID_SEARCH
mv -f experiment1b_final.md   $FOLDER/experiment1b_final.md
mv -f experiment1b_final.xlsx $FOLDER/experiment1b_final.xlsx

$PYTHON experiment2a.py runexperiment --njobs=$NJ --classifierstr=$CLS --nrounds=$NR --usecolsstr=$UC --windowsize=$SW --gridsearch=$GRID_SEARCH
mv -f experiment2a_final.md   $FOLDER/experiment2a_final.md
mv -f experiment2a_final.xlsx $FOLDER/experiment2a_final.xlsx

$PYTHON experiment2b.py runexperiment --njobs=$NJ --classifierstr=$CLS --nrounds=$NR --usecolsstr=$UC --windowsize=$SW --gridsearch=$GRID_SEARCH
mv -f experiment2b_final.md   $FOLDER/experiment2b_final.md
mv -f experiment2b_final.xlsx $FOLDER/experiment2b_final.xlsx

date
