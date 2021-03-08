import os
import sys
import math
import itertools
import warnings
import json
import time
import psutil
import socket
import shelve
import threading
import traceback
import inspect
import logging
from multiprocessing import Pool, Value, Manager, Queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from io import StringIO
from types import SimpleNamespace
from typing import Tuple, Dict, List, Any, Union, TypeVar, Type, Sequence
from datetime import datetime, timedelta
from dateutil import tz
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
import pandas as pd
import h5py
import humanize

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
    confusion_matrix,
    confusion_matrix,
)
from sklearn.exceptions import UndefinedMetricWarning

from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import fire

from utils.libutils import swfe  # pylint: disable-msg=E0611

from experiments import (
    runexperiment,
    get_config_combination_list,
    get_logging_config,
    loggerthread,
    cleandataset,
    humantime,
    readdir,
    csv2hdf,
    foldfn,
    check_nan,
    csv2hdfpar,
    get_classifiers,
    train_test_binary,
    horizontal_split_well,
    DefaultParams,
    split_and_save3,
)


@dataclass(frozen=False)
class Params(DefaultParams):
    """
    Experiment configuration (and model hyperparameters)

    DataClass offers a little advantage over simple dictionary in that it checks if
    the parameter actually exists. A dict would accept anything.
    """

    name = "Experiment 02B"
    experiment: str = "experiment2b"
    nrounds: int = 1
    nfolds: int = 5
    njobs: int = 1

    classifierstr: str = "1NN,QDA,LDA,GNB,ZERORULE"
    windowsize: int = 900
    stepsize: int = 900
    gridsearch: int = 0
    hostname: str = socket.gethostname()
    ncpu: int = psutil.cpu_count()
    shuffle: bool = True
    tzsp = tz.gettz("America/Sao_Paulo")
    datasetcols = [
        "timestamp",
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "T-JUS-CKGL",
        "QGL",
        "class",
    ]
    usecols = [1, 2, 3, 4, 5]
    read_and_split = None
    shuffle: bool = True
    logformat = "%(asctime)s %(levelname)-8s  %(name)-12s %(funcName)-12s %(lineno)-5d %(message)s"

    def __post_init__(self):
        super().__post_init__()
        self.positive = [4]
        self.negative = [0, 1, 2, 3, 5, 6, 7, 8]


def read_and_split_h5(fold: int, round_: int, params: Params) -> Tuple:
    """
    HDF files offer at least a couple advantages:
    1 - reading is faster than CSV
    2 - you dont have to read the whole dataset to get its size (shape)

    H5PY fancy indexing is very slow.
    https://github.com/h5py/h5py/issues/413
    """
    win = params.windowsize
    step = params.stepsize
    case = "2b"
    with h5py.File(f"datasets_folds_exp{case}.h5", "r") as ffolds:

        group = "pos"
        gk = f"/case{case}_{group}_r{round_}_nf{params.nfolds}_f{fold}_w{win}_s{step}"
        trainpositive = ffolds[gk]["xtrain"][()]
        testpositive = ffolds[gk]["xvalid"][()]

        group = "neg"
        gk = f"/case{case}_{group}_r{round_}_nf{params.nfolds}_f{fold}_w{win}_s{step}"
        trainnegative = ffolds[gk]["xtrain"][()]
        testnegative = ffolds[gk]["xvalid"][()]

        gk = f"/case{case}_{group}_r{round_}_nf{params.nfolds}_f-test_w{win}_s{step}"

    return trainnegative, trainpositive, testnegative, testpositive, None, None


def run_experiment(*args, **kwargs):
    params = Params(**kwargs)
    params.read_and_split = read_and_split_h5
    return runexperiment(params, *args, **kwargs)


def split_folds_wells(*args, **kwargs):
    params = Params(**kwargs)
    split_and_save3(params, "2b", "pos", params.positive)
    split_and_save3(params, "2b", "neg", params.negative)


if __name__ == "__main__":
    fire.Fire(
        {
            "runexperiment": run_experiment,
            "csv2hdf": csv2hdf,
            "csv2hdfpar": csv2hdfpar,
            "cleandataset": cleandataset,
            "splitfolds": split_folds_wells,
        }
    )
