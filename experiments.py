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
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np
import pandas as pd
import h5py
import humanize
from scipy.stats import pearsonr

from sklearn.utils import shuffle, resample
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
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

np.set_printoptions(precision=4, linewidth=120)
pd.set_option("precision", 4)
pd.set_option("display.width", 300)


@dataclass(frozen=True)
class Results:
    experiment: str
    timestamp: int
    class_: str
    seed: int
    foldoutter: int
    foldinner: int
    classifier: str
    classifiercfg: int
    classifiercfgs: int
    f1binp: float
    f1binn: float
    f1micro: float
    f1macro: float
    f1weighted: float
    f1samples: float
    precision: float
    recall: float
    accuracy: float
    accuracy2: float
    timeinnertrain: float
    timeouttertrain: float
    positiveclasses: str
    negativeclasses: str
    features: str
    nfeaturesvar: int
    nfeaturestotal: int
    ynegfinaltrain: int
    yposfinaltrain: int
    ynegfinaltest: int
    yposfinaltest: int
    yposfinalpred: int
    ynegfinalpred: int
    yfinaltrain: int
    yfinaltest: int
    yfinalpred: int
    postrainsamples: int
    negtrainsamples: int
    postestsamples: int
    negtestsamples: int
    tp: int
    tn: int
    fp: int
    fn: int
    bestfeatureidx: int
    bestvariableidx: int
    featurerank: str  # em ordem decrescente, tem o IDX da feature
    rankfeature: str  # em ordem das features, tem o RANK de cada uma

    def __post_init__(self):
        pass


P = TypeVar("T")



def humantime(*args, **kwargs):
    """
    Return time (duration) in human readable format.

    >>> humantime(seconds=3411)
    56 minutes, 51 seconds
    >>> humantime(seconds=800000)
    9 days, 6 hours, 13 minutes, 20 seconds
    """
    secs = float(timedelta(*args, **kwargs).total_seconds())
    units = [("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)]
    parts = []
    for unit, mul in units:
        if secs / mul >= 1 or mul == 1:
            if mul > 1:
                n = int(math.floor(secs / mul))
                secs -= n * mul
            else:
                # n = secs if secs != int(secs) else int(secs)
                n = int(secs) if secs != int(secs) else int(secs)
            parts.append("%s %s%s" % (n, unit, "" if n == 1 else "s"))
    return ", ".join(parts)


def get_md5(params):
    import hashlib
    experiment = f'nr{params.nrounds}_nf{params.nfolds}_w{params.windowsize}_s{params.stepsize}'.encode('utf-8')
    return hashlib.md5(experiment).hexdigest()


def loggerthread(q):
    """
    Main process thread receiver (handler) for log records.
    """
    while True:
        record = q.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


def one_hot(array, num_classes):
    return np.squeeze(np.eye(num_classes)[array.reshape(-1)])

def one_hot_inverse(array):
    return np.argmax(array, axis=1)

def readdir(path) -> Dict[str, List[Tuple[np.ndarray, str]]]:
    """
    Read the CSV content of a directory into a list of numpy arrays.

    The return type is actually a dict with the "class" as key.
    """
    well_vars = [
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "T-JUS-CKGL",
        "QGL",
    ]
    columns = ["timestamp"] + well_vars + ["class"]
    r = []
    class_ = path[-1]
    with os.scandir(path) as it:
        for entry in it:
            if not entry.name.startswith(".") and entry.is_file():
                frame = pd.read_csv(entry, sep=",", header=0, names=columns)

                # str timestamp to float
                frame["timestamp"] = np.array(
                    [
                        pd.to_datetime(d).to_pydatetime().timestamp()
                        for d in frame.loc[:, "timestamp"]
                    ],
                    dtype=np.float64,
                )
                # cast int to float
                frame["class"] = frame["class"].astype(np.float64)

                # remember that scikit has problems with float64
                array = frame.loc[:, columns].to_numpy()

                r.append((array, entry.name))
    rd = {}
    rd[class_] = r
    return rd


def get_logging_config():
    return {
        "version": 1,
        "formatters": {
            "detailed": {
                "class": "logging.Formatter",
                "format": (
                    "%(asctime)s %(name)-12s %(levelname)-8s %(processName)-10s "
                    "%(module)-12s %(funcName)-15s %(message)s"
                ),
            }
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "level": "INFO",},
            "file": {
                "class": "logging.FileHandler",
                "filename": "experiment1a.log",
                "mode": "w",
                "formatter": "detailed",
            },
            "errors": {
                "class": "logging.FileHandler",
                "filename": "experiment1a_errors.log",
                "mode": "w",
                "level": "ERROR",
                "formatter": "detailed",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file", "errors"]},
    }


def readdirparallel(path):
    """
    Read all CSV content of a directory in parallel.
    """
    njobs = psutil.cpu_count()
    results = []
    # with Pool(processes=njobs) as p:
    with ThreadPoolExecutor(max_workers=njobs) as p:
        # with ProcessPoolExecutor(max_workers=njobs) as p:
        # results = p.starmap(
        results = p.map(
            readdir, [os.path.join(path, str(c)) for c in [0, 1, 2, 3, 4, 5, 6, 7, 8]],
        )
    return results


def csv2bin(*args, **kwargs) -> None:
    """
    Read 3W dataset CSV files and save in a single numpy binary file.
    """
    raise Exception("not implemented")


def csv2hdf(*args, **kwargs) -> None:
    """
    Read 3W dataset CSV files and save in a single HDF5 file.
    """
    path: str = kwargs.get("path")
    useclasses = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    well_vars = [
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "T-JUS-CKGL",
        "QGL",
    ]
    columns = ["timestamp"] + well_vars + ["class"]
    print("read CSV and save HDF5 ...", end="", flush=True)
    t0 = time.time()
    with h5py.File("datasets.h5", "w") as f:
        for c in useclasses:
            grp = f.create_group(f"/{c}")
            with os.scandir(os.path.join(path, str(c))) as it:
                for entry in it:
                    if not entry.name.startswith(".") and entry.is_file() and 'WELL' in entry.name:
                        frame = pd.read_csv(entry, sep=",", header=0, names=columns)

                        # str timestamp to float
                        frame["timestamp"] = np.array(
                            [
                                pd.to_datetime(d).to_pydatetime().timestamp()
                                for d in frame.loc[:, "timestamp"]
                            ],
                            dtype=np.float64,
                        )
                        # cast int to float
                        frame["class"] = frame["class"].astype(np.float64)

                        # remember that scikit has problems with float64
                        array = frame.loc[:, columns].to_numpy()

                        # entire dataset is float, incluinding timestamp & class labels
                        grp.create_dataset(
                            f"{entry.name}", data=array, dtype=np.float64
                        )
    print(f"finished in {time.time()-t0:.1}s.")


def csv2hdfpar(*args, **kwargs) -> None:
    """
    Read 3W dataset CSV files and save in a single HDF5 file.
    """
    path: str = kwargs.get("path")

    print("read CSV and save HDF5 ...", end="", flush=True)
    t0 = time.time()
    with h5py.File("datasets.h5", "w") as f:
        datalist = readdirparallel(path)
        for dd in datalist:
            for key in dd:
                grp = f.create_group(f"/{key}")
                for (array, name) in dd[key]:
                    grp.create_dataset(f"{name}", data=array, dtype=np.float64)
    print(
        f"finished {humanize.naturalsize(os.stat('datasets.h5').st_size)} "
        f"in {humantime(seconds=time.time()-t0)}."
    )


def cleandataset(*args, **kwargs) -> None:
    """
    Read the the single file (with whole dataset), remove NaN and save 1 file per class.
    """
    well_vars = [
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "T-JUS-CKGL",
        "QGL",
    ]
    columns = ["timestamp"] + well_vars + ["class"]
    print("Reading dataset...")
    with h5py.File("datasets.h5", "r") as f:
        for c in range(0, 9):
            print(f"Processing class {c}")
            k = f"/{c}"
            soma = 0
            for s in f[k]:
                n = f[k][s].shape[0]
                soma = soma + n

            data = np.zeros([soma, 10], dtype=np.float64)
            i1 = 0
            # manual concatenation
            for s in f[k]:
                i2 = i1 + f[k][s].shape[0]
                data[i1:i2, :] = f[k][s][()]
                i1 = i2
            frame = pd.DataFrame(data=data, columns=columns)
            for col in ["P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP"]:
                frame[col].fillna(method="ffill", axis=0, inplace=True)

            fp = np.memmap(
                f"datasets_clean_{c}.dat", dtype="float64", mode="w+", shape=frame.shape
            )
            fp[:, ...] = frame.to_numpy()
            del fp

    print("finished")


def cleandataseth5(*args, **kwargs) -> None:
    """
    Read the the single file (with whole dataset), remove NaN and save 1 file per class.
    """
    well_vars = [
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "T-JUS-CKGL",
        "QGL",
    ]
    columns = ["timestamp"] + well_vars + ["class"]

    logger = logging.getLogger(f"clean")
    formatter = logging.Formatter(
        "%(asctime)s %(name)-12s %(levelname)-8s %(lineno)-5d %(funcName)-10s %(module)-10s %(message)s"
    )
    fh = logging.FileHandler(f"experiments_clean.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

    usecols = [1, 2, 3, 4, 5]
    good = [columns[i] for i, _ in enumerate(columns) if i in usecols]

    with h5py.File("datasets.h5", "r") as f:
        logger.debug("reading input file")
        with h5py.File("datasets_clean.h5", "w") as fc:
            logger.debug("created output file")
            for c in range(0, 9):
                grp = fc.create_group(f"/{c}")
                logger.debug(f"Processing class {c}")
                k = f"/{c}"
                for s in f[k]:
                    if s[0] != "W":
                        continue

                    logger.debug(f"{c} {s}")

                    data = f[k][s][()]
                    frame = pd.DataFrame(data=data, columns=columns)
                    frame.dropna(inplace=True, how="any", subset=good, axis=0)
                    array = frame.to_numpy()

                    n = check_nan(array[:, [1, 2, 3, 4, 5]], logger)
                    if n > 0:
                        logger.info(f"{c} {s} dataset contains NaN")

                    grp.create_dataset(f"{s}", data=array, dtype=np.float64)

    return None


def check_nan(array, logger) -> None:
    """
    Check array for inf, nan of null values.
    """
    logger.debug("*" * 50)
    n = 0

    test = array[array > np.finfo(np.float32).max]
    logger.debug(f"test for numpy float32overflow {test.shape}")
    n = n + test.shape[0]

    test = array[~np.isfinite(array)]
    logger.debug(f"test for numpy non finite {test.shape}")
    n = n + test.shape[0]

    test = array[np.isinf(array)]
    logger.debug(f"test for numpy inf {test.shape}")
    n = n + test.shape[0]

    test = array[np.isnan(array)]
    logger.debug(f"test for numpy NaN {test.shape}")
    n = n + test.shape[0]

    test = array[pd.isna(array)]
    logger.debug(f"test for pandas NA {test.shape}")
    n = n + test.shape[0]

    test = array[pd.isnull(array)]
    logger.debug(f"test for pandas isnull {test.shape}")
    n = n + test.shape[0]

    logger.debug("*" * 50)

    return n


def get_config_combination_list(settings, default=None) -> List:
    """
    Given a list of hyperparameters return all combinations of that.
    """
    keys = list(settings)
    r = []
    for values in itertools.product(*map(settings.get, keys)):
        d = dict(zip(keys, values))
        if default is not None:
            d.update(default)
        r.append(d)
    return r


def get_classifiers(clflist, n_jobs=1, default=False) -> Dict:
    """
    Classifiers and combinations of hyperparameters.
    """
    classifiers = OrderedDict()
    classifiers["ADA"] = {
        "config": get_config_combination_list(
            {
                "n_estimators": [5, 25, 50, 75, 100, 250, 500, 1000], 
                "algorithm": ["SAMME", "SAMME.R"],
            },
            {
                "random_state": None
            },
        ),
        "default": {"random_state": None},
        "model": AdaBoostClassifier,
    }
    classifiers["DT"] = {
        "config": get_config_combination_list(
            {
                "criterion": ["gini", "entropy"],
                "splitter": ["best", "random"],
                "max_depth": [None, 5, 10, 50],
                "min_samples_split": [2, 5, 10],
            },
            {"random_state": None},
        ),
        "default": {"random_state": None},
        "model": DecisionTreeClassifier,
    }
    classifiers["GBOOST"] = {
        "config": get_config_combination_list(
            {
                "n_estimators": [50, 100, 250],
                "min_samples_split": [2, 5, 10],
                "max_depth": [5, 10, 50],
            },
            {"random_state": None},
        ),
        "default": {"random_state": None},
        "model": GradientBoostingClassifier,
    }
    classifiers["1NN"] = {
        "config": [],
        "default": {
            "n_neighbors": 1,
            "weights": "distance",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "n_jobs": 1,
        },
        "model": KNeighborsClassifier,
    }
    classifiers["5NN"] = {
        "config": [],
        "default": {
            "n_neighbors": 5,
            "weights": "distance",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "n_jobs": 1,
        },
        "model": KNeighborsClassifier,
    }
    classifiers["3NN"] = {
        "config": [],
        "default": {
            "n_neighbors": 3,
            "weights": "distance",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "n_jobs": 1,
        },
        "model": KNeighborsClassifier,
    }
    classifiers["KNN"] = {
        "config": get_config_combination_list(
            {
                "n_neighbors": [1, 3, 5, 7, 10, 15],
            },
            {"n_jobs": n_jobs}
        ),
        "default": {"n_jobs": n_jobs},
        "model": KNeighborsClassifier,
    }
    classifiers["RF"] = {
        "config": get_config_combination_list(
            {
                'bootstrap': [True],
                "criterion": ["gini"],
                'max_features': ['auto'],
                "max_depth": [None, 5, 10, 50],
                "min_samples_split": [2],
                'min_samples_leaf': [1],
                "n_estimators": [10, 25, 50, 100, 500],
            },
            {"n_jobs": n_jobs, "random_state": None},
        ),
        "default": {"random_state": None},
        "model": RandomForestClassifier,
    }
    classifiers["SVM"] = {
        "config": get_config_combination_list({
            #"kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            #"gamma": ["scale", "auto"],
            "gamma": [0.001, 0.01, 0.1, 1.0],
            #"C": [1.0, 2.0],
            "C": [1.0],
        }),
        "default": {},
        "model": SVC,
    }
    classifiers["GNB"] = {
        "config": [],
        "default": {},
        "model": "GaussianNB",
    }
    classifiers["LDA"] = {
        "config": [],
        "default": {},
        "model": "LinearDiscriminantAnalysis",
    }
    classifiers["QDA"] = {
        "config": [],
        "default": {}, 
        "model": "QuadraticDiscriminantAnalysis"
    }


    classifiers["MLP"] = {
        "config": get_config_combination_list({
            "hidden_layer_sizes": [128, 256, 512],
            "solver": ["lbfgs", "sgd", "adam"],
            "activation": ["relu"],
            "max_iter": [500],
            "shuffle": [True],
            "momentum": [0.9],
            "power_t": [0.5],
            "learning_rate": ["constant"],
            "batch_size": ["auto"],
            "alpha": [0.0001],
        }),
        "default": {"random_state": None},
        "model": MLPClassifier,
    }
    classifiers["ZERORULE"] = {
        "config": get_config_combination_list(
            {
                "strategy": [
                    "constant"
                ],
            },
            {"random_state": None},
        ),
        "default": {"strategy": "most_frequent", "random_state": None},
        "model": DummyClassifier,
    }


    if isinstance(clflist, str):
        if not clflist or clflist.lower() != "all":
            clflist = clflist.split(",")
    elif isinstance(clflist, tuple):
        clflist = list(clflist)

    if default:
        for c in classifiers.keys():
            """
            if c[:5] != "DUMMY":
                classifiers[c]['config'] = {}
            else:
                del classifiers[c]
            """
            classifiers[c]["config"] = {}
        return classifiers
    else:
        ret = {}
        for c in clflist:
            if c in classifiers.keys():
                ret[c] = classifiers[c]
        return ret


def _split_data(n: int, folds: int) -> Sequence[Tuple[int, int]]:
    """
    Return list of tuples with array index for each fold.
    """
    raise Exception("depends on which experiment")


def _read_and_split_h5(fold: int, params: P) -> Tuple:
    """
    HDF files offer at least a couple advantages:
    1 - reading is faster than CSV
    2 - you dont have to read the whole dataset to get its size (shape)

    H5PY fancy indexing is very slow.
    https://github.com/h5py/h5py/issues/413
    """
    raise Exception("depends on which experiment")


def _read_and_split_bin(fold: int, params: P) -> Tuple:
    """
    HDF files offer at least a couple advantages:
    1 - reading is faster than CSV
    2 - you dont have to read the whole dataset to get its size (shape)

    Numpy needs to know 'shape' beforehand.
    https://numpy.org/doc/stable/reference/generated/numpy.memmap.html

    """
    raise Exception("depends on which experiment")


def train_test_binary(
    classif, xtrain, ytrain, xtest, ytest
) -> Tuple[Tuple, Tuple, Tuple, Tuple]:
    """
    Execute training and testing (binary) and return metrics (F1, Accuracy, CM)
    """
    p = 0.0
    r = 0.0
    f1binp, f1binn = 0.0, 0.0
    f1mic = 0.0
    f1mac = 0.0
    f1weigh = 0.0
    f1sam = 0.0
    acc = 0.0
    excp = []
    excptb = []
    ypred = np.full([xtrain.shape[0]], 0, dtype="int")
    ynpred = 0
    yppred = 0
    tn = 0
    tp = 0
    fp = 0
    fn = 0
    trt1 = 0.0
    try:

        trt0 = time.time()
        classif.fit(xtrain, ytrain)
        trt1 = time.time() - trt0

        try:
            ypred = classif.predict(xtest)
            yppred = np.sum(ypred)
            ynpred = ytest.shape[0] - yppred

            try:
                p, r, f1binp, _ = precision_recall_fscore_support(
                    ytest, ypred, average="binary", pos_label=1,
                )
                _, _, f1binn, _ = precision_recall_fscore_support(
                    ytest, ypred, average="binary", pos_label=0,
                )
            except Exception as ef1bin:
                excp.append(ef1bin)

            try:
                f1mic = f1_score(ytest, ypred, average="micro")
            except Exception as e2:
                excp.append(e2)

            try:
                f1mac = f1_score(ytest, ypred, average="macro")
            except Exception as e3:
                excp.append(e3)

            try:
                f1weigh = f1_score(ytest, ypred, average="weighted")
            except Exception as e4:
                excp.append(e4)

            try:
                acc = accuracy_score(ytest, ypred)
            except Exception as e_acc:
                excp.append(e_acc)

            try:
                tn, fp, fn, tp = confusion_matrix(
                    ytest, ypred, labels=[0, 1], sample_weight=None, normalize=None,
                ).ravel()
            except Exception as ecm:
                excp.append(ecm)

        except Exception as e_pred:
            excp.append(e_pred)
            raise e_pred

    except Exception as efit:
        excp.append(efit)
        einfo = sys.exc_info()
        excptb.append(einfo[2])
        raise efit

    return (
        (ypred, ynpred, yppred, trt1),
        (f1binp, f1binn, f1mic, f1mac, f1weigh, f1sam, p, r, acc),
        (tn, fp, fn, tp),
        tuple(excp),
    )

def vertical_split_bin(negative, positive):
    x = np.concatenate([negative, positive], axis=0).astype(np.float32)
    y = np.concatenate(
        [
            np.zeros(negative.shape[0], dtype=np.int32),
            np.ones(positive.shape[0], dtype=np.int32),
        ],
        axis=0,
    )
    return x, y


def horizontal_split_file(fold, nfolds, files, seed):
    count = 0
    samples = []
    sessions = []
    for s in files:
        if s[0] != "W":
            continue
        n = files[s].shape[0]
        count += 1
        samples.append(n)
        sessions.append(s)

    samples = np.array(samples)
    nf = int(count / nfolds)

    testfidx = np.reshape(np.arange(nf * nfolds), (5, -1)).T
    testsize = sum(samples[testfidx[:, fold]])
    test = np.zeros((testsize, 10), dtype=np.float64)
    stest = [sessions[i] for i in testfidx[:, fold]]

    i1 = 0
    i2 = 0
    for s in stest:
        i2 = i1 + files[s].shape[0]
        test[i1:i2, :] = files[s][()]
        i1 = i2
    print(f"fold {fold} ate {i2}")

    nstp = 0
    for s in sessions:
        if s in stest:
            continue
        nstp += files[s].shape[0]

    train = np.zeros((nstp, 10), dtype=np.float64)
    i1 = 0
    i2 = 0
    for k, s in enumerate(sessions):
        if s in stest:
            continue
        n = files[s].shape[0]
        i2 = i1 + n
        train[i1:i2, :] = files[s][()]
        i1 = i2

    return train, test


def horizontal_split_well(fold, nfolds, file, seed=None):
    wellstest = [1, 2, 4, 5, 7]
    welltest = wellstest[fold]
    count = 0
    wells = {}
    stest = []
    for s in file:
        if s[0] != "W":
            continue
        n = file[s].shape[0]
        count += 1
        welli = int(str(s[6:10]))
        if welli not in wells:
            wells[welli] = 0
        wells[welli] += n
        if welli == welltest:
            stest.append(s)

    if wellstest[fold] in wells:
        ntest = wells[wellstest[fold]]
        test = np.zeros((ntest, 10), dtype=np.float64)
        i1 = 0
        i2 = 0
        for s in stest:
            if s[0] != "W":
                continue
            i2 = i1 + file[s].shape[0]
            test[i1:i2, :] = file[s][()]
            i1 = i2
    else:
        print("data for this fault and well not available")
        test = np.empty((0, 10), dtype=np.float64)

    ntrain = sum(wells[k] for k in wells if k != welltest)
    train = np.zeros((ntrain, 10), dtype=np.float64)
    i1 = 0
    i2 = 0
    for s in file:
        if s[0] != "W":
            continue
        if s in stest:
            continue
        i2 = i1 + file[s].shape[0]
        train[i1:i2, :] = file[s][()]
        i1 = i2
        # print('well', s, i1, i2, ntrain)

    return train, test


def drop_nan(*args):
    for a in args:
        mask = np.any(
            np.isnan(a)
            # | (trainnegative > np.finfo(np.float32).max)
            | np.isinf(a) | ~np.isfinite(a),
            axis=1,
        )
        a = a[~mask]
    return args


def get_mask(*args):
    m = []
    for a in args:
        mask = np.any(
            np.isnan(a)
            # | (trainnegative > np.finfo(np.float32).max)
            | np.isinf(a) | ~np.isfinite(a),
            axis=1,
        )
        m.append(mask)
    return m


def split_and_save1(params, case, group, classes):
    """
    """
    win = params.windowsize
    step = params.stepsize
    #filename = get_md5(params)

    with h5py.File(f"datasets_clean.h5", "r") as file:
        n = 0
        skipped = 0
        for c in classes:
            f = file[f"/{c}"]
            for s in f:
                if s[0] != "W":
                    continue
                if len(params.skipwell) > 0:
                    # skip well by ID
                    if s[:10] in params.skipwell:
                        skipped += f[s].shape[0]
                        continue
                n += f[s].shape[0]

        data = np.zeros([n, 10], dtype=np.float64)
        test = np.zeros((skipped, 10), dtype=np.float64)

        for c in classes:
            f = file[f"/{c}"]
            for s in f:
                i1, i2 = 0, 0
                j1, j2 = 0, 0
                for s in f:
                    if s[0] != "W":
                        continue
                    if len(params.skipwell) > 0:
                        if s[:10] in params.skipwell:
                            j2 = j1 + f[s].shape[0]
                            test[j1:j2, :] = f[s][()]
                            j1 = j2
                            continue
                    i2 = i1 + f[s].shape[0]
                    data[i1:i2, :] = f[s][()]
                    i1 = i2

        xdata = swfe(params.windowsize, n, params.stepsize, data[:, params.usecols],)
        tdata = swfe(
            params.windowsize, skipped, params.stepsize, test[:, params.usecols],
        )

        if group == "pos":
            ydata = np.ones(xdata.shape[0], dtype=np.float64)
        elif group == "neg":
            ydata = np.zeros(xdata.shape[0], dtype=np.float64)

        with h5py.File(f"datasets_folds_exp{case}.h5", "a") as ffolds:
            for round_ in range(1, params.nrounds + 1):
                if params.shuffle:
                    kf = KFold(
                        n_splits=params.nfolds, random_state=round_, shuffle=True
                    )
                else:
                    kf = KFold(n_splits=params.nfolds, random_state=None, shuffle=False)
                for fold, (train_index, test_index) in enumerate(kf.split(xdata)):
                    gk = f"/case{case}_{group}_r{round_}_nf{params.nfolds}_f{fold}_w{win}_s{step}"

                    if gk in ffolds:
                        del ffolds[gk]

                    grp = ffolds.create_group(gk)

                    xtrain, ytrain = xdata[train_index], ydata[train_index]
                    xtest, ytest = xdata[test_index], ydata[test_index]

                    print(
                        gk,
                        "original data shape",
                        data.shape,
                        "final",
                        xdata.shape,
                        "xtrain",
                        xtrain.shape,
                        "xtest",
                        xtest.shape,
                    )

                    grp.create_dataset(f"xtrain", data=xtrain, dtype=np.float64)
                    grp.create_dataset(f"ytrain", data=ytrain, dtype=np.float64)
                    grp.create_dataset(f"xvalid", data=xtest, dtype=np.float64)
                    grp.create_dataset(f"yvalid", data=ytest, dtype=np.float64)

                    if tdata.shape[0] > 0:
                        gkt = f"/case{case}_{group}_r{round_}_nf{params.nfolds}_f-test_w{win}_s{step}"
                        if gkt in ffolds:
                            del ffolds[gkt]
                        grpt = ffolds.create_group(gkt)
                        grpt.create_dataset(f"xtest", data=tdata, dtype=np.float64)


def split_and_save2(params, case, group, classes):
    win = params.windowsize
    step = params.stepsize
    nfolds = params.nfolds
    filename = get_md5(params)
    with h5py.File(f"datasets_clean.h5", "r") as file:
        with h5py.File(f"datasets_folds_exp{case}.h5", "a") as ffolds:
            for round_ in range(1, params.nrounds + 1):
                samples = []
                sessions = []
                for class_ in classes:
                    files = file[f"/{class_}"]
                    for s in files:
                        if s[0] != "W":
                            continue
                        n = files[s].shape[0]
                        samples.append(n)
                        sessions.append(f"/{class_}/{s}")
                count = len(samples)
                samples = np.array(samples)
                nf = int(count / nfolds)

                # random
                if params.shuffle:
                    testfidx = np.random.RandomState(round_).choice(
                        range(0, count), size=(nf, params.nfolds), replace=False,
                    )
                else:
                    # sequence
                    testfidx = np.reshape(np.arange(nf * nfolds), (5, -1)).T

                for fold in range(0, params.nfolds):
                    gk = f"/case{case}_{group}_r{round_}_nf{params.nfolds}_f{fold}_w{win}_s{step}"
                    testsize = sum(samples[testfidx[:, fold]])
                    test = np.zeros((testsize, 10), dtype=np.float64)
                    stest = [sessions[i] for i in testfidx[:, fold]]

                    i1, i2 = 0, 0
                    #for class_ in classes:
                    #    files = file[f"/{class_}"]
                    for s in stest:
                        i2 = i1 + file[s].shape[0]
                        test[i1:i2, :] = file[s][()]
                        i1 = i2
                        # print(s)
                    # print(f'fold {fold} ate {i2}')

                    nstp = 0
                    for s in sessions:
                        if s in stest:
                            continue
                        nstp += file[s].shape[0]

                    train = np.zeros((nstp, 10), dtype=np.float64)
                    i1, i2 = 0, 0
                    for s in sessions:
                        if s in stest:
                            continue
                        i2 = i1 + file[s].shape[0]
                        train[i1:i2, :] = file[s][()]
                        i1 = i2

                    xtrain = swfe(
                        params.windowsize,
                        nstp,
                        params.stepsize,
                        train[:, params.usecols],
                    )
                    if classes == params.negative:
                        ytrain = np.zeros(xtrain.shape[0], dtype=np.float64)
                    else:
                        ytrain = np.ones(xtrain.shape[0], dtype=np.float64)

                    xtest = swfe(
                        params.windowsize,
                        testsize,
                        params.stepsize,
                        test[:, params.usecols],
                    )

                    if classes == params.negative:
                        ytest = np.zeros(xtest.shape[0], dtype=np.float64)
                    else:
                        ytest = np.ones(xtest.shape[0], dtype=np.float64)

                    if gk in ffolds:
                        del ffolds[gk]

                    grp = ffolds.create_group(gk)

                    print(
                        gk,
                        "original data shape",
                        np.sum(samples),
                        "train",
                        train.shape,
                        "test",
                        test.shape,
                        "xtrain",
                        xtrain.shape,
                        "xtest",
                        xtest.shape,
                    )

                    grp.create_dataset(f"xtrain", data=xtrain, dtype=np.float64)
                    grp.create_dataset(f"ytrain", data=ytrain, dtype=np.float64)
                    grp.create_dataset(f"xvalid", data=xtest, dtype=np.float64)
                    grp.create_dataset(f"yvalid", data=ytest, dtype=np.float64)


def split_and_save3(params, case, group, classes):
    win = params.windowsize
    step = params.stepsize
    wellstest = [1, 2, 4, 5, 7]
    filename = get_md5(params)
    with h5py.File(f"datasets_clean.h5", "r") as clean:
        with h5py.File(f"datasets_folds_exp{case}.h5", "a") as ffolds:
            for round_ in range(1, params.nrounds + 1):
                for fold in range(0, params.nfolds):
                    gk = f"/case{case}_{group}_r{round_}_nf{params.nfolds}_f{fold}_w{win}_s{step}"
                    welltest = wellstest[fold]
                    count = 0
                    wells = {}
                    strain = []
                    stest = []
                    n = 0
                    for class_ in classes:
                        files = clean[f"/{class_}"]
                        for s in files:
                            if s[0] != "W":
                                continue
                            count += 1
                            welli = int(str(s[6:10]))
                            if welli not in wells:
                                wells[welli] = 0
                            wells[welli] += files[s].shape[0]
                            if welli == welltest:
                                stest.append(f"/{class_}/{s}")
                            else:
                                strain.append(f"/{class_}/{s}")

                    ntrain = sum(wells[k] for k in wells if k != welltest)
                    train = np.zeros((ntrain, 10), dtype=np.float64)

                    if wellstest[fold] in wells:
                        ntest = wells[wellstest[fold]]
                        test = np.zeros((ntest, 10), dtype=np.float64)
                        i1, i2 = 0, 0
                        for s in stest:
                            i2 = i1 + clean[s].shape[0]
                            test[i1:i2, :] = clean[s][()]
                            i1 = i2
                    else:
                        print("data for this fault and well not available")
                        test = np.empty((0, 10), dtype=np.float64)

                    i1, i2 = 0, 0
                    for s in strain:
                        i2 = i1 + clean[s].shape[0]
                        train[i1:i2, :] = clean[s][()]
                        i1 = i2

                    xtrain = swfe(
                        params.windowsize,
                        ntrain,
                        params.stepsize,
                        train[:, params.usecols],
                    )
                    if classes == params.negative:
                        ytrain = np.zeros(xtrain.shape[0], dtype=np.float64)
                    else:
                        ytrain = np.ones(xtrain.shape[0], dtype=np.float64)

                    xtest = swfe(
                        params.windowsize,
                        ntest,
                        params.stepsize,
                        test[:, params.usecols],
                    )

                    if classes == params.negative:
                        ytest = np.zeros(xtest.shape[0], dtype=np.float64)
                    else:
                        ytest = np.ones(xtest.shape[0], dtype=np.float64)
                    
                    if params.shuffle:
                        xtrain, ytrain = resample(xtrain, ytrain, random_state=round_, replace=False)
                        xtest, ytest = resample(xtest, ytest, random_state=round_, replace=False)

                    if gk in ffolds:
                        del ffolds[gk]

                    grp = ffolds.create_group(gk)

                    print(gk, "xtrain", xtrain.shape, "xtest", xtest.shape)

                    grp.create_dataset(f"xtrain", data=xtrain, dtype=np.float64)
                    grp.create_dataset(f"ytrain", data=ytrain, dtype=np.float64)
                    grp.create_dataset(f"xvalid", data=xtest, dtype=np.float64)
                    grp.create_dataset(f"yvalid", data=ytest, dtype=np.float64)


def foldfn(round_: int, fold: int, params: P) -> List[Dict]:
    """
    Run one fold.

    It can be executed in parallel.
    """
    logging.captureWarnings(True)
    logger = logging.getLogger(f"fold{fold}")
    formatter = logging.Formatter(params.logformat)
    fh = logging.FileHandler(f"{params.experiment}_fold{fold}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    logger.debug(f"round {round_}")
    logger.debug(f"fold {fold}")

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
    # 6, 7, 8 are gas lift related
    # usecols = [1, 2, 3, 4, 5]
    # usecols = params.usecols

    classifiers = get_classifiers(params.classifierstr)

    try:

        # ==============================================================================
        # Read data from disk and split in folds
        # ==============================================================================
        logger.debug(f"read and split data in folds")

        try:
            xtrainneg, xtrainpos, xtestneg, xtestpos, xfn, xfp = params.read_and_split(
                fold, round_, params
            )
        except Exception as e000:
            print(e000)
            raise e000

        x_outter_train, y_outter_train = vertical_split_bin(xtrainneg, xtrainpos)
        x_outter_test, y_outter_test = vertical_split_bin(xtestneg, xtestpos)

        if len(params.usecols) > 0:
            usecols = []
            for c in params.usecols:
                for ck in range((c - 1) * params.nfeaturesvar, c * params.nfeaturesvar):
                    usecols.append(ck)
            print('use measured variables', str(params.usecols), ' keep features ', str(usecols))
            logger.info('use measured variables' + str(params.usecols) + ' keep features ' + str(usecols))
            x_outter_train = x_outter_train[:, usecols]
            x_outter_test = x_outter_test[:, usecols]

        logger.debug(f"train neg={str(xtrainneg.shape)} pos={str(xtrainpos.shape)}")
        logger.debug(f"test  neg={str(xtestneg.shape)}  pos={str(xtestpos.shape)}")

        if 0 in xtestpos.shape or 0 in xtestneg.shape:
            breakpoint()
            raise Exception("dimension zero")

        if xtestpos.shape[0] > xtestneg.shape[0]:
            #
            # print('Binary problem with unbalanced classes: NEG is not > POS')
            logger.warn("Binary problem with unbalanced classes: NEG is not > POS")
            # raise Exception()

        logger.debug(
            f"shapes train={str(x_outter_train.shape)} test={str(x_outter_test.shape)}"
        )
        # ==============================================================================
        # After feature extraction, some NaN appear again in the arrays
        # ==============================================================================
        logger.debug(f"check NaN #1")
        mask = np.any(
            np.isnan(x_outter_train)
            | (x_outter_train > np.finfo(np.float32).max)
            | np.isinf(x_outter_train)
            | ~np.isfinite(x_outter_train),
            axis=1,
        )
        # breakpoint()
        logger.debug(f"NaN Mask size {np.count_nonzero(mask, axis=0)}")

        x_outter_train = x_outter_train[~mask]
        y_outter_train = y_outter_train[~mask]

        mask = np.any(
            np.isnan(x_outter_test)
            | (x_outter_test > np.finfo(np.float32).max)
            | np.isinf(x_outter_test)
            | ~np.isfinite(x_outter_test),
            axis=1,
        )
        # mask = ~mask
        x_outter_test = x_outter_test[~mask]
        y_outter_test = y_outter_test[~mask]

        logger.debug(f"check NaN #2")
        check_nan(x_outter_train, logger)

        logger.debug(
            f"shapes train={str(x_outter_train.shape)} test={str(x_outter_test.shape)}"
        )

        # ==================
        # Feature selection
        # ==================
        # 0 - MAX
        # 1 - Mean
        # 2 - Median
        # 3 - Min
        # 4 - Std
        # 5 - Var
        # usefeatures = [1, 4, 5]
        # usefeatures = [0, 1, 2, 3, 4, 5]
        # logger.info('Use features ' + str(usefeatures))
        # x_outter_train = x_outter_train[:, usefeatures]
        # x_outter_test = x_outter_test[:, usefeatures]

        # ==============================================================================
        # Normalization
        # ==============================================================================
        logger.debug(f"normalization AFTER feature extraction")
        scalerafter = StandardScaler()
        scalerafter.fit(x_outter_train)
        x_outter_train = scalerafter.transform(x_outter_train)
        x_outter_test = scalerafter.transform(x_outter_test)

        # ==============================================================================
        # Covariance and Person's Correlation
        # ==============================================================================
        logger.info("Covariance and correlation - over features")

        nc = len(usecols) * 6
        corr1 = np.zeros((nc, nc), dtype=np.float64)
        pers1 = np.zeros((nc, nc), dtype=np.float64)
        corr2 = np.zeros((nc, nc), dtype=np.float64)
        pers2 = np.zeros((nc, nc), dtype=np.float64)
        for a, b in itertools.combinations(range(0, nc), 2):
            try:
                corr1[a, b], pers1[a, b] = pearsonr(
                    x_outter_train[:, a], x_outter_train[:, b]
                )
            except:
                corr1[a, b], pers1[a, b] = 0.0, 0.0
            try:
                corr2[a, b], pers2[a, b] = pearsonr(
                    x_outter_test[:, a], x_outter_test[:, b]
                )
            except:
                corr2[a, b], pers2[a, b] = 0.0, 0.0

        logger.info("Pearson R train \n" + str(corr1))
        logger.info("Pearson R test  \n" + str(corr2))

        resultlist = []

        for clf in classifiers:
            logger.debug(f'Classifier {clf}')
            if isinstance(classifiers[clf]["model"], str):
                model = eval(classifiers[clf]["model"])
            elif callable(classifiers[clf]["model"]):
                model = classifiers[clf]["model"]

            if params.gridsearch != 0:
                #raise Exception("not implemented")
                
                inner = []
                innerkf = KFold(n_splits=4, shuffle=True, random_state=round_)
                nxotr = x_outter_train.shape[0] // 4
                for kfi, (itrainidx, itestidx) in enumerate(innerkf.split(x_outter_train)):
                    logger.debug(f'inner fold {kfi}')

                    x_inner_train = x_outter_train[itrainidx, :]
                    y_inner_train = y_outter_train[itrainidx]

                    x_inner_test = x_outter_train[itestidx, :]
                    y_inner_test = y_outter_train[itestidx]

                    for cfgk, cfg in enumerate(classifiers[clf]["config"]):
                        logger.debug(f'classifier config {cfgk+1:3d}/{len(classifiers[clf]["config"])}')
                        if clf == "ELM":
                            cfg['inputs'] = x_inner_train.shape[1]
                            yintr = one_hot(y_inner_train, 2)
                        else:
                            yintr = y_inner_train
                        classif_in = model(**cfg)
                        f1_in = 0.0
                        trt0 = time.time()
                        try:
                            classif_in.fit(x_inner_train, yintr)
                            trt1 = time.time() - trt0

                            p_in, r_in, f1_in = 0.0, 0.0, 0.0

                            try:
                                ypred_in = classif_in.predict(x_inner_test)

                                try:
                                    p_in, r_in, f1_in, _ = precision_recall_fscore_support(
                                        y_inner_test, ypred_in, average="macro",
                                    )
                                    logger.debug(f'classifier config {cfgk:3d} F1={f1_in:.4f} time={trt1:11.4f}')
                                except Exception as inef1:
                                    print(cfgk, inef1)
                            except Exception as inetr:
                                print(cfgk, inetr)
                        except Exception as e_inner:
                            logger.exception(e_inner)
                        
                        inner.append({
                            'config': cfgk,
                            'metric': f1_in,
                        })

                        del classif_in

                validf = pd.DataFrame(data=inner)
                valid = pd.pivot_table(validf, index='config', values=['metric'], aggfunc={'metric': ['mean']})
                valid = valid['metric']
                valid = valid.reindex(
                    valid.sort_values(by=['mean', 'config'], axis=0, ascending=[False, True], inplace=False).index
                )
                idx = list(valid.index)[0]
                logger.debug(f'best config {idx}, F1={valid.iat[0, 0]}')
                best_config = classifiers[clf]["config"][idx]

                # ===============
                # FIM GRID SEARCH
                # ===============

            else:
                idx = 0
                best_config = classifiers[clf]["default"]

            if "random_state" in best_config:
                best_config["random_state"] = round_

            classif = None

            if clf == "ZERORULE":
                pass
            if clf == "ELM":
                best_config['inputs'] = x_outter_train.shape[1]
                youttr = one_hot(y_outter_train, 2)
            else:
                youttr = y_outter_train

            classif = model(**best_config)

            r1, r2, r3, r4 = train_test_binary(
                classif, x_outter_train, youttr, x_outter_test, y_outter_test
            )
            y_outter_pred, ynpred, yppred, traintime = r1
            f1bin4, f1bin0, f1mic, f1mac, f1weigh, f1sam, p, r, acc = r2
            tn, fp, fn, tp = r3

            for exp in r4:
                logger.exception(exp)

            logger.info(f"Classifier {clf} acc={acc:.4f} f1mac={f1mac:.4f} f1bin4={f1bin4:.4f}  f1bin0={f1bin0:.4f}")

            resultlist.append(
                vars(
                    Results(
                        class_="4",
                        experiment=params.experiment,
                        nfeaturestotal=0,
                        timestamp=int(f"{params.sessionts:%Y%m%d%H%M%S}"),
                        seed=round_,
                        foldoutter=fold,
                        foldinner=-1,
                        classifier=clf,
                        classifiercfg=idx,
                        classifiercfgs=len(classifiers[clf]["config"]),
                        f1binp=f1bin4,
                        f1binn=f1bin0,
                        f1micro=f1mic,
                        f1macro=f1mac,
                        f1weighted=f1weigh,
                        f1samples=f1sam,
                        precision=p,
                        recall=r,
                        accuracy=acc,
                        accuracy2=0.0,
                        timeinnertrain=0,
                        timeouttertrain=traintime,
                        positiveclasses="4",
                        negativeclasses="0",
                        features="",
                        nfeaturesvar=6,
                        postrainsamples=0,
                        negtrainsamples=0,
                        postestsamples=0,
                        negtestsamples=0,
                        ynegfinaltrain=xtrainneg.shape[0],
                        yposfinaltrain=xtrainpos.shape[0],
                        ynegfinaltest=xtestneg.shape[0],
                        yposfinaltest=xtestpos.shape[0],
                        yposfinalpred=yppred,
                        ynegfinalpred=ynpred,
                        yfinaltrain=y_outter_train.shape[0],
                        yfinaltest=y_outter_test.shape[0],
                        yfinalpred=y_outter_pred.shape[0],
                        tp=tp,
                        tn=tn,
                        fp=fp,
                        fn=fn,
                        bestfeatureidx='',
                        bestvariableidx='',
                        featurerank='',
                        rankfeature='',
                    )
                )
            )

            if len(params.skipwell) > 0 and xfn is not None:
                yft = np.concatenate(
                    (
                        np.zeros(xfn.shape[0], dtype=np.int32),
                        np.ones(xfp.shape[0], dtype=np.int32),
                    ),
                    axis=0,
                )
                try:
                    yftpred = classif.predict(np.concatenate((xfn, xfp), axis=0))
                    try:
                        p, r, f1bin, _ = precision_recall_fscore_support(
                            yft, yftpred, average="binary",
                        )
                    except Exception as ef1bin:
                        print(ef1bin)

                    acc = 0.0
                    try:
                        acc = accuracy_score(yft, yftpred)
                    except Exception as eacc:
                        print(eacc)
                        raise eacc

                    resultlist.append(
                        vars(
                            Results(
                                class_="4",
                                experiment=params.experiment,
                                nfeaturestotal=0,
                                timestamp=int(f"{params.sessionts:%Y%m%d%H%M%S}"),
                                seed=round_,
                                foldoutter=-1,
                                foldinner=-1,
                                classifier=clf,
                                classifiercfg=idx,
                                f1bin=f1bin,
                                f1micro=0,
                                f1macro=0,
                                f1weighted=0,
                                f1samples=0,
                                precision=p,
                                recall=r,
                                accuracy=0.0,
                                accuracy2=0.0,
                                timeinnertrain=0,
                                timeouttertrain=0,
                                positiveclasses="4",
                                negativeclasses="0",
                                features="",
                                nfeaturesvar=6,
                                postrainsamples=0,
                                negtrainsamples=0,
                                postestsamples=0,
                                negtestsamples=0,
                                ynegfinaltrain=0,
                                yposfinaltrain=0,
                                ynegfinaltest=xfn.shape[0],
                                yposfinaltest=xfp.shape[0],
                                yposfinalpred=0,
                                ynegfinalpred=0,
                                yfinaltrain=0,
                                yfinaltest=0,
                                yfinalpred=0,
                                tp=0,
                                tn=0,
                                fp=0,
                                fn=0,
                            )
                        )
                    )
                except Exception as eft:
                    print(eft)
                    raise eft

        return resultlist

    except Exception as efold:
        logger.exception(efold)
        breakpoint()
        raise efold

    return []


def runexperiment(params, *args, **kwargs) -> None:
    """
    Run experiment - train, validation (optional) and test.
    """
    all_results_list = []
    partial = []

    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(params.logformat)
    fh = logging.FileHandler(f"{params.experiment}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    logger.info(params.name)
    logger.info("=" * 79)

    t0 = time.time()

    for round_ in range(1, params.nrounds + 1):
        logger.info(f"Round {round_}")
        if params.njobs > 1:
            logger.debug(f"Running {params.njobs} parallel jobs")
            with Pool(processes=params.njobs) as p:
                # 'results' is a List of List
                results = p.starmap(
                    foldfn,
                    zip(
                        [round_] * params.nfolds,
                        range(params.nfolds),
                        [params] * params.nfolds,
                    ),
                )
            results_list_round = []
            for r in results:
                results_list_round.extend(r)
        else:
            logger.debug(f"Running single core")
            results_list_round = []
            for foldout in range(0, params.nfolds):
                results_list_round.extend(foldfn(round_, foldout, params))

        partial = pd.DataFrame(data=results_list_round)
        try:
            partial.to_excel(f"{params.experiment}_parcial_{round_}.xlsx", index=False)
        except Exception as e1:
            logger.exception(e1)

        all_results_list.extend(results_list_round)

    results = pd.DataFrame(data=all_results_list)
    results["e"] = params.experiment[:-1]
    results["case"] = params.experiment[-1]

    try:
        results.to_excel(f"{params.experiment}_final.xlsx", index=False)
    except Exception as e2:
        logger.exception(e2)

    try:
        markdown = str(pd.pivot_table(
            data=results[results["foldoutter"] >= 0],
            values=["f1macro", "ynegfinaltest", "yposfinaltest"],
            index=["classifier", "e"],
            aggfunc={
                "f1macro": ["mean", "std"],
                "ynegfinaltest": "max",
                "yposfinaltest": "max",
            },
            columns=["case"],
        ).to_markdown(buf=None, index=True))
        logger.debug(markdown)
    except:
        pass

    try:

        with open(f"{params.experiment}_final.md", "w") as f:
            f.writelines("\n")
            f.writelines("All rounds\n")
            pd.pivot_table(
                data=results[results["foldoutter"] >= 0],
                values=["f1macro", "ynegfinaltest", "yposfinaltest", "class_"],
                index=["classifier", "e"],
                aggfunc={
                    "f1macro": ["mean", "std"],
                    "ynegfinaltest": "max",
                    "yposfinaltest": "max",
                    "class_": "count"
                },
                columns=["case"],
            ).to_markdown(buf=f, index=True)
            f.writelines("\n\n")
            f.writelines(
                # folds de "teste"
                pd.pivot_table(
                    data=results[results["foldoutter"] < 0],
                    values=["f1macro", "ynegfinaltest", "yposfinaltest"],
                    index=["classifier", "e"],
                    aggfunc={
                        "f1macro": ["mean", "std"],
                        "ynegfinaltest": "max",
                        "yposfinaltest": "max",
                    },
                    columns=["case"],
                ).to_markdown(buf=None, index=True)
            )
            f.writelines("\n\n\n")
            f.writelines("Round 1\n")
            pd.pivot_table(
                data=results[(results["foldoutter"] >= 0) & (results["foldinner"] < 0) &(results["seed"] == 1)],
                values=["f1macro", "ynegfinaltest", "yposfinaltest", "class_"],
                index=["classifier", "e"],
                aggfunc={
                    "f1macro": ["mean", "std"],
                    "ynegfinaltest": "max",
                    "yposfinaltest": "max",
                    "class_": "count",
                },
                columns=["case"],
            ).to_markdown(buf=f, index=True)
            f.writelines("\n")

    except Exception as e3:
        logger.exception(e3)

    logger.debug(f"finished in {humantime(seconds=(time.time()-t0))}")


@dataclass(frozen=False)
class DefaultParams:
    """
    Experiment configuration (and model hyperparameters)
    DataClass offers a little advantage over simple dictionary in that it checks if
    the parameter actually exists. A dict would accept anything.
    """

    name: str = ""
    experiment: str = ""
    nrounds: int = 1
    nfolds: int = 5
    njobs: int = 1
    windowsize: int = 900
    stepsize: int = 900
    gridsearch: int = 0
    classifierstr: str = "1NN,3NN,QDA,LDA,GNB,RF,ZERORULE"
    usecolsstr: str = "1,2,3,4,5"
    usecols: list = field(default_factory=list)
    nfeaturesvar: int = 6
    hostname: str = socket.gethostname()
    ncpu: int = psutil.cpu_count()
    datasetcols: list = field(default_factory=list)
    tzsp = tz.gettz("America/Sao_Paulo")
    logformat: str = "%(asctime)s %(levelname)-8s  %(name)-12s %(funcName)-12s %(lineno)-5d %(message)s"
    shuffle: bool = True
    skipwellstr: str = ""
    read_and_split = None

    def __post_init__(self):
        self.njobs = max(min(self.nfolds, self.njobs, self.ncpu), 1)
        if isinstance(self.classifierstr, str):
            self.classifiers = self.classifierstr.split(",")
        elif isinstance(self.classifierstr, tuple):
            self.classifiers = list(self.classifierstr)
        self.skipwell = self.skipwellstr.split(",")
        self.sessionts = datetime.now(tz=self.tzsp)
        if isinstance(self.usecolsstr, str):
            self.usecols = list(map(int, self.usecolsstr.split(",")))
        elif isinstance(self.usecolsstr, tuple):
            self.usecols = list(self.usecolsstr)
        elif isinstance(self.usecolsstr, int):
            self.usecols = [self.usecolsstr]
        self.datasetcols = [
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


def concat_excel(*files, output='compilado.xlsx'):
    lista = []
    for f in files:
        frame = pd.read_excel(f, header=0)
        lista.append(frame)
    final = pd.concat(lista, axis=0)

    final['case'] = list(map(lambda x: x[-2:-1], final['experiment']))
    final['scenario'] = list(map(lambda x: x[-1], final['experiment']))
    final['welltest'] = 0
    final.to_excel(output, index=False)


if __name__ == "__main__":

    fire.Fire(
        {
            "concatexcel": concat_excel,
            "cleandataset": cleandataset,
            "cleandataseth5": cleandataseth5,
            "csv2hdf": csv2hdf,
            "csv2hdfpar": csv2hdfpar,
        }
    )
