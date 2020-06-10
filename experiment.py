#!/usr/bin/env python
# encoding: utf-8
# =============================================================================
# Experiment
# ----------
# Bruno Carvalho
# Last Updated: 2020-06-08 12:00 UTC-3
# https://github.com/bgcarvalho/3w-dataset_flow-instability-detection
# =============================================================================
import os
import sys
import itertools
import glob
import warnings
import time
import re
from copy import copy
from datetime import datetime
from dateutil import tz
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning

from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

def get_results_frame():
    return pd.DataFrame(columns=[
        'ts',
        'class',
        'seed',
        'fold_outter',
        'fold_inner',
        'classifier',
        'clf_cfg',
        'F1',
        'precision',
        'recall',
        'accuracy',
        'time',
        'window_size',
        'step',
        '0_files_train',
        '0_files_valid',
        '0_files_test',
        '4_files_train',
        '4_files_valid',
        '4_files_test',
        '0_n_files_train',
        '0_n_files_valid',
        '0_n_files_test',
        '4_n_files_train',
        '4_n_files_valid',
        '4_n_files_test',
        'sw_train',
        'sw_valid',
        'sw_test',
        'labels_train',
        'labels_valid',
        'labels_final_train',
        'labels_final_test',
        'labels_pred',
        'y0_train',
        'y0_valid',
        'y0_final_train',
        'y0_final_test',
        'y0_pred',
        'y0_final_pred',
        'y4_train',
        'y4_valid',
        'y4_final_train',
        'y4_final_test',
        'y4_pred',
        'y4_final_pred',
        'TP',
        'TN',
        'FP',
        'FN',
    ])

def get_config_combination_list(settings, default=None):
    keys = list(settings)
    r = []
    for values in itertools.product(*map(settings.get, keys)):
        d = dict(zip(keys, values))
        if default is not None:
            d.update(default)
        r.append(d)
    return r

def f_sliding_window(window_size, total, step, frame1, frame2):
    """Create 'id' column and fill it with window's identifier.
    Column 'id' is used by tsfresh.
    """
    frame1['id'] = 0
    id1 = list(frame1.columns).index('id')

    frame2['id'] = 0
    id2 = list(frame2.columns).index('id')

    for i in range(total):
        a = i * step
        b = a + window_size
        frame1.iloc[a:b, id1] = i + 1
        frame2.iloc[a:b, id2] = i + 1

    return frame1, frame2

def f_extract_features(*args, **kwargs):
    """
    Use TSFRESH to extract statistical features of input dataframes.
    TSFRESH uses 'joblib' to extract from multivariate in parallel.
    """
    if kwargs.get('printfn', None) is None:
        printfn = print
    else:
        printfn = kwargs.get('printfn')
    r = []
    for a in args:
        if a.shape[0] == 0:
            printfn('fold: ', kwargs.get('foldout', '?'))
            printfn(a.shape)
            printfn(a.describe)
            raise Exception('invalid shape')
        if 'timestamp' not in a.columns:
            printfn(a.columns)
            printfn(a.shape)
            printfn(a.sample(5))
            raise Exception('timestamp column missing')
        if 'id' not in a.columns:
            printfn(a.columns)
            printfn(a.shape)
            printfn(a.sample(5))
            raise Exception('id column missing')

        tmp = extract_features(
            a,
            column_id='id',
            column_sort='timestamp',
            default_fc_parameters=kwargs.get('params'),
            impute_function=impute,
            n_jobs=kwargs.get('n_jobs', 1),
            disable_progressbar=True
        )
        tmp.drop(
            columns=['id', 'timestamp'],
            errors='ignore',
            inplace=True
        )
        r.append(
            tmp.to_numpy().astype('float32')
        )
    return tuple(r)

def get_classifiers(n_jobs=1):
    """
    Experiment classifiers and combinations of hyperparameters values.
    """
    return {
        'KNN': {
            'config': get_config_combination_list(
                {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['ball_tree'],
                    'leaf_size': [5, 25, 50],
                },
                {'n_jobs': n_jobs}
            ),
            'default': None,
            'model': KNeighborsClassifier,
        },
        'DT': {
            'config': get_config_combination_list(
                {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 5, 10, 50],
                    'min_samples_split': [2, 5, 10]
                },
                {'random_state': None}
            ),
            'default': None,
            'model': DecisionTreeClassifier,
        },
        'ADA': {
            'config': get_config_combination_list(
                {
                    'n_estimators': [5, 25, 50, 75, 100],
                    'algorithm': ['SAMME', 'SAMME.R'],
                },
                {'random_state': None}
            ),
            'default': None,
            'model': AdaBoostClassifier,
        },
        'RF': {
            'config': get_config_combination_list(
                {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10, 50],
                    'min_samples_split': [2, 5, 10]
                },
                {'n_jobs': n_jobs, 'n_estimators': 50, 'random_state': None}
            ),
            'default': None,
            'model': RandomForestClassifier,
        },
        'GBOOST': {
            'config': get_config_combination_list(
                {
                    'loss': ['deviance', 'exponential'],
                    'n_estimators': [50, 100, 200],
                    'min_samples_split': [2, 5, 10],
                    'max_depth': [None, 5, 10, 50],
                },
                {'random_state': None}
            ),
            'default': None,
            'model': GradientBoostingClassifier,
        },
    }

def load_instance(instance_path, columns):
    """
    Read CSV files and return Pandas DataFrame.
    """
    try:
        df = pd.read_csv(
            instance_path,
            sep=',',
            header=0,
            names=columns,
        )
        return df
    except Exception as e:
        raise Exception(
            f'error reading file {instance_path}: {e}'
        )

def fill_cache(file_list,
               file_cache,
               columns,
               window_size,
               well_stats,
               fclass,
               flimit=1000000):
    """
    Call CSV reader and process each dataframe into a dict with:
    - the dataframe itself;
    - number of windows
    - compute total windows, start and finish

    File cache is filled in-memory.
    Return stats.
    """
    l0 = min(flimit, len(file_list))
    with tqdm(total=l0) as pbar:
        for fi, f in enumerate(file_list[:l0]):
            file_cache[fi] = {}
            file_cache[fi]['name'] = f
            file_cache[fi]['file'] = load_instance(f, columns)
            file_cache[fi]['frame'] = file_cache[fi]['file'].copy()
            file_cache[fi]['frame'].drop(
                columns=['P-JUS-CKGL', 'T-JUS-CKGL', 'QGL', 'class'],
                axis=1,
                inplace=True,
            )

            file_cache[fi]['frame']['P-MON-CKP'].fillna(
                value=0.0, axis=0, inplace=True
            )
            file_cache[fi]['frame']['T-JUS-CKP'].fillna(
                value=0.0, axis=0, inplace=True
            )

            file_cache[fi]['before_drop'] = file_cache[fi]['frame'].shape[0]

            file_cache[fi]['frame'].dropna(
                axis=0, how='any', inplace=True
            )
            file_cache[fi]['after_drop'] = file_cache[fi]['frame'].shape[0]
            file_cache[fi]['dropped_rows'] = (
                file_cache[fi]['before_drop'] - file_cache[fi]['after_drop']
            )

            file_cache[fi]['n_windows'] = np.int(
                file_cache[fi]['frame'].shape[0] / window_size
            )

            if file_cache[fi]['n_windows'] == 0:
                raise Exception('invalid file or pre-processing')

            well_stats = well_stats.append(
                {
                    'well': f[-24:-19],
                    'class': fclass,
                    'frame': file_cache[fi]['frame'].shape[0],
                    'samples': file_cache[fi]['file'].shape[0],
                    'start': file_cache[fi]['file']['timestamp'].iloc[0],
                    'finish': file_cache[fi]['file']['timestamp'].iloc[-1],
                    'fcount': 1,
                },
                ignore_index=True,
                sort=False
            )
            file_cache[fi]['file'] = None

            pbar.update(1)

    return well_stats

def execute_fold(
        round_,
        foldout,
        ts,
        fclass,
        rounds,
        k_folds,
        file_cache_0,
        file_cache_4,
        columns,
        window_size,
        step,
        norm_before,
        norm_after,
        path0,
        path4,
        nottest_0,
        nottest_4,
        test_files_0,
        test_files_4,
        n_test_files):

    njobs = 1
    results = get_results_frame()
    rcolumns = list(results.columns)
    classifiers = get_classifiers(n_jobs=njobs)
    innerfold = list(range(k_folds - 1))

    df_fc_min = MinimalFCParameters()
    df_fc_min.pop('sum_values')
    df_fc_min.pop('length')

    good_vars = [
        'P-PDG',
        'P-TPT',
        'T-TPT',
        'P-MON-CKP',
        'T-JUS-CKP',
    ]

    min_win_n = 100000000
    for fi in file_cache_0.keys():
        min_win_n = min(min_win_n, file_cache_0[fi]['n_windows'])
    for fi in file_cache_4.keys():
        min_win_n = min(min_win_n, file_cache_4[fi]['n_windows'])

    if min_win_n == 0:
        raise Exception('invalid minimum number of windows')

    # ==============================
    # Outter fold TRAIN
    # ==============================
    tmpminwin0 = [
        file_cache_0[fi]['n_windows'] for fi in nottest_0[foldout]
    ]

    tmpminwin4 = [
        file_cache_4[fi]['n_windows'] for fi in nottest_4[foldout]
    ]

    testminwin0 = [
        file_cache_0[f]['n_windows'] for f in test_files_0[foldout]
    ]
    testminwin4 = [
        file_cache_4[f]['n_windows'] for f in test_files_4[foldout]
    ]

    tmpminwin = np.amin(
        [np.sum(tmpminwin0), np.sum(tmpminwin4)]
    )
    max_idx = np.int(np.amin(tmpminwin0 + tmpminwin4) * window_size)

    nottest_frame_0 = pd.DataFrame()
    nottest_frame_4 = pd.DataFrame()

    nt0l = [file_cache_0[f]['frame'] for f in nottest_0[foldout]]
    nottest_frame_0 = pd.concat(nt0l, ignore_index=True, sort=False)
    nt4l = [file_cache_4[f]['frame'] for f in nottest_4[foldout]]
    nottest_frame_4 = pd.concat(nt4l, ignore_index=True, sort=False)

    # ==============================
    # Outter fold TEST
    # ==============================
    test_frame_0 = pd.DataFrame()
    test_frame_4 = pd.DataFrame()

    testminwin = np.amin(
        [np.sum(testminwin0), np.sum(testminwin4)]
    )
    min_win_out = min(
        np.amin(testminwin0 + testminwin4), np.amin(tmpminwin0 + tmpminwin4)
    )
    max_idx = np.int(np.amin(testminwin0 + testminwin4) * window_size)

    tmp = [file_cache_0[f]['frame'] for f in test_files_0[foldout]]
    test_frame_0 = pd.concat(tmp, ignore_index=True, sort=False)
    tmp = [file_cache_4[f]['frame'] for f in test_files_4[foldout]]
    test_frame_4 = pd.concat(tmp, ignore_index=True, sort=False)

    min_win_out = min(
        tmpminwin,
        testminwin
    )

    nottest_frame_0, nottest_frame_4 = f_sliding_window(
        window_size,
        tmpminwin,
        window_size,
        nottest_frame_0,
        nottest_frame_4,
    )

    test_frame_0, test_frame_4 = f_sliding_window(
        window_size,
        testminwin,
        window_size,
        test_frame_0,
        test_frame_4,
    )

    scaler_before_out = StandardScaler()
    if norm_before:
        scaler_before_out.fit(
            pd.concat(
                [nottest_frame_0, nottest_frame_4], axis=0, ignore_index=True
            )[good_vars]
        )
        nottest_frame_0[good_vars] = scaler_before_out.transform(
            nottest_frame_0[good_vars]
        )
        nottest_frame_4[good_vars] = scaler_before_out.transform(
            nottest_frame_4[good_vars]
        )

        test_frame_0[good_vars] = scaler_before_out.transform(
            test_frame_0[good_vars]
        )
        test_frame_4[good_vars] = scaler_before_out.transform(
            test_frame_4[good_vars]
        )

    nottest_X_0, nottest_X_4 = f_extract_features(
        nottest_frame_0, nottest_frame_4, params=df_fc_min, n_jobs=njobs
    )
    test_X_0, test_X_4 = f_extract_features(
        test_frame_0, test_frame_4, params=df_fc_min, n_jobs=njobs
    )

    scaler_after_out = StandardScaler()
    if norm_after:
        scaler_after_out.fit(
            np.concatenate([nottest_X_0, nottest_X_4], axis=0)
        )

        nottest_X_0 = scaler_after_out.transform(nottest_X_0)
        nottest_X_4 = scaler_after_out.transform(nottest_X_4)

        test_X_0 = scaler_after_out.transform(test_X_0)
        test_X_4 = scaler_after_out.transform(test_X_4)

    X_outter_train = np.concatenate(
        [nottest_X_0, nottest_X_4], axis=0
    )
    y_outter_train = np.concatenate([
        np.zeros([nottest_X_0.shape[0]], dtype='int'),
        np.ones( [nottest_X_4.shape[0]], dtype='int')
    ])

    X_outter_test = np.concatenate(
        [test_X_0, test_X_4], axis=0
    )
    y_outter_test = np.concatenate([
        np.zeros([test_X_0.shape[0]], dtype='int'),
        np.ones( [test_X_4.shape[0]], dtype='int')
    ])

    # =========================================================================
    # Inner fold - training and validation sets
    # =========================================================================

    validchoice_0 = np.random.RandomState(round_).choice(
        nottest_0[foldout],
        size=[k_folds - 1, n_test_files],
        replace=False
    )
    validchoice_4 = np.random.RandomState(round_).choice(
        nottest_4[foldout],
        size=[k_folds - 1, n_test_files],
        replace=False
    )

    trainchoice_0 = np.zeros(
        [k_folds - 1, n_test_files * (k_folds - 2)], dtype='int'
    )
    trainchoice_4 = np.zeros(
        [k_folds - 1, n_test_files * (k_folds - 2)], dtype='int'
    )

    for fin in innerfold:
        finvf0 = validchoice_0[fin]
        trainchoice_0[fin] = [n for n in nottest_0[foldout] if n not in finvf0]

        finvf4 = validchoice_4[fin]
        trainchoice_4[fin] = [n for n in nottest_4[foldout] if n not in finvf4]

    inner_result_list = []

    for foldin in innerfold:
        tmpminwin0 = [
            file_cache_0[fi]['n_windows'] for fi in trainchoice_0[foldin]
        ]
        tmpminwin4 = [
            file_cache_4[fi]['n_windows'] for fi in trainchoice_4[foldin]
        ]

        tmp = [file_cache_0[f]['frame'] for f in trainchoice_0[foldin]]
        train_frame_0 = pd.concat(tmp, ignore_index=True, sort=False, copy=True)
        tmp = [file_cache_4[f]['frame'] for f in trainchoice_4[foldin]]
        train_frame_4 = pd.concat(tmp, ignore_index=True, sort=False, copy=True)

        tmpminwin = np.amin(
            [np.sum(tmpminwin0), np.sum(tmpminwin4)]
        )

        valid_frame_0 = pd.DataFrame()
        valid_frame_4 = pd.DataFrame()

        testminwin4 = [
            file_cache_4[f]['n_windows'] for f in validchoice_4[foldin]
        ]
        testminwin4 = [
            file_cache_4[f]['n_windows'] for f in validchoice_4[foldin]
        ]

        testminwin = np.amin(
            [np.sum(testminwin0), np.sum(testminwin4)]
        )

        tmp = [file_cache_0[f]['frame'] for f in validchoice_0[foldin]]
        valid_frame_0 = pd.concat(tmp, ignore_index=True, sort=False)
        tmp = [file_cache_4[f]['frame'] for f in validchoice_4[foldin]]
        valid_frame_4 = pd.concat(tmp, ignore_index=True, sort=False)

        min_win_in = min(tmpminwin, testminwin)

        train_frame_0, train_frame_4 = f_sliding_window(
            window_size,
            tmpminwin,
            window_size,
            train_frame_0,
            train_frame_4,
        )

        valid_frame_0, valid_frame_4 = f_sliding_window(
            window_size,
            testminwin,
            window_size,
            valid_frame_0,
            valid_frame_4,
        )

        scaler_before_in = StandardScaler()
        if norm_before:
            scaler_before_in.fit(
                pd.concat(
                    [train_frame_0, train_frame_4], axis=0, ignore_index=True
                )[good_vars]
            )
            train_frame_0[good_vars] = scaler_before_in.transform(
                train_frame_0[good_vars]
            )
            train_frame_4[good_vars] = scaler_before_in.transform(
                train_frame_4[good_vars]
            )

            valid_frame_0[good_vars] = scaler_before_in.transform(
                valid_frame_0[good_vars]
            )
            valid_frame_4[good_vars] = scaler_before_in.transform(
                valid_frame_4[good_vars]
            )

        X_inner_train_0, X_inner_train_4 = f_extract_features(
            train_frame_0, train_frame_4, params=df_fc_min, n_jobs=njobs
        )

        X_valid_0, X_valid_4 = f_extract_features(
            valid_frame_0, valid_frame_4, params=df_fc_min, n_jobs=njobs
        )

        if norm_after:
            scaler_after_in = StandardScaler()
            scaler_after_in.fit(
                np.concatenate(
                    [X_inner_train_0, X_inner_train_4], axis=0
                )
            )

            X_inner_train_0 = scaler_after_in.transform(X_inner_train_0)
            X_inner_train_4 = scaler_after_in.transform(X_inner_train_4)

            X_valid_0 = scaler_after_in.transform(X_valid_0)
            X_valid_4 = scaler_after_in.transform(X_valid_4)

        X_inner_train = np.concatenate(
            [X_inner_train_0, X_inner_train_4], axis=0
        )
        y_inner_train = np.concatenate([
            np.zeros([X_inner_train_0.shape[0]], dtype='int'),
            np.ones( [X_inner_train_4.shape[0]], dtype='int')
        ])

        X_inner_valid = np.concatenate(
            [X_valid_0, X_valid_4], axis=0
        )
        y_inner_valid = np.concatenate([
            np.zeros([X_valid_0.shape[0]], dtype='int'),
            np.ones( [X_valid_4.shape[0]], dtype='int')
        ])

        for clf in classifiers:
            for cc, clfcfg in enumerate(classifiers[clf]['config']):
                if 'random_state' in clfcfg:
                    clfcfg['random_state'] = round_

                classif = classifiers[clf]['model'](**clfcfg)

                try:
                    classif.fit(X_inner_train, y_inner_train)
                except Exception as efit:
                    print(efit)

                try:
                    y_inner_pred = classif.predict(X_inner_valid)
                except Exception as epred:
                    print(epred)

                try:
                    p, r, f1new, _ = precision_recall_fscore_support(
                        y_inner_valid,
                        y_inner_pred,
                        average='binary',
                    )
                except UndefinedMetricWarning as emet:
                    print(emet)
                    p = 0.0
                    r = 0.0
                    f1new = 0.0

                try:
                    acc = accuracy_score(y_inner_valid, y_inner_pred)
                except:
                    acc = None

                rd = {
                    'seed': round_,
                    'fold_outter': foldout,
                    'fold_inner': foldin,
                    'classifier': clf,
                    'clf_cfg': cc,
                    'F1': f1new,
                    'precision': p,
                    'recall': r,
                    'accuracy': acc,
                    'time': 0,
                    '0_files_train': str(trainchoice_0[foldin]),
                    '0_files_valid': str(validchoice_0[foldin]),
                    '0_files_test': None,
                    '4_files_train': str(trainchoice_4[foldin]),
                    '4_files_valid': str(validchoice_4[foldin]),
                    '4_files_test': None,
                    'sw_train': min_win_n * n_test_files * (k_folds - 2),
                    'sw_train_final': min_win_n * n_test_files * (k_folds - 1),
                    'sw_valid': min_win_n * n_test_files * 1,
                    'sw_test': min_win_n * n_test_files * 1,
                    'y0_train': X_inner_train_0.shape[0],
                    'y4_train': X_inner_train_4.shape[0],
                    'y0_valid': X_valid_0.shape[0],
                    'y4_valid': X_valid_4.shape[0],
                    'y4_pred': np.sum(y_inner_pred),
                    'y0_pred': X_inner_valid.shape[0] - np.sum(y_inner_pred),
                    'y0_final_train': 0,
                    'y4_final_train': 0,
                    'y0_final_test': 0,
                    'y4_final_test': 0,
                }

                inner_result_list.append(rd)

    results = pd.DataFrame(data=inner_result_list, columns=rcolumns)

    tmp = results[results['fold_outter'] == foldout]
    tmp = tmp[tmp['fold_inner'] > -1]  # remove partials
    pvt1 = pd.pivot_table(
        data=tmp,
        values=['F1'],
        aggfunc={
            'F1': ['mean', ],
        },
        index=['classifier', 'clf_cfg'],
    )
    pvt1 = pvt1.reindex(
        pvt1['F1'].sort_values(
            by=['mean', 'clf_cfg'],
            ascending=[False, True]
        ).index
    )

    for clf in classifiers:
        idx = list(pvt1.loc[clf].idxmax())[0]
        best_config = classifiers[clf]['config'][idx]
        if 'random_state' in best_config:
            best_config['random_state'] = round_

        classif = classifiers[clf]['model'](**best_config)

        try:
            classif.fit(
                X_outter_train,
                y_outter_train
            )
        except Exception as efit:
            print(efit)

        try:
            y_pred = classif.predict(X_outter_test)
        except Exception as epred:
            print(epred)

        try:
            p, r, f1final, _ = precision_recall_fscore_support(
                y_outter_test,
                y_pred,
                average='binary',
            )
        except UndefinedMetricWarning as e:
            print(e)
            p = 0.0
            r = 0.0
            f1final = 0.0

        try:
            acc = accuracy_score(
                y_outter_test,
                y_pred,
                normalize=True
            )
        except:
            acc = None

        try:
            cm = confusion_matrix(
                y_outter_test,
                y_pred,
                labels=[0, 1],
                sample_weight=None,
                normalize=None
            )
        except:
            cm = np.zeros([2, 2], dtype=int)

        results = results.append({
            'ts': np.int(f'{ts:%Y%m%d%H%M%S}'),
            'class': fclass,
            'seed': round_,
            'fold_outter': foldout,
            'fold_inner': -100,
            'classifier': clf,
            'clf_cfg': idx,
            'F1': f1final,
            'precision': p,
            'recall': r,
            'accuracy': acc,
            'time': 0,
            'window_size': window_size,
            'step': step,
            '0_n_files_train': n_test_files * (k_folds - 1),
            '0_n_files_valid': None,
            '0_n_files_test': n_test_files,
            '4_n_files_train': n_test_files * (k_folds - 1),
            '4_n_files_valid': None,
            '4_n_files_test': n_test_files,
            'sw_train': min_win_n * n_test_files * (k_folds - 2),
            'sw_valid': min_win_n * n_test_files,
            'sw_train_final': min_win_n * n_test_files * (k_folds - 1),
            'sw_test': min_win_n * n_test_files,
            'y0_train': 0,
            'y4_train': 0,
            'y0_valid': 0,
            'y4_valid': 0,
            'y0_final_train': nottest_X_0.shape[0],
            'y4_final_train': nottest_X_4.shape[0],
            'y0_final_test': test_X_0.shape[0],
            'y4_final_test': test_X_4.shape[0],
            'y4_final_pred': np.sum(y_pred),
            'y0_final_pred': y_pred.shape[0] - np.sum(y_pred),
            'TP': cm[0, 0],
            'TN': cm[1, 1],
            'FP': cm[1, 0],
            'FN': cm[0, 1],
        }, ignore_index=True, sort=False)

    return results

def main():

    fclass = 4
    round0 = 1
    rounds = 1
    folds = 10
    normalizebeforefe = False
    normalizeafterfe = True
    window_size = 900
    step = 900
    overlap = 0
    ntestfiles = 1
    TZ = tz.UTC

    events_names = {
        0: 'Normal',
        1: 'Abrupt Increase of BSW',
        2: 'Spurious Closure of DHSV',
        3: 'Severe Slugging',
        4: 'Flow Instability',
        5: 'Rapid Productivity Loss',
        6: 'Quick Restriction in PCK',
        7: 'Scaling in PCK',
        8: 'Hydrate in Production Line'
    }

    ts = datetime.now(tz=TZ)
    tstart = time.time()

    print('Running ', ts)
    print('Event', events_names[fclass])

    well_vars = [
        'P-PDG',
        'P-TPT',
        'T-TPT',
        'P-MON-CKP',
        'T-JUS-CKP',
        'P-JUS-CKGL',
        'T-JUS-CKGL',
        'QGL',
    ]

    df_fc_min = MinimalFCParameters()
    df_fc_min.pop('sum_values')
    df_fc_min.pop('length')
    columns = ['timestamp'] + well_vars + ['class']
    path0 = os.path.join(
        os.path.dirname(__file__), '3w_dataset', 'data', '0'
    )
    path4 = os.path.join(
        os.path.dirname(__file__), '3w_dataset', 'data', '4'
    )
    f_pattern = 'WELL-[0-9]*_[0-9]*.csv'
    well_list_0 = sorted(glob.glob(path0 + os.sep + f_pattern))
    well_list_4 = sorted(glob.glob(path4 + os.sep + f_pattern))

    mintot = min(len(well_list_0), len(well_list_4))
    if ntestfiles * folds > mintot:
        print('Check relative paths:')
        raise Exception(
            f'invalid combination: n_test ({ntestfiles}) * folds ({folds}) '
            f'> min_total_files ({mintot})'
        )

    k_folds = folds
    n_test_files = ntestfiles
    well_macro_stats = pd.DataFrame(
        columns=['well', 'class', 'samples', 'frame', 'nan']
    )
    file_cache_0 = {}
    file_cache_4 = {}
    well_macro_stats = fill_cache(
        well_list_0,
        file_cache_0,
        columns,
        window_size,
        well_macro_stats,
        0,
    )
    well_macro_stats = fill_cache(
        well_list_4,
        file_cache_4,
        columns,
        window_size,
        well_macro_stats,
        fclass,
    )

    pvtstat = pd.pivot_table(
        data=well_macro_stats,
        index=[
            'well',
            'class'
        ],
        values=[
            'fcount',
            'samples',
            'frame',
            'start',
            'finish',
        ],
        aggfunc={
            'samples': ['sum'],
            'frame': 'sum',
            'fcount': 'count',
            'start': 'min',
            'finish': 'max',
        },
    )

    rkf = range(k_folds)
    results = get_results_frame()
    warnings.filterwarnings("error")
    outterfold = list(rkf)
    range_0 = list(file_cache_0.keys())
    range_4 = list(file_cache_4.keys())
    total_loops = rounds * k_folds

    with tqdm(total=total_loops) as pbar:
        # ======================================================================
        # SEED ROUND
        # ======================================================================
        for round_ in range(round0, rounds + round0):

            test_files_0 = np.random.RandomState(round_).choice(
                range_0, size=[k_folds, ntestfiles], replace=False
            )
            test_files_4 = np.random.RandomState(round_).choice(
                range_4, size=[k_folds, ntestfiles], replace=False
            )

            nottest_0 = np.zeros([k_folds, ntestfiles * (k_folds - 1)], dtype='int')
            nottest_4 = np.zeros([k_folds, ntestfiles * (k_folds - 1)], dtype='int')
            for f in outterfold:
                tmp_0 = [i for i in range_0 if i not in test_files_0[f]]
                nottest_0[f] = np.random.RandomState(round_).choice(
                    tmp_0, size=[ntestfiles * (k_folds - 1)], replace=False
                )

                tmp_4 = [i for i in range_4 if i not in test_files_4[f]]
                nottest_4[f] = np.random.RandomState(round_).choice(
                    tmp_4, size=[ntestfiles * (k_folds - 1)], replace=False
                )

            # ==================================================================
            # Outter fold
            # ==================================================================
            results_list = [None for f in rkf]
            for foldout in range(k_folds):
                results_list[foldout] = execute_fold(
                    round_,
                    foldout,
                    ts,
                    fclass,
                    rounds,
                    k_folds,
                    file_cache_0,
                    file_cache_4,
                    columns,
                    window_size,
                    step,
                    normalizebeforefe,
                    normalizeafterfe,
                    path0,
                    path4,
                    nottest_0,
                    nottest_4,
                    test_files_0,
                    test_files_4,
                    ntestfiles,
                )

                pbar.update(1)

            results_list.append(results)
            results = pd.concat(
                results_list, axis=0, ignore_index=True
            )

    try:
        results.to_csv('results.zip')  # with compression
    except Exception as ecsv:
        print('Exception while saving final CSV:')
        print(ecsv)


    pvt2 = pd.pivot_table(
        data=results[results['fold_inner'] == -100],
        values=[
            'class',
            'seed',
            'F1',
            'window_size',
            'accuracy'
        ],
        aggfunc={
            'seed': 'max',
            'F1': ['mean'],
            'accuracy': 'mean',
            'window_size': ['max'],
        },
        index=['classifier'],
    )

    pvt3 = pd.pivot_table(
        data=results[results['fold_inner'] == -100],
        values=[
            'F1',
            'window_size',
        ],
        aggfunc={
            'F1': ['mean'],
            'window_size': ['max'],
        },
        index=['classifier'],
    )

    print(pvt2)
    print(pvt3)

if __name__ == "__main__":
    main()
