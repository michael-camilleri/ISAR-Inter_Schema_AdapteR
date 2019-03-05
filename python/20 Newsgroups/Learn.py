# Load Python Specific Libraries
from pandas.api.types import CategoricalDtype as CDType
import pandas as pd
import numpy as np
import argparse
import sys
import os

# Load SKLearn Stuff
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load own packages
sys.path.append('..')
from Models import MultISAR

# Default Parameters
DEFAULTS = \
    {'Source': '20NG_BOW',   # Source of the Data
     'Output': 'Results',    # Result file
     'Random': '0',          # Random Seed offset
     'Numbers':  ['0', '30'],  # Range: start index, number of runs
     'FullSpec': '0.02',     # Proportion of Fully-Specified Labels
     'Alpha': '0.005',       # Laplace Smoothing Parameter for counts
     'Steps': ['0.01', '0.02', '0.04', '0.06', '0.08', '0.1', '0.15', '0.2', '0.3', '0.4', '0.5', '0.6', '0.8', '1.0']}


if __name__ == '__main__':

    _arg_parse = argparse.ArgumentParser(description='Train a Multinomial Class-Conditional ISAR')
    _arg_parse.add_argument('-s', '--source', help='Source file for Data: default is {}'.format(DEFAULTS['Source']),
                            default=DEFAULTS['Source'])
    _arg_parse.add_argument('-o', '--output', help='Output Result file: default is {}'.format(DEFAULTS['Output']),
                            default=DEFAULTS['Output'])
    _arg_parse.add_argument('-r', '--random', help='Seed (offset) for all Random States: ensures repeatibility. '
                                                   'Defaults to {}'.format(DEFAULTS['Random']),
                            default=DEFAULTS['Random'])
    _arg_parse.add_argument('-n', '--numbers', help='Range of Runs to simulate: tuple containing the start index and '
                                                    'number of runs. Defaults to {}/{}'.format(*DEFAULTS['Numbers']),
                            nargs=2, default=DEFAULTS['Numbers'])
    _arg_parse.add_argument('-f', '--fullspec', help='Proportion of the data to have fully-specified labels. Default '
                                                     'is {}'.format(DEFAULTS['FullSpec']), default=DEFAULTS['FullSpec'])
    _arg_parse.add_argument('-a', '--alpha', help='Laplace Smoothing Parameter for Multinomial Distribution: Defaults '
                                                  'to {}'.format(DEFAULTS['Alpha']), default=DEFAULTS['Alpha'])
    _arg_parse.add_argument('-s', '--steps', help='Steps at which to add the remaining data.', nargs='*',
                            default=DEFAULTS['Steps'])
    _args = _arg_parse.parse_args()



    # Load own packages
    from Tools.Common import skext
    from Tools.Simulators import simulate_annotators


    # Load the Data
    _data = np.load(os.path.join(Const['Data.Clean'], '20NewsGroups_{}.npz'.format(DATA_TYPE)))

    # Prepare Directory
    FILE_NAME = 'MultISAC_SA_{0}_Augment_{1}_[{2:03d}-{3:03d}]_[{4:.3f}]'\
        .format(DATA_TYPE, SIM_NUMBER, RUN_OFFSET, RUN_LENGTH, EXACT_PERCENT)

    # Extract the different feature-matrices
    X_train = _data['X_train']
    y_train = _data['y_train']
    z_train = _data['z_train']
    X_test = _data['X_test']
    y_test = _data['y_test']
    _f_lab = _data['fine_labels']
    _c_lab = _data['coarse_labels']
    _map = _data['mapper']

    # Simulate Annotators
    _groups = [[] for _ in _c_lab]
    for _i, _v in enumerate(_map):
        _groups[_v].append(_i)
    simplistic_anns = simulate_annotators(_groups)

    # Construct Phi
    # Build Mapper (for Phi)
    Phi = np.zeros([len(_f_lab), len(_c_lab), len(_f_lab) + len(_c_lab)])
    mapper = np.zeros([len(_c_lab), len(_f_lab)])
    for _i in range(len(_c_lab)):
        mapper[_i, np.squeeze(np.where(_map == _i))] = 1
    for z in range(len(_f_lab)):
        for s in range(len(_c_lab)):
            for y in range(len(_f_lab) + len(_c_lab)):
                if mapper[s, z] == 1:  # If in schema (Expertise)
                    if y == z:
                        Phi[z, s, y] = 1.0
                elif (y >= len(_f_lab)) and (mapper[y - len(_f_lab), z] == 1):  # Not In this Schema (Expertise)
                    Phi[z, s, y] = 1.0
    map_dict = {k+len(_f_lab):v for k, v in enumerate(mapper)}

    # Iterate over Runs:
    _performance = np.empty([RUN_LENGTH, len(AUG_PERCENT) + 1, 4])      # Create Array for Performance
    _augmented = np.empty([RUN_LENGTH, len(AUG_PERCENT) + 1])         # Create Array to compute how many points
    for run in range(RUN_OFFSET, RUN_OFFSET+RUN_LENGTH):
        np.random.seed(run)

        print('Executing Simulation Run: {} ... \n Simulating Data...'.format(run))
        # [A] Split first into two sets, the first for exact data, the other for coarse labelling only
        X_ex, X_crs, y_ex, y_crs, z_ex, z_crs = train_test_split(X_train, y_train, z_train, train_size=EXACT_PERCENT,
                                                                 stratify=y_train, random_state=run)

        # Do First Metric: Just the MNB:
        mnb_clf = MultinomialNB(alpha=SMOOTH_PARAMS).fit(X_ex, y_ex)
        _performance[run-RUN_OFFSET, 0, :] = skext.evaluate(mnb_clf, X_test, y_test, np.arange(len(_f_lab)), 'MNBD')
        _augmented[run - RUN_OFFSET, 0] = len(y_ex)
        _starts = [(np.exp(mnb_clf.class_log_prior_), np.exp(mnb_clf.feature_log_prob_))]

        # Now do the metrics for augmentations
        for _i, _a in enumerate(AUG_PERCENT):
            # Find the Length of Indices to assign
            _top_idx = np.ceil(len(X_crs) * _a).astype(int)
            _augmented[run - RUN_OFFSET, _i + 1] = _top_idx + len(y_ex)
            # Setup X-Data
            _X_train = np.concatenate((X_ex, X_crs[:_top_idx, :]), axis=0)
            # Setup Y-Data
            _y_train = np.concatenate((y_ex, z_crs[:_top_idx] + len(_f_lab)), axis=0)
            _y_train = pd.Series(_y_train, dtype=CDType(categories=np.arange(len(_f_lab) + len(_c_lab))))
            _y_train = pd.get_dummies(_y_train).values
            # Setup Z-Data
            _z_train = np.concatenate((z_ex, (z_crs[:_top_idx]+1)%len(_c_lab)), axis=0)
            _z_train = pd.Series(_z_train, dtype=CDType(categories=np.arange(len(_c_lab))))
            _z_train = pd.get_dummies(_z_train).values
            # Train Model
            misac_clf = MultISAR(_phi=Phi, _num_proc=-1, _max_iter=100)
            misac_clf.fit_model(_X_train, _y_train, _z_train, SMOOTH_PARAMS, _f_lab, _starts)
            _performance[run-RUN_OFFSET, _i+1, :] = skext.evaluate(misac_clf, X_test, y_test, np.arange(len(_f_lab)), 'MISAC')

    # Store the Results to File
    print('Storing Results to file ... ')
    np.savez_compressed(os.path.join(Const['Results.Scratch'], FILE_NAME),
                        performance=_performance, augmented=_augmented)


