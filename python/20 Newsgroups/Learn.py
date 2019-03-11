"""
Script to Simulate and Train the MultISAR model on the 20 Newsgroup Dataset
"""
# Load Python Specific Libraries
from pandas.api.types import CategoricalDtype as CDType
import pandas as pd
import numpy as np
import argparse
import sys

# Load SKLearn Stuff
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

# Load own packages
sys.path.append('..')
from Models import MultISAR

# Default Parameters
DEFAULTS = \
    {'Source': '20NG_BOW.npz',   # Source of the Data
     'Output': 'Results_02',    # Result file
     'Random': '0',          # Random Seed offset
     'Numbers':  ['0', '30'],  # Range: start index, number of runs
     'FullSpec': '0.02',     # Proportion of Fully-Specified Labels
     'Alpha': '0.005',       # Laplace Smoothing Parameter for counts
     'Steps': ['0.01', '0.02', '0.04', '0.06', '0.08', '0.1', '0.15', '0.2', '0.3', '0.4', '0.5', '0.6', '0.8', '1.0']}


if __name__ == '__main__':

    # ==== Parse Arguments: ==== #
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
    _arg_parse.add_argument('-i', '--increments', help='Steps at which to add the remaining data: should range from >0 '
                                                       'to 1.0', nargs='*', default=DEFAULTS['Steps'])
    # Parse Arguments and transform as required
    args = _arg_parse.parse_args()
    args.random = int(args.random)
    run_offset = int(args.numbers[0])
    run_length = int(args.numbers[1])
    args.fullspec = float(args.fullspec)
    args.alpha = float(args.alpha)
    args.increments = np.asarray([float(s) for s in args.increments])

    # ===== Load/Prepare the Data: ===== #
    print('Loading and Parsing the Data...')
    _data = np.load(args.source)
    # ---- Extract the different feature-matrices ---- #
    # Raw Data-Sets
    X_train = _data['X_train']
    y_train = _data['y_train']
    g_train = _data['g_train']
    X_test = _data['X_test']
    y_test = _data['y_test']
    # Label System
    labels_full = _data['full_labelset']
    labels_group = _data['group_labels']
    labels_map = _data['mapper']
    # Keep track of sizes
    sZ = len(labels_full)
    sS = len(labels_group)

    # ---- Construct Omega: ---- #
    #    Omega is the Deterministic [0/1] Schema mapping. We do this based on domain knowledge of the groups
    omega = np.zeros([sZ, sS, sZ + sS])
    # Create forward Mapping (i.e. from group to labels)
    mapper = np.zeros([sS, sZ])
    for _i in range(sS):
        mapper[_i, np.squeeze(np.where(labels_map == _i))] = 1
    # Populate omega
    for z in range(sZ):
        for s in range(sS):
            for y in range(sZ+sS):
                if mapper[s, z] == 1:  # If in schema (Expertise)
                    if y == z:
                        omega[z, s, y] = 1.0
                elif (y >= sZ) and (mapper[y - sZ, z] == 1):  # Not In this Schema (Expertise)
                    omega[z, s, y] = 1.0

    # ===== Simulate and Simultaneously Train Models: ===== #
    f1_metric = np.empty([run_length, len(args.increments) + 1])   # Array for Performance Metric
    aug_sizes = np.empty([run_length, len(args.increments) + 1])   # Array to keep track of lengths (due to randomness)
    # ---- Run RUN_LENGTH independent trials ----
    for run in range(run_offset, run_offset+run_length):
        print('Executing Simulation Run: {} ... \n  - Splitting Data...'.format(run))

        # Seed with Random Offset + run value
        np.random.seed(args.random + run)

        # [A] Split first into two sets, the first for EXact data, the other for CoaRSe labelling only. To do this, we
        #   use the train_test_split method of SKLearn. We use stratification to ensure balanced classes.
        X_ex, X_crs, y_ex, y_crs, g_ex, g_crs = train_test_split(X_train, y_train, g_train, train_size=args.fullspec,
                                                                 stratify=y_train, random_state=args.random + run)

        # [B] Train using the Baseline: Multinomial Class conditionals on just the fully-specified data. This is done
        #   using SKLearn's MultinomialNB Classifier. We then evaluate the performance on the entirely-held out test
        #   set.
        print('  - Training MultinomialNB...')
        mnb_clf = MultinomialNB(alpha=args.alpha).fit(X_ex, y_ex)
        _y_pred = mnb_clf.predict(X_test)
        f1_metric[run-run_offset, 0] = f1_score(y_test, _y_pred, labels=np.arange(sZ), average='weighted')
        aug_sizes[run-run_offset, 0] = len(y_ex)

        # [C] Now train and evaluate the performance when augmenting the data-set with coarsely-labelled data.
        #   To do this, we first compute the starting point for EM as the model parameters of the MNB Model
        sys.stdout.write('  - Training Augmented ISAR Model:')
        _starts = [(np.exp(mnb_clf.class_log_prior_), np.exp(mnb_clf.feature_log_prob_))]
        for _i, _a in enumerate(args.increments):
            sys.stdout.write(' {}'.format(_a)); sys.stdout.flush()
            # Find the Number of samples to augment with (from the coarse set)
            _top_idx = np.ceil(len(X_crs) * _a).astype(int)
            aug_sizes[run - run_offset, _i + 1] = _top_idx + len(y_ex)
            # Setup X-Data
            _X_train = np.concatenate((X_ex, X_crs[:_top_idx, :]), axis=0)
            # Setup Training Data (y-targets with some of the labels coarsely specified). Due to legacy reasons during
            #   development of the code, the labels must be one-hot encoded (we use Pandas to achieve this)
            _y_train = np.concatenate((y_ex, g_crs[:_top_idx] + sZ), axis=0)
            _y_train = pd.Series(_y_train, dtype=CDType(categories=np.arange(sZ + sS)))
            _y_train = pd.get_dummies(_y_train).values
            # Setup Schema Indicators: this is basically the group value. Again, this is one-hot encoded
            _s_train = np.concatenate((g_ex, (g_crs[:_top_idx]+1)%sS), axis=0)
            _s_train = pd.Series(_s_train, dtype=CDType(categories=np.arange(sS)))
            _s_train = pd.get_dummies(_s_train).values
            # Train Model
            misac_clf = MultISAR(omega=omega, _num_proc=-1, _max_iter=100)
            misac_clf.fit_model(_X_train, _y_train, _s_train, args.alpha, _starts)
            # Evaluate performance (on held-out test set)
            _y_pred = misac_clf.predict(X_test)
            f1_metric[run-run_offset, _i+1] = f1_score(y_test, _y_pred, labels=np.arange(sZ), average='weighted')
        print('  Done\n')
    # ===== Finally store the results to File: ===== #
    print('Storing Results to file ... ')
    np.savez_compressed(args.output, score=f1_metric, sizes=aug_sizes)
