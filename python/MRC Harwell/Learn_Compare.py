"""
Script to compare, in simulation, the performance of the ISAC model to plain old DS, under realistic (simulation)
conditions.

The Simulation process does not generate any Missing-At-Random data, and hence, in this case, all Samples where the
assigned annotator does not provide a label, are interpreted as NIS.
"""

from pandas.api.types import CategoricalDtype as CDType
from itertools import cycle
import pandas as pd
import numpy as np
import argparse
import sys

# Load own packages
sys.path.append('..')
from Tools import npext
from Models import AnnotDS, AnnotISAR

# Default Parameters
DEFAULTS = \
    {'Output': ['../../data/Compare_DSS', '../../data/Compare_DSA', '../../data/Compare_ISAR'],
     'Random': '0',                                               # Random Seed offset
     'Numbers': ['0', '20'],                                      # Range: start index, number of runs
     'Lengths': ['60', '5400'],                                   # Number and length of segments
     'Schemas': ['13', '15', '17', '10'],                         # Probability over Schemas,
     'Folds': '10'}                                               # Number of Folds to use (folding is by Segment)

# Some Constants
PDF_ANNOT = npext.sum_to_one([49, 31, 11, 4, 10, 25, 9, 7, 6, 3, 3])    # Probability over Annotators
sK = len(PDF_ANNOT)
sS = 4
# CD-Type per schema
CDTYPES = (CDType(categories=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),           # I
           CDType(categories=[0, 1, 5, 6, 7, 10, 12]),                           # II
           CDType(categories=[0, 1, 4, 5, 6, 7, 8, 9, 10, 11]),                  # III
           CDType(categories=[0, 1, 2, 5, 6, 7, 10, 11]),                        # IV
           CDType(categories=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))        # Full Label-Set (for DS_ALL)

nA = 3              # Number of Annotators in any one sample
DSS, DSA, ISAR = (0, 1, 2)   # Position (index) into arrays

if __name__ == '__main__':

    # ==== Parse Arguments ==== #
    _arg_parse = argparse.ArgumentParser(description='Simulate and Train Annotator Models (based on DS and ISAR)')
    _arg_parse.add_argument('-o', '--output', help='Output Result files, one each for the results from the DS trained '
                                                   'per schema, DS trained holistically and ISAR models. Put "None" '
                                                   'for any that you do not want to simulate. Defaults to {}'
                            .format(DEFAULTS['Output']), default=DEFAULTS['Output'], nargs=3)
    _arg_parse.add_argument('-r', '--random', help='Seed (offset) for all Random States: ensures repeatibility. '
                                                   'Defaults to {}'.format(DEFAULTS['Random']),
                            default=DEFAULTS['Random'])
    _arg_parse.add_argument('-n', '--numbers', help='Range of Runs to simulate: tuple containing the start index and '
                                                    'number of runs. Defaults to {}/{}'.format(*DEFAULTS['Numbers']),
                            nargs=2, default=DEFAULTS['Numbers'])
    _arg_parse.add_argument('-l', '--lengths', help='Number and length of each segment. Default {}'
                            .format(DEFAULTS['Lengths']), default=DEFAULTS['Lengths'], nargs=2)
    _arg_parse.add_argument('-f', '--folds', help='Number of splits (folds) to operate on per run. Default {}'
                            .format(DEFAULTS['Folds']), default=DEFAULTS['Folds'])
    _arg_parse.add_argument('-s', '--schemas', help='Schema Probabilities. Default is: {}'.format(DEFAULTS['Schemas']),
                            default=DEFAULTS['Schemas'], nargs=sS)
    _arg_parse.add_argument('-a', '--all', help='If Set, then store all the data (i.e. including the simulation): note '
                                                'that this will generate large files. Default is off.', dest='save_all',
                            action='store_true')
    _arg_parse.set_defaults(save_all=False)
    # ---- Parse Arguments and transform as required ---- #
    args = _arg_parse.parse_args()
    args.random = int(args.random)
    run_offset = int(args.numbers[0])
    run_length = int(args.numbers[1])
    pdf_schema = npext.sum_to_one([float(f) for f in args.schemas])  # Probability over Schemas
    sN, sT = [int(l) for l in args.lengths]
    sF = int(args.folds)
    if args.save_all:
        print('Saving everything is not currently supported and will be ignored.')

    # ==== Load/Prepare the Data ==== #
    # ---- Load Baseline Models ---- #
    with np.load('../../data/model.mrc.npz') as _data:
        omega = _data['omega']
        pi = _data['pi']
        psi = _data['psi']
    # ---- Extract Sizes ---- #
    sZ, sK, sU = psi.shape       # Latent Model [Z, #Annotators, U]
    sS, _, sY = omega.shape      # Number of Schemas and Observeable Labels
    NIS = sZ                     # NIS Label
    iterator = cycle(range(sF))  # Fold Iterator

    # ==== Simulate (and Learn) Model(s) ==== #
    # ---- Prepare Placeholders ---- #
    # Note that we use the same array for both: DS appears in the first, and ISAR in the second position
    if args.output[DSS].lower() != 'none':
        pred_corr_DS_S = np.zeros([run_length, sS], dtype=float)      # Array for Absolute Accuracy
        pred_wght_DS_S = np.zeros([run_length, sS], dtype=float)      # Array for Predictive Log Probability
    else:
        pred_corr_DS_S = None; pred_wght_DS_S = None
    if args.output[DSA].lower() != 'none':
        pred_corr_DS_A = np.zeros([run_length, sS], dtype=float)
        pred_wght_DS_A = np.zeros([run_length, sS], dtype=float)
    else:
        pred_corr_DS_A = None; pred_wght_DS_A = None
    if args.output[ISAR].lower() != 'none':
        pred_corr_ISAR = np.zeros([run_length, sS], dtype=float)
        pred_wght_ISAR = np.zeros([run_length, sS], dtype=float)
    else:
        pred_corr_ISAR = None; pred_wght_ISAR = None
    run_sizes = np.zeros([run_length, sS], dtype=float)         # Number of Samples per Schema per Run

    # --- Iterate over Runs ---- #
    for run in range(run_offset, run_offset + run_length):
        print('Executing Simulation Run: {} ...'.format(run))

        # Seed with Random Offset + run value
        np.random.seed(args.random + run)

        # [A] - Generate Data
        print(' - Generating Data:')
        Z = np.random.choice(sZ, size=sN*sT, p=pi)                                          # Latent State
        S = np.repeat(np.random.choice(sS, size=sN, p=pdf_schema), sT)                      # Schema
        F = np.repeat([next(iterator) for _ in range(sN)], sT)                              # Folds: sequentially
        # With regards to the observations, have to do on a sample-by-sample basis.
        Y = np.full([sN * sT, sK], fill_value=np.NaN)   # Observations Matrix
        A = np.empty([sN * sT, nA], dtype=int)          # Annotator Selection Matrix
        for n in range(sN):  # Iterate over all segments
            A[n*sT:(n+1)*sT, :] = np.random.choice(sK, size=nA, replace=False, p=PDF_ANNOT)  # Annotators
            for nt in range(n*sT, (n+1)*sT):    # Iterate over time-instances in this Segment
                for k in A[nt]:    # Iterate over Annotators chosen in this time-instant
                    u_k = np.random.choice(sU, p=psi[Z[nt], k, :])  # Compute Annotator Emission (confusion)
                    if omega[S[nt], u_k, u_k] == 1: Y[nt, k] = u_k  # Project Observation (if in schema)
        # Now store sizes
        for s in range(sS):
            run_sizes[run, s] = np.equal(S, s).sum()

        # [B] - Train DS Model
        if args.output[DSS].lower() != 'none':
            print(' - Training DS Model (per-Schema):')
            for s in range(sS):     # Have to train independently per schema
                print(' ---> Schema {}'.format(s))
                schema_idcs = np.equal(S, s)
                for fold in range(sF):  # Iterate over folds
                    print('   ------> Fold {}'.format(fold))
                    # Split Training/Validation Sets
                    train_idcs = np.logical_and(np.not_equal(F, fold), schema_idcs)
                    valid_idcs = np.logical_and(np.equal(F, fold), schema_idcs)
                    # Get Label-Set associated with this Schema
                    label_set = CDTYPES[s].categories.values
                    sL = len(label_set)
                    priors = [np.ones(sL), np.ones([sL, sK * sL])]  # Prior Smoothers of 1s
                    starts = [(npext.sum_to_one(np.ones(sL)),       # Starting Probabilities
                               np.tile(npext.sum_to_one(np.eye(sL, sL) + np.full([sL, sL], fill_value=0.01), axis=1), sK))]
                    # For Legacy Reasons, we need 1-Hot Encoded Data
                    _Y_Hot_train = pd.get_dummies(pd.DataFrame(Y[train_idcs]).astype(CDTYPES[s])).values
                    _Y_Hot_valid = pd.get_dummies(pd.DataFrame(Y[valid_idcs]).astype(CDTYPES[s])).values
                    sM = len(_Y_Hot_train)
                    # Train Model
                    ds_model = AnnotDS([sM, sL, sK, sL], -1, 100, sink=sys.stdout)
                    results = ds_model.fit_model(_Y_Hot_train, priors, starts)
                    # Validate Model
                    map_pred = ds_model.estimate_map(results['Pi'], results['Psi'], _Y_Hot_valid, label_set, max_size=sZ)
                    pred_corr_DS_S[run, s] += np.equal(np.argmax(map_pred, axis=1), Z[valid_idcs]).sum()
                    pred_wght_DS_S[run, s] += np.log(map_pred[np.arange(len(map_pred)), Z[valid_idcs]]).sum()

        # [C] - Train DS Model Holistically
        if args.output[DSA].lower() != 'none':
            print(' - Training DS Model (Holistically):')
            for fold in range(sF):  # Iterate over folds
                print(' ---> Fold {}'.format(fold))
                # Split Training/Testing Sets
                train_idcs = np.not_equal(F, fold)
                valid_idcs = np.equal(F, fold)
                priors = [np.ones(sZ), np.ones([sZ, sK * sU])]  # Prior Smoothers of 1s
                starts = [(npext.sum_to_one(np.ones(sZ)),  # Starting Probabilities
                           np.tile(npext.sum_to_one(np.eye(sZ, sU) + np.full([sZ, sU], fill_value=0.01), axis=1), sK))]
                _Y_Hot_train = pd.get_dummies(pd.DataFrame(Y[train_idcs]).astype(CDTYPES[-1])).values
                _Y_Hot_valid = pd.get_dummies(pd.DataFrame(Y[valid_idcs]).astype(CDTYPES[-1])).values
                label_set = CDTYPES[-1].categories.values
                sM = len(_Y_Hot_train)
                # Train Model
                ds_model = AnnotDS([sM, sZ, sK, sU], -1, 100, sink=sys.stdout)
                results = ds_model.fit_model(_Y_Hot_train, priors, starts)
                # Validate Model
                Z_valid = Z[valid_idcs]
                S_valid = S[valid_idcs]
                map_pred = ds_model.estimate_map(results['Pi'], results['Psi'], _Y_Hot_valid, label_set, max_size=sZ)
                predictions = np.argmax(map_pred, axis=1)
                pred_likels = map_pred[np.arange(len(map_pred)), Z_valid]
                for s in range(sS):
                    schema_idcs = np.equal(S_valid, s)
                    pred_corr_DS_A[run, s] += np.equal(predictions[schema_idcs], Z_valid[schema_idcs]).sum()
                    pred_wght_DS_A[run, s] += np.log(pred_likels[schema_idcs]).sum()

        # [D] - Train ISAR Model - but first, we have to identify NIS (i.e. when the responsible annotators do not
        #       provide a label.
        if args.output[ISAR].lower() != 'none':
            print(' - Training ISAR Model (holistically):')
            for nt in range(sN*sT):
                for a in A[nt, :]:
                    if np.isnan(Y[nt, a]): Y[nt, a] = NIS # If we are putting all as NIS
            for fold in range(sF):  # Iterate over folds
                print(' ---> Fold {}'.format(fold))
                # Split Training/Testing Sets
                train_idcs = np.not_equal(F, fold)
                valid_idcs = np.equal(F, fold)
                priors = [np.ones(sZ), np.ones([sZ, sK, sU])]
                starts = [(npext.sum_to_one(np.ones(sZ)),
                           np.stack([npext.sum_to_one(np.eye(sZ, sZ) + np.full([sZ, sZ], fill_value=0.01), axis=1)
                                    for _ in range(sK)]).swapaxes(0, 1))]
                # Train Model
                isar_model = AnnotISAR(omega, -1, 100, sink=sys.stdout)
                results = isar_model.fit_model(Y[train_idcs], S[train_idcs], priors, starts)
                # Validate Model
                Z_valid = Z[valid_idcs]
                S_valid = S[valid_idcs]
                m_omega = AnnotISAR.omega_msg(omega, Y[valid_idcs], S_valid)
                map_pred = isar_model.estimate_map(results.Pi, results.Psi, m_omega, None)
                predictions = np.argmax(map_pred, axis=1)
                pred_likels = map_pred[np.arange(len(map_pred)), Z_valid]
                for s in range(sS):
                    schema_idcs = np.equal(S_valid, s)
                    pred_corr_ISAR[run, s] += np.equal(predictions[schema_idcs], Z_valid[schema_idcs]).sum()
                    pred_wght_ISAR[run, s] += np.log(pred_likels[schema_idcs]).sum()

    # ===== Finally store the results to File: ===== #
    print('Storing Results to file ... ')
    if args.output[DSS].lower() != 'none':
        np.savez_compressed(args.output[DSS], accuracy=pred_corr_DS_S, log_loss=-pred_wght_DS_S, sizes=run_sizes)
    if args.output[DSA].lower() != 'none':
        np.savez_compressed(args.output[DSA], accuracy=pred_corr_DS_A, log_loss=-pred_wght_DS_A, sizes=run_sizes)
    if args.output[ISAR].lower() != 'none':
        np.savez_compressed(args.output[ISAR], accuracy=pred_corr_ISAR, log_loss=-pred_wght_ISAR, sizes=run_sizes)
