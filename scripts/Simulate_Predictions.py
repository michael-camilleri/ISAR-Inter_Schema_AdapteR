"""
Script to compare, in simulation, the performance of the ISAC model to plain old DS, under realistic (simulation)
conditions.

The Simulation process does not generate any Missing-At-Random data, and hence, in this case, all Samples where the
assigned annotator does not provide a label, are interpreted as NIS.

We report three scores:
 * Accuracy: simplest metric
 * Macro-F1: to avoid label imbalance (basically all important)
 * Predictive Log-Likelihood

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
http://www.gnu.org/licenses/.

Author: Michael P. J. Camilleri
"""

from sklearn.metrics import accuracy_score, f1_score
from mpctools.extensions import npext
from itertools import cycle
import numpy as np
import argparse
import sys

# Load own packages
sys.path.append('..')
from isar.models import DawidSkeneIID, InterSchemaAdapteRIID

# 1 - Reduced
# 2 - Uniform
# 3 - Dirich (10)
# 4 - Biased  (True)
# 5 - Biased (Unif)
# Default Parameters
DEFAULTS = \
    {'Output': ['../data/Compare_DS', '../data/Compare_ISAR'],    # File-Names
     'Random': '0',                                               # Random Seed offset
     'Numbers': ['0', '20'],                                      # Range: start index, number of runs
     'Lengths': ['60', '5400'],                                   # Number and length of segments
     'Schemas': ['13', '15', '17', '10'],                         # Probability over Schemas,
     'Pi': 'true',                                                # Pi mode
     'Folds': '10'}                                               # Number of Folds to use (folding is by Segment)

# Some Constants
PDF_ANNOT = npext.sum_to_one([49, 31, 11, 4, 10, 25, 9, 7, 6, 3, 3])  # Probability over Annotators
sK = len(PDF_ANNOT)
sS = 4
nA = 3              # Number of Annotators in any one sample
DS, ISAR = (0, 1)   # Position (index) into arrays

if __name__ == '__main__':

    # ==== Parse Arguments ==== #
    _arg_parse = argparse.ArgumentParser(description='Simulate and Train Annotator models (based on DS and ISAR)')
    _arg_parse.add_argument('-o', '--output', help='Output Result files, one each for the results from the DS (trained '
                                                   'holistically) and ISAR models. Put "None" for any that you do not '
                                                   'want to simulate. Defaults to {}'
                            .format(DEFAULTS['Output']), default=DEFAULTS['Output'], nargs=2)
    _arg_parse.add_argument('-r', '--random', help='Seed (offset) for all Random States: ensures repeatibility. '
                                                   'Defaults to {}'.format(DEFAULTS['Random']),
                            default=DEFAULTS['Random'])
    _arg_parse.add_argument('-n', '--numbers', help='Range of Runs to simulate: tuple containing the start index and '
                                                    'number of runs. Defaults to [{} {}]'.format(*DEFAULTS['Numbers']),
                            nargs=2, default=DEFAULTS['Numbers'])
    _arg_parse.add_argument('-l', '--lengths', help='Number and length of each segment. Default {}'
                            .format(DEFAULTS['Lengths']), default=DEFAULTS['Lengths'], nargs=2)
    _arg_parse.add_argument('-f', '--folds', help='Number of splits (folds) to operate on per run. Default {}'
                            .format(DEFAULTS['Folds']), default=DEFAULTS['Folds'])
    _arg_parse.add_argument('-s', '--schemas', help='Schema Probabilities. Default is: {}'.format(DEFAULTS['Schemas']),
                            default=DEFAULTS['Schemas'], nargs=sS)
    _arg_parse.add_argument('-p', '--pi', help='How to simulate Pi: can be {{true}} to use the PI learnt from the MRC '
                                               'Harwell data, {{unif}} to use a fully uniform distribution or {{f}} '
                                               'where f is a number, in which case it is interpreted as the alpha'
                                               'weight for the Dirichlet from which Pi is sampled. Default is "{}"'
                            .format(DEFAULTS['Pi']), default=DEFAULTS['Pi'])
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
    np.random.seed(args.random)
    # ---- Load Baseline models ---- #
    with np.load('../data/model.mrc.npz') as _data:
        omega = _data['omega']
        if args.pi.lower() == 'true':
            pi = _data['pi']
        elif args.pi.lower() == 'unif':
            pi = npext.sum_to_one(np.ones_like(_data['pi']))
        else:
            pi = np.random.dirichlet(np.full_like(_data['pi'], float(args.pi)))
        psi = _data['psi']
    # ---- Extract Sizes ---- #
    sZ, sK, sU = psi.shape       # Latent Model [Z, #Annotators, U]
    sS, _, sY = omega.shape      # Number of Schemas and Observeable Labels
    NIS = sZ                     # NIS Label
    iterator = cycle(range(sF))  # Fold Iterator

    # ==== Simulate (and Learn) Model(s) ==== #
    # ---- Prepare Placeholders ---- #
    # Note that we use the same array for both: DS appears in the first, and ISAR in the second position. Also, we
    #  compute both per-schema and global scores (after doing all folds)!
    if args.output[DS].lower() != 'none':
        pred_acc_DS = np.zeros([run_length, sS + 1], dtype=float)     # Array for Absolute Accuracy
        pred_wght_DS = np.zeros([run_length, sS + 1], dtype=float)    # Array for Predictive Log Probability
        pred_f1_DS = np.zeros([run_length, sS + 1], dtype=float)      # F1-Score Array
    else:
        pred_acc_DS = None; pred_wght_DS = None; pred_f1_DS = None
    if args.output[ISAR].lower() != 'none':
        pred_acc_ISAR = np.zeros([run_length, sS + 1], dtype=float)
        pred_wght_ISAR = np.zeros([run_length, sS + 1], dtype=float)
        pred_f1_ISAR = np.zeros([run_length, sS + 1], dtype=float)
    else:
        pred_acc_ISAR = None; pred_wght_ISAR = None; pred_f1_ISAR = None
    run_sizes = np.zeros([run_length, sS], dtype=float)         # Number of Samples per Schema per Run

    # --- Iterate over Runs ---- #
    for run in range(run_offset, run_offset + run_length):
        print('Executing Simulation Run: {} ...'.format(run))

        # Seed with Random Offset + run value
        np.random.seed(args.random + run)

        # [A] - Generate Data
        print(' - Generating Data:')
        sampler = InterSchemaAdapteRIID([sZ, sK, sS], omega, params=(pi, psi), random_state=args.random + run)
        Z, S, A, Y = sampler.sample(sN, sT, nA, pdf_schema, PDF_ANNOT)
        F = np.repeat([next(iterator) for _ in range(sN)], sT)  # Folds: sequentially
        # Now store sizes
        for s in range(sS):
            run_sizes[run, s] = np.equal(S, s).sum()

        # [B] - Train ISAR Model
        if args.output[ISAR].lower() != 'none':
            print(' - Training ISAR Model (holistically):')
            # Prepare some placeholders for the scores
            z_predictions = np.empty_like(Z)
            z_log_cposter = np.empty_like(Z, dtype=float)
            for fold in range(sF):  # Iterate over folds
                print(' ---> Fold {}'.format(fold))
                # Split Training/Testing Sets
                train_idcs = np.not_equal(F, fold)
                valid_idcs = np.equal(F, fold)
                priors = [np.ones(sZ) * 2, np.ones([sZ, sK, sU]) * 2]
                starts = [(npext.sum_to_one(np.ones(sZ)),
                           np.tile(npext.sum_to_one(np.eye(sZ, sU) + np.full([sZ, sU], fill_value=0.01), axis=1)[:, np.newaxis, :], [1, sK, 1]))]
                # Train Model
                Y_train = Y[train_idcs]
                Y_valid = Y[valid_idcs]
                S_train = S[train_idcs]
                S_valid = S[valid_idcs]
                Z_valid = Z[valid_idcs]
                isar_model = InterSchemaAdapteRIID([sZ, sK, sS], omega, random_state=args.random + run, sink=sys.stdout)
                isar_model.fit(Y_train, S_train, priors, starts)
                # Validate Model
                z_posterior = isar_model.predict_proba(Y_valid, S_valid)
                z_predictions[valid_idcs] = np.argmax(z_posterior, axis=-1)
                z_log_cposter[valid_idcs] = z_posterior[np.arange(len(Z_valid)), Z_valid]
            # Now Store Values
            for s in range(sS):
                schema_idcs = np.equal(S, s)
                pred_acc_ISAR[run, s] = accuracy_score(Z[schema_idcs], z_predictions[schema_idcs])
                pred_wght_ISAR[run, s] = np.log(z_log_cposter[schema_idcs]).mean()
                pred_f1_ISAR[run, s] = f1_score(Z[schema_idcs], z_predictions[schema_idcs], labels=np.arange(sZ), average='macro')
            # And Globals
            pred_acc_ISAR[run, -1] = accuracy_score(Z, z_predictions)
            pred_wght_ISAR[run, -1] = np.log(z_log_cposter).mean()
            pred_f1_ISAR[run, -1] = f1_score(Z, z_predictions, labels=np.arange(sZ), average='macro')

        # [C] - Train DS Model Holistically
        if args.output[DS].lower() != 'none':
            print(' - Training DS Model (Holistically):')
            # First need to convert NIS to NaN
            Y[Y == sZ] = np.NaN
            # Prepare some placeholders for the scores
            z_predictions = np.empty_like(Z)
            z_log_cposter = np.empty_like(Z, dtype=float)
            for fold in range(sF):  # Iterate over folds
                print(' ---> Fold {}'.format(fold))
                # Split Training/Testing Sets and get the relevant subsets
                train_idcs = np.not_equal(F, fold)
                valid_idcs = np.equal(F, fold)
                U_train = Y[train_idcs]
                U_valid = Y[valid_idcs]
                Z_valid = Z[valid_idcs]
                S_valid = S[valid_idcs]
                # Train Model
                priors = [np.ones(sZ)*2, np.ones([sZ, sK, sU])*2]   # Prior Probabilities (actual alphas)
                starts = [(npext.sum_to_one(np.ones(sZ)),           # Starting Probabilities
                           np.tile(npext.sum_to_one(np.eye(sZ, sU) + np.full([sZ, sU], fill_value=0.01), axis=1)[:, np.newaxis, :], [1, sK, 1]))]
                ds_model = DawidSkeneIID([sZ, sK], None, random_state=args.random + run, sink=sys.stdout)
                ds_model.fit(U_train, None, priors, starts)
                # Validate Model
                z_posterior = ds_model.predict_proba(U_valid)
                z_predictions[valid_idcs] = np.argmax(z_posterior, axis=-1)
                z_log_cposter[valid_idcs] = z_posterior[np.arange(len(Z_valid)), Z_valid]
            # Now Store Values
            for s in range(sS):
                schema_idcs = np.equal(S, s)
                pred_acc_DS[run, s] = accuracy_score(Z[schema_idcs], z_predictions[schema_idcs])
                pred_wght_DS[run, s] = np.log(z_log_cposter[schema_idcs]).mean()
                pred_f1_DS[run, s] = f1_score(Z[schema_idcs], z_predictions[schema_idcs], labels=np.arange(sZ),
                                                average='macro')
            # And Globals
            pred_acc_DS[run, -1] = accuracy_score(Z, z_predictions)
            pred_wght_DS[run, -1] = np.log(z_log_cposter).mean()
            pred_f1_DS[run, -1] = f1_score(Z, z_predictions, labels=np.arange(sZ), average='macro')

    # ===== Finally store the results to File: ===== #
    print('Storing Results to file ... ')
    if args.output[DS].lower() != 'none':
        np.savez_compressed(args.output[DS], accuracy=pred_acc_DS, f1=pred_f1_DS, log_loss=-pred_wght_DS)
    if args.output[ISAR].lower() != 'none':
        np.savez_compressed(args.output[ISAR], accuracy=pred_acc_ISAR, f1=pred_f1_ISAR, log_loss=-pred_wght_ISAR)
