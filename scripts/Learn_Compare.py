"""
Script to compare, in simulation, the performance of the ISAC model to plain old DS, under realistic (simulation)
conditions.

The Simulation process does not generate any Missing-At-Random data, and hence, in this case, all Samples where the
assigned annotator does not provide a label, are interpreted as NIS.
"""
from mpctools.extensions import npext
from itertools import cycle
import numpy as np
import argparse
import sys

# Load own packages
sys.path.append('..')
from isar.models import DawidSkeneIID, InterSchemaAdapteRIID

# Default Parameters
DEFAULTS = \
    {'Output': ['../data/Compare_DS_F1', '../data/Compare_ISAR_F1'],  # File-Names
     'Random': '0',                                             # Random Seed offset
     'Numbers': ['0', '10'],                                    # Range: start index, number of runs
     'Lengths': ['50', '100'],                                  # Number and length of segments
     'Schemas': ['13', '15', '17', '10'],                       # Probability over Schemas,
     'Pi': 'unif',                                              # Pi mode
     'Folds': '5'}                                             # Number of Folds to use (folding is by Segment)

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
    # Note that we use the same array for both: DS appears in the first, and ISAR in the second position
    if args.output[DS].lower() != 'none':
        pred_corr_DS = np.zeros([run_length, sS], dtype=float)      # Array for Absolute Accuracy
        pred_wght_DS = np.zeros([run_length, sS], dtype=float)      # Array for Predictive Log Probability
        pred_f1_DS = np.zeros([run_length, sS+1], dtype=float)      # F1-Score Array - this requires global computation
    else:
        pred_corr_DS = None; pred_wght_DS = None; pred_f1_DS = None
    if args.output[ISAR].lower() != 'none':
        pred_corr_ISAR = np.zeros([run_length, sS], dtype=float)
        pred_wght_ISAR = np.zeros([run_length, sS], dtype=float)
        pred_f1_ISAR = np.zeros([run_length, sS+1], dtype=float)
    else:
        pred_corr_ISAR = None; pred_wght_ISAR = None; pred_f1_ISAR = None
    run_sizes = np.zeros([run_length, sS], dtype=float)         # Number of Samples per Schema per Run

    # --- Iterate over Runs ---- #
    for run in range(run_offset, run_offset + run_length):
        print('Executing Simulation Run: {} ...'.format(run))

        # Seed with Random Offset + run value
        np.random.seed()

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
            # Prepare some placeholders for the F1-scores
            z_predictions = np.empty_like(Z)
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
                results = isar_model.fit(Y_train, S_train, priors, starts)
                # Validate Model
                z_posterior = isar_model.predict_proba(Y_valid, S_valid)
                z_predict = np.argmax(z_posterior, axis=-1)
                z_log_correct_post = z_posterior[np.arange(len(Z_valid)), Z_valid]
                for s in range(sS):
                    schema_idcs = np.equal(S_valid, s)
                    pred_corr_ISAR[run, s] += np.equal(z_predict[schema_idcs], Z_valid[schema_idcs]).sum()
                    pred_wght_ISAR[run, s] += np.log(z_log_correct_post[schema_idcs]).sum()

        # [C] - Train DS Model Holistically
        if args.output[DS].lower() != 'none':
            print(' - Training DS Model (Holistically):')
            # First need to convert NIS to NaN
            Y[Y == sZ] = np.NaN
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
                z_predict = ds_model.predict(U_valid)
                z_log_correct_post = ds_model.predict_proba(U_valid)[np.arange(len(Z_valid)), Z_valid]
                for s in range(sS):
                    schema_idcs = np.equal(S_valid, s)
                    pred_corr_DS[run, s] += np.equal(z_predict[schema_idcs], Z_valid[schema_idcs]).sum()
                    pred_wght_DS[run, s] += np.log(z_log_correct_post[schema_idcs]).sum()

    # ===== Finally store the results to File: ===== #
    print('Storing Results to file ... ')
    if args.output[DS].lower() != 'none':
        np.savez_compressed(args.output[DS], accuracy=pred_corr_DS, log_loss=-pred_wght_DS, sizes=run_sizes)
    if args.output[ISAR].lower() != 'none':
        np.savez_compressed(args.output[ISAR], accuracy=pred_corr_ISAR, log_loss=-pred_wght_ISAR, sizes=run_sizes)
