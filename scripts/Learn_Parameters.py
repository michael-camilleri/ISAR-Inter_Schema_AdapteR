"""
Script to analyse the ability of ISAR to learn the true parameters the data-generating process.

Note that while the script allows some flexibility, it should probably not combine sizes < [13, 11] with non extreme
(one-vs-rest) scenarios.

Note also that in these simulations, it is (again) assumed that each annotator either labels a segment or puts NIS.
This stems from our modelling behaviour, in that if the annotator does not label the sample it is solely because it is
not in-schema (i.e. we do not model missing-at-random behaviour.
"""

from mpctools.extensions import npext
import numpy as np
import argparse
import sys

# Load own packages
sys.path.append('..')
from isar.models import InterSchemaAdapteRIID

# Default Parameters
DEFAULTS = {'Output':  '../../data/Parameters_ISAR',      # Result File
            'Random':  '0',                               # Random Seed offset
            'Numbers': ['0', '20'],                       # Range: start index, number of runs
            'Lengths': ['60', '5400'],                    # Number and length of segments
            'Sizes':   ['13', '11'],                      # Dimensionality of the data: sZ/sK
            'Steps':   ['0.001', '0.005', '0.01', '0.05', '0.1', '0.5', '1.0'],  # Step Sizes
            'Extreme': False,                             # One-V-Rest?
            'Different': False                            # Different schemas within Sample?
            }
nA = 3

if __name__ == '__main__':

    # ==== Parse Arguments ==== #
    _arg_parse = argparse.ArgumentParser(description='Simulate and Train ISAR to evaluate Parameter Learning')
    _arg_parse.add_argument('-o', '--output', help='Output Result file: defaults to {}'.format(DEFAULTS['Output']),
                            default=DEFAULTS['Output'])
    _arg_parse.add_argument('-r', '--random', help='Seed (offset) for all Random States: ensures repeatibility. '
                                                   'Defaults to {}'.format(DEFAULTS['Random']),
                            default=DEFAULTS['Random'])
    _arg_parse.add_argument('-n', '--numbers', help='Range of Runs to simulate: tuple containing the start index and '
                                                    'number of runs. Defaults to [{} {}]'.format(*DEFAULTS['Numbers']),
                            nargs=2, default=DEFAULTS['Numbers'])
    _arg_parse.add_argument('-l', '--lengths', help='Number and length of each segment. Default {}'
                            .format(DEFAULTS['Lengths']), default=DEFAULTS['Lengths'], nargs=2)
    _arg_parse.add_argument('-s', '--sizes', help='Dimensionality of the Problem in terms of number of latent states '
                                                  '|Z| and number of annotators |K|. Default is {} (i.e. the same as in'
                                                  ' the real data). Note that the values cannot exceed these defaults, '
                                                  'since the parameters are extracted as sub-matrices of the true ones.'
                                                  ' Note also that the behaviour is undefined if |Z| is not {} when '
                                                  'using MRC Harwell schemas.'
                            .format(DEFAULTS['Sizes'], DEFAULTS['Sizes'][0]), default=DEFAULTS['Sizes'], nargs=2)
    _arg_parse.add_argument('-i', '--increments', help='Fractions of the data-set to evaluate the parameters at. Must '
                                                       'range from >0 to 1.0', nargs='*', default=DEFAULTS['Steps'])
    _arg_parse.add_argument('-d', '--different', help='If set, then annotators may label using different schemas per '
                                                      'sample: otherwise, within the same sample, all annotators use '
                                                      'the same schema. Default is False.', action='store_true')
    _arg_parse.add_argument('-e', '--extreme',  help='If set, then there are |Z| schemas corresponding to |Z| latent '
                                                     'states in a one-vs-rest configuration: otherwise (default) use '
                                                     'the MRC Harwell schema set. Note that in the former case, the '
                                                     'probability over Schemas and Annotators is uniform.',
                            action='store_true')

    _arg_parse.set_defaults(different=DEFAULTS['Different'])
    _arg_parse.set_defaults(extreme=DEFAULTS['Extreme'])
    # ---- Parse Arguments and transform as required ---- #
    args = _arg_parse.parse_args()
    args.random = int(args.random)
    run_offset = int(args.numbers[0])
    run_length = int(args.numbers[1])
    args.increments = [float(s) for s in args.increments]
    sN, sT = [int(l) for l in args.lengths]
    sZ, sK = [int(s) for s in args.sizes]
    sU = sZ
    NIS = sZ  # NIS Label

    # ==== Load/Prepare the Data ==== #
    with np.load('../../data/model.mrc.npz') as _data:
        # ---- Load Baseline Components ---- #
        pi = npext.sum_to_one(_data['pi'][:sZ])                         # Note that we have to ensure normalisation
        psi = npext.sum_to_one(_data['psi'][:sZ, :sK, :sU], axis=-1)
        # ---- Now Handle Omega ---- #
        if args.extreme:
            sS = sZ
            sY = sZ+1
            omega = np.zeros([sS, sU, sY])
            for s in range(sS):
                for u in range(sU):
                    for y in range(sY):
                        if (y == u) and (u == s):
                            omega[s, u, y] = 1.0
                        elif (y == sY-1) and (u != s):
                            omega[s, u, y] = 1.0
        else:
            omega = _data['omega']
            sS, _, sY = omega.shape
        # ---- Also Handle Probabilities ---- #
        if args.extreme:
            PDF_ANNOT = npext.sum_to_one(np.ones(sK))   # Probability over Annotators
            PDF_SCHEMA = npext.sum_to_one(np.ones(sS))  # Probability over Schemas
        else:
            PDF_ANNOT = npext.sum_to_one([49, 31, 11, 4, 10, 25, 9, 7, 6, 3, 3][:sK])  # Probability over Annotators
            PDF_SCHEMA = npext.sum_to_one([13, 15, 17, 10])                            # Probability over Schemas

    # ==== Simulate (and Learn) Model(s) ==== #
    # ---- Prepare Storage ---- #
    # Basically, for each parameter, we index by run, then progression in number of steps and finally the dimensions of
    #  the parameter itself.
    pi_true = np.empty([run_length, sZ])                         # True Values
    pi_isar = np.empty([run_length, len(args.increments), sZ])   # As learnt using ISAR
    pi_full = np.zeros([run_length, len(args.increments), sZ])   # As learnt from fully-observeable data
    psi_true = np.empty([run_length, sZ, sK, sU])
    psi_isar = np.empty([run_length, len(args.increments), sZ, sK, sU])
    psi_full = np.zeros([run_length, len(args.increments), sZ, sK, sU])
    sizes = np.zeros([run_length, len(args.increments)], dtype=int) # Sizes of the Data

    # --- Iterate over Runs ---- #
    for run in range(run_offset, run_offset + run_length):
        print('Executing Simulation Run: {} ...'.format(run))

        # Seed with Random Offset + run value
        np.random.seed(args.random + run)

        # [A] - Generate Data
        print(' - Generating Data:')
        # Add some noise to Parameters to introduce some randomness
        if run == run_offset:
            Pi = pi
            Psi = psi
        else:
            Pi = npext.sum_to_one(pi + np.random.uniform(0.0, 0.05, size=sZ))
            Psi = npext.sum_to_one(psi + np.random.uniform(0.0, 0.05, size=[sZ, sK, sU]), axis=-1)
        # Keep Track of Parameters
        pi_true[run, :] = Pi
        psi_true[run, :, :, :] = Psi
        # Latent State
        Z = np.random.choice(sZ, size=sN * sT, p=Pi)  # Latent State
        # Schema - need to handle different cases
        S = np.repeat(np.random.choice(sS, size=[sN, sK], p=PDF_SCHEMA), sT, axis=0) if args.different else \
            np.repeat(np.random.choice(sS, size=sN, p=PDF_SCHEMA), sT)
        # The Observations, we have to do per-sample due to the ancestral sampling nature.
        U = np.full([sN*sT, sK], fill_value=np.NaN)
        Y = np.full([sN*sT, sK], fill_value=np.NaN)
        for n in range(sN):  # Iterate over all segments
            A_nt = np.random.choice(sK, size=nA, replace=False, p=PDF_ANNOT)  # Annotators
            for nt in range(n*sT, (n+1)*sT):    # Iterate over time-instances in this Segment
                for k in A_nt:    # Iterate over Annotators chosen in this time-instant
                    U[nt, k] = np.random.choice(sU, p=Psi[Z[nt], k, :])            # Compute Annotator Confusion
                    if args.different:
                        Y[nt, k] = U[nt, k] if (omega[S[nt, k], int(U[nt, k]), int(U[nt, k])] == 1) else NIS  # Project
                    else:
                        Y[nt, k] = U[nt, k] if (omega[S[nt], int(U[nt, k]), int(U[nt, k])] == 1) else NIS  # Project
        # And Shuffle all to avoid same sizes when
        permutation = np.random.permutation(sN*sT)
        Z = Z[permutation]
        S = S[permutation, :] if args.different else S[permutation]
        U = U[permutation, :]
        Y = Y[permutation, :]

        # [B] - Learn using fully-labelled Data
        print(' - Learning using fully-observed Data')
        for i, inc in enumerate(args.increments):
            # Extract Sub-Set to operate on
            sizes[run, i] = int(inc * sN * sT)
            z_i = Z[:sizes[run, i]]
            u_i = U[:sizes[run, i], :]
            # Train Pi: basically relative counts of all latent-states
            idcs, cnts = np.unique(z_i, return_counts=True)
            pi_full[run, i, idcs] = npext.sum_to_one(cnts)
            # Train Psi:
            for z in range(sZ):
                u_z = u_i[z_i == z, :]      # Extract only where Z is equal to z
                for k in range(sK):
                    u_kz = u_z[:, k]
                    idcs, cnts = np.unique(u_kz[~np.isnan(u_kz)], return_counts=True)
                    psi_full[run, i, z, k, idcs.astype(int)] = npext.sum_to_one(cnts)

        # [C] - Learn using ISAR
        print(' - Learning using ISAR')
        for i, inc in enumerate(args.increments):
            # Extract Sub-Set(s) to operate on
            y_i = Y[:sizes[run, i], :]
            s_i = S[:sizes[run, i], :] if args.different else S[:sizes[run, i]]
            priors = [np.ones(sZ)*2, np.ones([sZ, sK, sU])*2]
            starts = [(npext.sum_to_one(np.ones(sZ)),
                       np.stack([npext.sum_to_one(np.eye(sZ, sZ) + np.full([sZ, sZ], fill_value=0.01), axis=1)
                                 for _ in range(sK)]).swapaxes(0, 1))]
            # Train ISAR Model
            isar_model = InterSchemaAdapteRIID(omega, -1, 200, sink=sys.stdout)
            results = isar_model.fit_model(y_i, s_i, priors, starts)
            # Store results
            pi_isar[run, i, :] = results.Pi
            psi_isar[run, i, :, :, :] = results.Psi

    # ===== Finally store the results to File: ===== #
    print('Storing Results to file ... ')
    np.savez_compressed(args.output, Pi_true=pi_true, Pi_full=pi_full, Pi_isar=pi_isar, Psi_true=psi_true,
                        Psi_full=psi_full, Psi_isar=psi_isar, Sizes=sizes)
