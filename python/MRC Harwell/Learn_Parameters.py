##########################################################################
## Script to analyse the ability of ISAR to learn the true parameters   ##
## the data-generating process.                                         ##
## Note that while the script allows some flexibility, it should        ##
## probably not combine sizes < [13, 11] with non extreme scenarios.    ##
##########################################################################

from pandas.api.types import CategoricalDtype as CDType
import pandas as pd
import numpy as np
import argparse
import sys

# Load own packages
sys.path.append('..')
from Tools import npext
from Models import AnnotISAR

# Default Parameters
DEFAULTS = {'Output':  ['../../data/Parameters_ISAR'],    # Result File
            'Random':  '0',                               # Random Seed offset
            'Numbers': ['0', '20'],                       # Range: start index, number of runs
            'Lengths': ['60', '5400'],                    # Number and length of segments
            'Sizes':   ['13', '11'],                      # Dimensionality of the data: sZ/sK
            'Steps':   ['0.001', '0.005', '0.01', '0.05', '0.1', '0.5', '1.0']  # Step Sizes
            }

if __name__ == '__main__':

    # ==== Parse Arguments ==== #
    _arg_parse = argparse.ArgumentParser(description='Simulate and Train ISAR to evaluate Parameter Learning')
    _arg_parse.add_argument('-o', '--output', help='Output Result file: defaults to {}'.format(DEFAULTS['Output']),
                            default=DEFAULTS['Output'])
    _arg_parse.add_argument('-r', '--random', help='Seed (offset) for all Random States: ensures repeatibility. '
                                                   'Defaults to {}'.format(DEFAULTS['Random']),
                            default=DEFAULTS['Random'])
    _arg_parse.add_argument('-n', '--numbers', help='Range of Runs to simulate: tuple containing the start index and '
                                                    'number of runs. Defaults to {}/{}'.format(*DEFAULTS['Numbers']),
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

    _arg_parse.set_defaults(different=False)
    _arg_parse.set_defaults(extreme=False)
    # ---- Parse Arguments and transform as required ---- #
    args = _arg_parse.parse_args()
    args.random = int(args.random)
    run_offset = int(args.numbers[0])
    run_length = int(args.numbers[1])
    args.steps = [float(s) for s in args.steps]
    sN, sT = [int(l) for l in args.lengths]
    sZ, sK = [int(s) for s in args.sizes]
    sU = sZ

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




#
# sK = len(PDF_ANNOT)
# PDF_SCHEMA =

# #
#
# # Chose the Simulation Type
# #   * 0 = Extreme Case: 1 schema per label with inter-annotator disagreement
# #   * 1 = Realistic Case: Real-Schema configuration
#
# SIM_TYPE = 0
# FLIP = 2 # -1 #
#
# # Testing Parameters
# DUMP_NAME = 'SYNTH_{0:03d}'
# DATA_PERC = [1.0] #
#
# # Simulation Parameters
# if SIM_TYPE == 0:
#     sN = 500  # Number of Segments to generate
#     sZ = 7    # Number of Latent States
#     sT = 100  # Number of samples per segment
#     sU = 7    # Number of Annotator States
#     sX = 8    # Includes NIS
#     nA = 3    # Number of Annotators in any one sample
#     sK = 6    # Number of Annotators
#     sS = 7    # Number of Schemas
#
#     SperK = True    # Whether to simulate a different schema per annotator...
#
#     pAnnot =    # Probability over Annotators
#     pSchema =   # Probability over Schemas
#     Xi =
#
#     NIS_ALL = True
#
# elif SIM_TYPE == 1:
#     sN = 60   # Number of Segments to generate
#     sZ = 13   # Number of Latent States
#     sT = 5400 # Number of samples per segment
#     sU = 13   # Number of Annotator States
#     sX = 14   # Includes NIS
#     nA = 3    # Number of Annotators in any one sample
#     sK = 11   # Number of Annotators
#     sS = 4    # Number of Schemas
#
#     SperK = False  # Whether to simulate a different schema per annotator...
#
#     pAnnot = npext.sum_to_one([49, 31, 11, 4, 10, 25, 9, 7, 6, 3, 3])  # Probability over Annotators
#     pSchema = npext.sum_to_one([13, 15, 17, 10])                       # Probability over Schemas
#     Xi = np.load(os.path.join(Const['Data.Clean'], 'XI.npy'))
#
#     NIS_ALL = True
# else:
#     pSchema = []
#     pAnnot = []
#     Xi = None
#     sN = None  # Number of Segments to generate
#     sZ = None  # Number of Latent States
#     sT = None  # Number of samples per segment
#     sU = None  # Number of Annotator States
#     sX = None  # Includes NIS
#     nA = None  # Number of Annotators in any one sample
#     sK = None  # Number of Annotators
#     sS = None  # Number of Schemas
#     SperK = None  # Whether to simulate a different schema per annotator...
#     NIS_ALL = None
#
# NIS_idx = sZ  # NIS Index
#
# # Load the Parameters from the real-data
# _Pi = np.load(os.path.join(Const['Data.Clean'], 'PHI.npy'))
# _Psi = np.load(os.path.join(Const['Data.Clean'], 'PSI.npy'))
#
#
#
# if __name__ == '__main__':
#
#     r_start = 0
#     r_len = 20
#
#     for run in range(r_start, r_start + r_len):
#         print('Simulating Run ', run)
#
#         if run == r_start:
#             Pi = npext.sum_to_one(_Pi[:sZ])
#             Psi = npext.sum_to_one(_Psi[:sZ, :sK, :sU], axis=-1)
#         else:
#             Pi = npext.sum_to_one(_Pi[:sZ] + np.random.uniform(0.0, 0.05, size=sZ))
#             Psi = npext.sum_to_one(_Psi[:sZ, :sK, :sU] + np.random.uniform(0.0, 0.05, size=[sZ, sK, sU]), axis=-1)
#
#         # ================= GENERATE DATA ================= #
#         # Data
#         Z = np.random.choice(sZ, size=[sN*sT], p=Pi)                                    # Latent State
#         if SperK:
#             S = np.repeat(np.random.choice(sS, size=[sN, sK], p=pSchema), sT, axis=0)   # Schema
#         else:
#             S = np.repeat(np.random.choice(sS, size=sN, p=pSchema), sT)
#         X = np.full(shape=[sN*sT, sK], fill_value=np.NaN)                               # Observed
#
#         # Now do per-sample
#         for n in range(sN):
#             # Decide on Annotators
#             A = np.random.choice(sK, size=nA, replace=False, p=pAnnot)
#
#             # Iterate over Time
#             for t in range(sT):
#                 # Time Instant
#                 _t_i = n*sT + t
#
#                 # Iterate over Annotators chosen
#                 nis = not(NIS_ALL)
#                 for k in A:
#                     # Check Schema for this annotator
#                     s = S[_t_i, k] if SperK else S[_t_i]
#
#                     # Compute Annotator Emission
#                     u_k = np.random.choice(sU, p=Psi[Z[_t_i], k, :])
#
#                     # Compute Observation (if in schema)
#                     if Xi[s, u_k, u_k] == 1:
#                         nis = False
#                         X[_t_i, k] = u_k
#                     elif NIS_ALL:
#                         X[_t_i, k] = NIS_idx
#
#                 # Now potentially assign NIS
#                 if nis:
#                     X[_t_i, A] = NIS_idx
#
#         # Now Prepare for way amenable to models
#         if SperK:
#             _msg_xi = ISAC.msg_xi_update(Xi, X, S)
#         else:
#             _labels_hot = pd.get_dummies(pd.DataFrame(X).astype(CDType(categories=np.arange(sX)))).values
#             _schema_hot = pd.get_dummies(pd.Series(S).astype(CDType(categories=np.arange(sS)))).values
#             _msg_xi = ISAC.msg_xi(Xi, _labels_hot, _schema_hot)
#
#         # Now Permute
#         np.random.shuffle(_msg_xi)
#
#         # Keep Track of Parameters
#         with shelve.open(os.path.join(Const['Results.Scratch'], DUMP_NAME.format(run)), 'n') as storage:
#             storage['Dims'] = [sZ, sK, sU, sX, sS]
#             storage['Z'] = Z
#             storage['S'] = S
#             storage['X_ISAC'] = X
#             storage['True'] = {'Pi': Pi, 'Psi': Psi}
#
#
#             # ================= TRAIN MODELS ================= #
#             print('Training Run ', run)
#
#             _pi = {}
#             _psi = {}
#             _pred = {}
#             for _sub_per in DATA_PERC:
#                 print('Training on {:.1f}% of the Data'.format(_sub_per*100.0))
#                 # Extract Sub-Set to operate on
#                 _sub_set = _msg_xi[:int(_sub_per * sN * sT), :]
#
#                 # Data
#                 if Xi is None:
#                     _common = None
#                 else:
#                     _common = ISAC.EMWorker.ComputeCommon_t(max_iter=200, tolerance=1e-4, update_rate=3, m_xi=_sub_set,
#                                                             prior_phi=np.ones(sZ), prior_psi=np.ones([sZ, sK, sU]))
#                 _data = [npext.sum_to_one(np.ones(sZ)),
#                          np.stack(npext.sum_to_one(np.eye(sZ, sZ) + np.full([sZ, sZ], fill_value=0.01), axis=1)
#                                   for _ in range(sK)).swapaxes(0, 1)]
#
#                 # Now Train ISAC Model
#                 tester = Tester()
#                 isac_worker = ISAC.EMWorker(0, tester)
#                 results = isac_worker.parallel_compute(_common, _data)
#
#                 # Now Store Data for later retrieval
#                 _pi[_sub_per] = results.Phi
#                 _psi[_sub_per] = results.Psi
#                 _pred[_sub_per] = ISAC.estimate_map(results.Phi, results.Psi, _sub_set, np.arange(sZ))
#
#             storage['Learnt'] = {'Pi': _pi, 'Psi': _psi}
#             storage['Pred'] = _pred