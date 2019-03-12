##########################################################################
## Script to compare, in simulation, the performance of the ISAC model  ##
## to plain old NAM, under some Extreme Conditions.                     ##
##########################################################################

from pandas.api.types import CategoricalDtype as CDType
from sklearn.metrics import f1_score
from itertools import cycle
import pandas as pd
import numpy as np
import argparse
import sys

# Load own packages
sys.path.append('..')
from Models import AnnotDS, AnnotISAR





# Configuration Parameters
sN = 60   # Number of Segments to generate
sZ = 13   # Number of Latent States
sT = 5400 # Number of samples per segment
sU = 13   # Number of Annotator States
sX = 14   # Includes NIS
nA = 3    # Number of Annotators in any one sample
sK = 11   # Number of Annotators
sS = 4    # Number of Schemas

pAnnot = npext.sum_to_one([49, 31, 11, 4, 10, 25, 9, 7, 6, 3, 3])  # Probability over Annotators
pSchema = npext.sum_to_one([13, 15, 17, 10])                       # Probability over Schemas
Xi = np.load(os.path.join(Const['Data.Clean'], 'XI.npy'))
cdtypes = (CDType(categories=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),  # I
           CDType(categories=[0, 1, 5, 6, 7, 10, 12]),                  # II
           CDType(categories=[0, 1, 4, 5, 6, 7, 8, 9, 10, 11]),         # III
           CDType(categories=[0, 1, 2, 5, 6, 7, 10, 11]))               # IV


NIS_idx = sZ  # NIS Index

# Load the Parameters from the real-data
_Pi = np.load(os.path.join(Const['Data.Clean'], 'PHI.npy'))
_Psi = np.load(os.path.join(Const['Data.Clean'], 'PSI.npy'))


class Tester:

    def __init__(self):
        self.Iterations = 0
        self.PrevIter = 0

    def _register_worker(self, id):
        """
        To conform to the Handler Interface
        """
        pass

    @property
    def Queue(self):
        """
        To conform to the Handler Interface
        """
        return self

    def put(self, data, block=False):
        """
        Print Progress
        """
        if data[1] % 5 == 0:
            print('Iteration: {0}'.format(data[1]))
        self.Iterations = self.PrevIter
        self.PrevIter = data[1]

    def reset(self):
        self.Iterations = 0
        self.PrevIter = 0
        return self


if __name__ == '__main__':

    r_start = 700
    r_len = 20

    # Fold Iterator
    iterator = cycle(range(sF))

    for run in range(r_start, r_start + r_len):
        print('Simulating Run ', run)

        Pi = npext.sum_to_one(_Pi[:sZ])
        Psi = npext.sum_to_one(_Psi[:sZ, :sK, :sU], axis=-1)

        # ================= GENERATE DATA ================= #
        # Data
        Z = np.random.choice(sZ, size=[sN*sT], p=Pi)                                    # Latent State
        S = np.repeat(np.random.choice(sS, size=sN, p=pSchema), sT)                     # Schema
        F = np.repeat([next(iterator) for _ in range(sN)], sT)                          # Folds
        G = np.repeat(np.arange(sN), sT)                                                # Segments
        X = np.full(shape=[sN * sT, sK], fill_value=np.NaN)                             # Observed

        # Now Generate Observations (have to do on a per-sample basis)
        for n in range(sN):
            # Decide on Annotators
            A = np.random.choice(sK, size=nA, replace=False, p=pAnnot)

            # Iterate over Time
            for t in range(sT):
                # Time Instant
                _t_i = n*sT + t

                # Iterate over Annotators chosen
                for k in A:
                    # Check Schema for this annotator
                    s = S[_t_i]

                    # Compute Annotator Emission
                    u_k = np.random.choice(sU, p=Psi[Z[_t_i], k, :])

                    # Compute Observation (if in schema)
                    if Xi[s, u_k, u_k] == 1:
                        nis = False
                        X[_t_i, k] = u_k

        # Prepare a priori Message for ISAC:
        _labels_hot = pd.get_dummies(pd.DataFrame(X).astype(CDType(categories=np.arange(sX)))).values
        _schema_hot = pd.get_dummies(pd.Series(S).astype(CDType(categories=np.arange(sS)))).values
        M_Phi = ISAC.msg_xi(Xi, _labels_hot, _schema_hot)

        # Keep Track of Parameters
        with shelve.open(os.path.join(Const['Results.Scratch'], DUMP_NAME.format(run)), 'n') as storage:
            storage['Dims'] = [sZ, sK, sU, sX, sS]
            storage['Z'] = Z
            storage['S'] = S
            storage['F'] = F
            storage['G'] = G
            storage['X'] = X
            storage['True'] = {'Pi': Pi, 'Psi': Psi}

            # ================= TRAIN MODELS ================= #
            print('Training Run {} using {}-fold CV'.format(run, sF))

            # Prepare Placeholders for Predictions: 0 is for ISAC, 1 is for NAM
            _pred = np.empty([sN*sT, 2], dtype=bool)    # Correct/Incorrect Predictions
            _wght = np.empty([sN*sT, 2], dtype=float)   # Weight given to correct prediction
            _f1 = np.empty([sS, 2], dtype=float)        # F1-Score
            tester = Tester()

            # Iterate over folds
            for fold in range(sF):
                print(' ---> Fold {}'.format(fold))

                # Split Training/Testing Sets
                train_idcs = np.not_equal(F, fold)
                valid_idcs = np.equal(F, fold)

                # ----------- Now Train ISAC Model -----------
                print(' ========> ISAC')
                # Prepare Data
                M_Phi_t = M_Phi[train_idcs, :]
                M_Phi_v = M_Phi[valid_idcs, :]

                # Train Model
                _common = ISAC.EMWorker.ComputeCommon_t(max_iter=200, tolerance=1e-4, update_rate=3, m_xi=M_Phi_t,
                                                        prior_phi=np.ones(sZ), prior_psi=np.ones([sZ, sK, sU]))
                _data = [npext.sum_to_one(np.ones(sZ)),
                         np.stack(npext.sum_to_one(np.eye(sZ, sZ) + np.full([sZ, sZ], fill_value=0.01), axis=1) for _ in range(sK)).swapaxes(0, 1)]
                results = ISAC.EMWorker(0, tester.reset()).parallel_compute(_common, _data)

                # Now Validate Model
                _map = ISAC.estimate_map(results.Phi, results.Psi, M_Phi_v, None)
                _pred[valid_idcs, 0] = np.equal(np.argmax(_map, axis=1), Z[valid_idcs])
                _wght[valid_idcs, 0] = _map[np.arange(len(_map)), Z[valid_idcs]]
                for s in range(sS):
                    schema_idcs = np.equal(S, s)
                    _s_valid = np.logical_and(valid_idcs, schema_idcs)
                    _f1[s, 0] = f1_score(Z[_s_valid], np.argmax(_map, axis=1)[schema_idcs[valid_idcs]], labels=np.arange(sZ),
                                         average='weighted')

                # ----------- Now Train NAM Model -----------
                print(' ========> NAM')
                # We have to do this per-schema
                for s in range(sS):
                    # Prepare Subset of Data
                    schema_idcs = np.equal(S, s)
                    _s_train = np.logical_and(train_idcs, schema_idcs)
                    _s_valid = np.logical_and(valid_idcs, schema_idcs)
                    sL = len(cdtypes[s].categories)
                    label_set = cdtypes[s].categories.values

                    # Now Extract Data
                    _XHot_train = pd.get_dummies(pd.DataFrame(X[_s_train]).astype(cdtypes[s])).values
                    _XHot_valid = pd.get_dummies(pd.DataFrame(X[_s_valid]).astype(cdtypes[s])).values

                    # Train Model
                    _common = (200, 1e-4, 3, [len(_XHot_train), sL, sK, sL], _XHot_train,  [np.ones(sL), np.ones([sL, sK*sL])], False)
                    _data = [npext.sum_to_one(np.ones(sL)),
                             np.tile(npext.sum_to_one(np.eye(sL, sL) + np.full([sL, sL], fill_value=0.01), axis=1), sK)]
                    results = NAM.EMWorker(0, tester.reset()).parallel_compute(_common, _data)

                    # Validate Model
                    _map = NAM.estimate_map(results[0], results[1], _XHot_valid, label_set, 13)
                    _pred[_s_valid, 1] = np.equal(np.argmax(_map, axis=1), Z[_s_valid])
                    _wght[_s_valid, 1] = _map[np.arange(len(_map)), Z[_s_valid]]
                    _f1[s, 1] = f1_score(Z[_s_valid], np.argmax(_map, axis=1), labels=np.arange(sZ), average='weighted')

            # Store Result
            storage['Correctness'] = _pred
            storage['Belief'] = _wght
            storage['F1'] = _f1