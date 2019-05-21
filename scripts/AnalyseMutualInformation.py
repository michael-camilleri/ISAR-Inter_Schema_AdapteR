"""
Script to analyse the ISAR model in terms of its Entropy/Information Content. We do this by way of the Mutual
Information and Redundancy. Refer to the Paper for references.
"""
from mpctools.extensions import npext
from itertools import combinations
from scipy.special import comb
import numpy as np

# Some Defaults
SCHEMAS = ['I', 'II', 'III', 'IV']
['A{}'.format(k) for k in range(11)]

if __name__=='__main__':

    # ==== Load/Prepare the Data ==== #
    # ---- Load Parameters ---- #
    with np.load('../../data/model.mrc.npz') as _data:
        omega = _data['omega']
        pi = _data['pi']
        psi = _data['psi']
    # ---- Compute Sizes ---- #
    sZ, sK, sU = psi.shape
    sS, _, sY = omega.shape
    # ---- Finally Construct Conditional Emission P(Y|Z) (i.e. summing out U)
    pY_Z = []                   # We do it as a list to indicate different in schemas...
    for s in range(sS):
        pY_Z.append(np.empty([sZ, sK, sY]))
        for k in range(sK):
            pY_Z[-1][:, k, :] = np.matmul(psi[:, k, :], omega[s, :, :])

    # ==== Generate the Mutual Information/Redundancy ==== #
    # ---- Y = U: There is consistent labelling with the entire label-set ---- #
    Im_Same = np.zeros(sK)
    for k in range(sK):
        Im_Same[k] = npext.mutual_information(pi, psi[:, k, :])
    # ---- Single Schema ---- #
    Im_Single = np.zeros([sS, sK])
    for s, py_z in enumerate(pY_Z):
        for k in range(sK):
            Im_Single[s, k] = npext.mutual_information(pi, py_z[:, k, :])
    # ---- Pairwise Combinations ---- #
    Im_Pair = np.empty([int(comb(sS, 2)), sK])
    Red_Pair = np.empty([int(comb(sS, 2)), sK])
    for gi, ((i1, p1), (i2, p2)) in enumerate(combinations(enumerate(pY_Z), 2)):
        for k in range(sK):
            Im_Pair[gi, k] = npext.mutual_information(pi, [p1[:, k, :], p2[:, k, :]])
            Red_Pair[gi, k] = (Im_Single[i1, k] + Im_Single[i2, k]) - Im_Pair[gi, k]
    # ---- Triplets ---- #
    Im_Triple = np.empty([int(comb(sS, 3)), sK])
    Red_Triple = np.empty([int(comb(sS, 3)), sK])
    for gi, ((i1, p1), (i2, p2), (i3, p3)) in enumerate(combinations(enumerate(pY_Z), 3)):
        for k in range(sK):
            Im_Triple[gi, k] = npext.mutual_information(pi, [p1[:, k, :], p2[:, k, :], p3[:, k, :]])
            Red_Triple[gi, k] = (Im_Single[i1, k] + Im_Single[i2, k] + Im_Single[i3, k]) - Im_Triple[gi, k]
    # ---- All-Four Schemas ---- #
    Im_Quad = np.empty(sK)
    Red_Quad = np.empty(sK)
    for k in range(sK):
        Im_Quad[k] = npext.mutual_information(pi, [py_z[:, k, :] for py_z in pY_Z])
        Red_Quad[k] = Im_Single[:, k].sum() - Im_Quad[k]


    # ==== Finally Print Results ==== #
    print('           |      I(Z;Y)      |      R(Z;Y)')
    print('           | Mean  Min.  Max. | Mean  Min.  Max.')
    print('-------------------------------------------------')
    print(' Y = U     | {:4.2f}  {:4.2f}  {:4.2f} |'.format(Im_Same.mean(), Im_Same.min(), Im_Same.max()))
    print('-------------------------------------------------')
    for s in range(sS):
        print(' {:3s}       | {:4.2f}  {:4.2f}  {:4.2f} |'.format(SCHEMAS[s], Im_Single[s, :].mean(),
                                                                   Im_Single[s, :].min(), Im_Single[s, :].max()))
    print('-------------------------------------------------')
    for si, (s1, s2) in enumerate(combinations(SCHEMAS, 2)):
        print(' {:6s}    | {:4.2f}  {:4.2f}  {:4.2f} | {:4.2f}  {:4.2f}  {:4.2f}'
              .format('{}+{}'.format(s1, s2), Im_Pair[si, :].mean(), Im_Pair[si, :].min(), Im_Pair[si, :].max(),
                      Red_Pair[si, :].mean(), Red_Pair[si, :].min(), Red_Pair[si, :].max()))
    print('-------------------------------------------------')
    for si, (s1, s2, s3) in enumerate(combinations(SCHEMAS, 3)):
        print(' {:9s} | {:4.2f}  {:4.2f}  {:4.2f} | {:4.2f}  {:4.2f}  {:4.2f}'
              .format('{}+{}+{}'.format(s1, s2, s3), Im_Triple[si, :].mean(), Im_Triple[si, :].min(),
                      Im_Triple[si, :].max(), Red_Triple[si, :].mean(), Red_Triple[si, :].min(),
                      Red_Triple[si, :].max()))
    print('-------------------------------------------------')
    print(' ALL       | {:4.2f}  {:4.2f}  {:4.2f} | {:4.2f}  {:4.2f}  {:4.2f}'
          .format(Im_Quad.mean(), Im_Quad.min(), Im_Quad.max(), Red_Quad.mean(), Red_Quad.min(), Red_Quad.max()))
    print('-------------------------------------------------')
