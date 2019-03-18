"""
Script to analyse the ISAR model in terms of its Entropy/Information Content. We do this by way of the Mutual
Information and Redundancy. Refer to the Paper for references.
"""

import numpy as np
import sys

sys.path.append('..')
from Tools import npext

# Some Defaults
SCHEMAS = ['I', 'II', 'III', 'IV']
['A{}'.format(k) for k in range(11)]

if __name__=='__main__':

    # ==== Load/Prepare the Data ==== #
    # ---- Load Parameters ---- #
    with np.load('../../data/model.mrc.npz') as _data:
        omega_mrc = _data['omega']
        pi = _data['pi']
        psi = _data['psi']
    # ---- Compute Sizes ---- #
    sZ, sK, sU = psi.shape
    sY = sU+1  # Contains NIS
    # ---- Construct Omega for One-v-Rest Schemas ---- #
    omega_ovr = np.zeros([sZ, sU, sY])
    for s in range(sZ):
        for u in range(sU):
            for y in range(sY):
                if ((y == u) and (u == s)) or ((y == sU) and (u != s)):
                    omega_ovr[s, u, y] = 1.0

    # ==== Generate the Mutual Information ==== #
    # ---- Y = U: There is consistent labelling with the entire label-set ---- #
    Im_Same = np.zeros(sK)
    for k in range(sK):
        Im_Same[k] = npext.mutual_information(pi, psi[:, k, :])
    # ---- Single Schema ---- #


    # ==== Finally Print Results ==== #
    print('          |      I(Z;Y)      |      R(Z;Y)')
    print('          | Mean  Min.  Max. | Mean  Min.  Max.')
    print('-------------------------------------------------')
    print(' Y = U    | {:4.2f}  {:4.2f}  {:4.2f} |'.format(Im_Same.mean(), Im_Same.min(), Im_Same.max()))
    print('-------------------------------------------------')
    print(' {:3s}        | {:4.2f}  {:4.2f}  {:4.2f} |')
    print(' I+III+IV |')




