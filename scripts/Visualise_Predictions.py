"""
Script to Visualise the Results of Training DS/ISAR models on the MRC Harwell Data (after Learn_Compare)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
http://www.gnu.org/licenses/.

Author: Michael P. J. Camilleri
"""
from scipy.stats import ttest_rel
import numpy as np
import argparse

DEFAULTS = {'Results': ['../data/Compare_DS.npz', '../data/Compare_ISAR.npz'],
            'Schemas': ['I', 'II', 'III', 'IV']}

if __name__ == '__main__':

    # ==== Parse Arguments: ==== #
    _arg_parse = argparse.ArgumentParser(description='Visualise the results of Training (MRC Harwell Data)')
    _arg_parse.add_argument('-r', '--results', help='List of (2) Result files for the DS/ISAR models respectively '
                                                    '(separated by spaces). Defaults are: {}'.
                            format(DEFAULTS['Results']), default=DEFAULTS['Results'], nargs=2)
    _arg_parse.add_argument('-s', '--schemas', help='The Schema Names: Default is {}'.format(DEFAULTS['Schemas']),
                            default=DEFAULTS['Schemas'], nargs='*')
    args = _arg_parse.parse_args()

    # ==== Load the Data: ==== #
    with np.load(args.results[0]) as res_ds:
        accuracy_ds = res_ds['accuracy']
        log_loss_ds = -res_ds['log_loss']
        f1_score_ds = res_ds['f1']
    with np.load(args.results[1]) as res_isar:
        accuracy_isar = res_isar['accuracy']
        log_loss_isar = -res_isar['log_loss']
        f1_score_isar = res_isar['f1']

    # ==== Now Print: ==== #
    np.set_printoptions(linewidth=120)
    print('            |     DS (All)    |       ISAR      | ------- T-Test ------')
    print('            |                 |                 |       DS vs ISAR')
    # ---- Accuracy First ---- #
    print('=============================== Accuracy ================================')
    # [A] - Per-Schema Accuracies
    for s in range(len(args.schemas)):
        print(' Schema {:3} | {:.1f}% +/- {:4.2f}% | {:.1f}% +/- {:.2f}% | t={:8.2f}, p={:.2e}'
              .format(args.schemas[s], accuracy_ds[:, s].mean()*100, accuracy_ds[:, s].std()*100,
                      accuracy_isar[:, s].mean()*100, accuracy_isar[:, s].std()*100,
                      *ttest_rel(accuracy_ds[:, s], accuracy_isar[:, s])))
    # [B] - Now do global
    print('--------------------------------------------------------------------------')
    print(' Global     | {:.1f}% +/- {:.2f}% | {:.1f}% +/- {:.2f}% | t={:8.2f}, p={:.2e}'
          .format(accuracy_ds[:, -1].mean()*100, accuracy_ds[:, -1].std()*100, accuracy_isar[:, -1].mean()*100,
                  accuracy_isar[:, -1].std()*100, *ttest_rel(accuracy_ds[:, -1], accuracy_isar[:, -1])))
    print('--------------------------------------------------------------------------')

    # ---- Next Log Loss ---- #
    print('======================= Predictive Log-Likelihood ========================')
    # [A] - Per-Schema PLL
    for s in range(len(args.schemas)):
        print(' Schema {:3} | {:6.2f} +/- {:4.2f} | {:.2f} +/- {:.2f}  | t={:8.2f}, p={:.2e}'
              .format(args.schemas[s], log_loss_ds[:, s].mean(), log_loss_ds[:, s].std(),
                      log_loss_isar[:, s].mean(), log_loss_isar[:, s].std(),
                      *ttest_rel(log_loss_ds[:, s], log_loss_isar[:, s])))
    # [B] - Now do global
    print('--------------------------------------------------------------------------')
    print(' Global     |  {:.2f} +/- {:.2f} | {:5.2f} +/- {:4.2f}  | t={:8.2f}, p={:.2e} '
          .format(log_loss_ds[:, -1].mean(), log_loss_ds[:, -1].std(), log_loss_isar[:, -1].mean(),
                  log_loss_isar[:, -1].std(), *ttest_rel(log_loss_ds[:, -1], log_loss_isar[:, -1])))
    print('--------------------------------------------------------------------------')

    # ---- Finally F1-Score ---- #
    print('=============================== F1-Score ================================')
    # [A] - Per-Schema F1
    for s in range(len(args.schemas)):
        print(' Schema {:3} | {:.3f} +/- {:4.3f} | {:.3f} +/- {:.3f} | t={:8.2f}, p={:.2e}'
              .format(args.schemas[s], f1_score_ds[:, s].mean(), f1_score_ds[:, s].std(), f1_score_isar[:, s].mean(),
                      f1_score_isar[:, s].std(), *ttest_rel(f1_score_ds[:, s], f1_score_isar[:, s])))
    # [B] - Now do global
    print('--------------------------------------------------------------------------')
    print(' Global     | {:.3f} +/- {:.3f} | {:.3f} +/- {:.3f} | t={:8.2f}, p={:.2e}'
          .format(f1_score_ds[:, -1].mean(), f1_score_ds[:, -1].std(), f1_score_isar[:, -1].mean(),
                  f1_score_isar[:, -1].std(), *ttest_rel(f1_score_ds[:, -1], f1_score_isar[:, -1])))
    print('--------------------------------------------------------------------------')