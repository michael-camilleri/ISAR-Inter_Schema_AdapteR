"""
Script to Visualise the Results of Training DS/ISAR Models on the MRC Harwell Data (after Learn_Compare)
"""
from scipy.stats import ttest_rel
import numpy as np
import argparse

DEFAULTS = {'Results': ['../../data/Compare_DS_004.npz', '../../data/Compare_ISAR_004.npz'],
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
        log_loss_ds = res_ds['log_loss']
        sizes_all = res_ds['sizes']
    with np.load(args.results[1]) as res_isar:
        accuracy_isar = res_isar['accuracy']
        log_loss_isar = res_isar['log_loss']

    # ==== Now Print: ==== #
    np.set_printoptions(linewidth=120)
    print('            |     DS (All)    |       ISAR      | ------- T-Test ------')
    print('            |                 |                 |       DS vs ISAR')
    # ---- Accuracy First ---- #
    print('=============================== Accuracy ================================')
    # [A] - Per-Schema Accuracies
    acc_ds_schema = np.divide(accuracy_ds, sizes_all)
    acc_isar_schema = np.divide(accuracy_isar, sizes_all)
    for s in range(len(args.schemas)):
        print(' Schema {:3} | {:.1f}% +/- {:4.2f}% | {:.1f}% +/- {:.2f}% | t={:8.2f}, p={:.2e}'
              .format(args.schemas[s], acc_ds_schema[:, s].mean()*100, acc_ds_schema[:, s].std()*100,
                      acc_isar_schema[:, s].mean()*100, acc_isar_schema[:, s].std()*100,
                      *ttest_rel(acc_ds_schema[:, s], acc_isar_schema[:, s])))
    # [B] - Now do global
    acc_ds_global = np.divide(accuracy_ds.sum(axis=1), sizes_all.sum(axis=1))
    acc_isar_global = np.divide(accuracy_isar.sum(axis=1), sizes_all.sum(axis=1))
    print('--------------------------------------------------------------------------')
    print(' Global     | {:.1f}% +/- {:.2f}% | {:.1f}% +/- {:.2f}% | t={:8.2f}, p={:.2e}'
          .format(acc_ds_global.mean()*100, acc_ds_global.std()*100, acc_isar_global.mean()*100,
                  acc_isar_global.std()*100, *ttest_rel(acc_ds_global, acc_isar_global)))
    print('--------------------------------------------------------------------------')

    # ---- Next Log Loss ---- #
    print('======================= Predictive Log-Likelihood ========================')
    # [A] - Per-Schema PLL
    pll_ds_schema = np.divide(-log_loss_ds, sizes_all)
    pll_isar_schema = np.divide(-log_loss_isar, sizes_all)
    for s in range(len(args.schemas)):
        print(' Schema {:3} | {:6.2f} +/- {:4.2f} | {:.2f} +/- {:.2f}  | t={:8.2f}, p={:.2e}'
              .format(args.schemas[s], pll_ds_schema[:, s].mean(), pll_ds_schema[:, s].std(),
                      pll_isar_schema[:, s].mean(), pll_isar_schema[:, s].std(),
                      *ttest_rel(pll_ds_schema[:, s], pll_isar_schema[:, s])))
    # [B] - Now do global
    pll_ds_global = np.divide(-log_loss_ds.sum(axis=1), sizes_all.sum(axis=1))
    pll_isar_global = np.divide(-log_loss_isar.sum(axis=1), sizes_all.sum(axis=1))
    print('--------------------------------------------------------------------------')
    print(' Global     |  {:.2f} +/- {:.2f} | {:5.2f} +/- {:4.2f}  | t={:8.2f}, p={:.2e} '
          .format(pll_ds_global.mean(), pll_ds_global.std(), pll_isar_global.mean(), pll_isar_global.std(),
                  *ttest_rel(pll_ds_global, pll_isar_global)))
    print('--------------------------------------------------------------------------')
