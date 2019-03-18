"""
Script to Visualise the Results of Training DS/ISAR Models on the MRC Harwell Data (after Learn_Compare)
"""
from scipy.stats import ttest_rel
import numpy as np
import argparse

DEFAULTS = {'Results': ['../../data/Compare_DSS_002.npz', '../../data/Compare_DSA_002.npz', '../../data/Compare_ISAR_002.npz'],
            'Schemas': ['I', 'II', 'III', 'IV']}

if __name__ == '__main__':

    # ==== Parse Arguments: ==== #
    _arg_parse = argparse.ArgumentParser(description='Visualise the results of Training (MRC Harwell Data)')
    _arg_parse.add_argument('-r', '--results', help='List of (3) Result files for the DSS/DSA/ISAR models respectively '
                                                    '(separated by spaces). Defaults are: {}'.
                            format(DEFAULTS['Results']), default=DEFAULTS['Results'], nargs=3)
    _arg_parse.add_argument('-s', '--schemas', help='The Schema Names: Default is {}'.format(DEFAULTS['Schemas']),
                            default=DEFAULTS['Schemas'], nargs='*')
    args = _arg_parse.parse_args()

    # ==== Load the Data: ==== #
    with np.load(args.results[0]) as res_ds_s:
        accuracy_dss = res_ds_s['accuracy']
        log_loss_dss = res_ds_s['log_loss']
        sizes_all = res_ds_s['sizes']
    with np.load(args.results[1]) as res_ds_a:
        accuracy_dsa = res_ds_a['accuracy']
        log_loss_dsa = res_ds_a['log_loss']
    with np.load(args.results[2]) as res_isar:
        accuracy_isar = res_isar['accuracy']
        log_loss_isar = res_isar['log_loss']

    # ==== Now Print: ==== #
    np.set_printoptions(linewidth=120)
    print('            |    DS (Schema)   |     DS (All)    |       ISAR      | ------------------ T-Test -------------------')
    print('            |                  |                 |                 |       DSS vs ISAR      #      DSA vs ISAR')
    # ---- Accuracy First ---- #
    print('=================================================== Accuracy ====================================================')
    # [A] - Per-Schema Accuracies
    acc_dss_schema = np.divide(accuracy_dss, sizes_all)
    acc_dsa_schema = np.divide(accuracy_dsa, sizes_all)
    acc_isar_schema = np.divide(accuracy_isar, sizes_all)
    for s in range(len(args.schemas)):
        print(' Schema {:3} | {:.1f}% +/- {:5.2f}% | {:.1f}% +/- {:4.2f}% | {:.1f}% +/- {:.2f}% | t={:8.2f}, p={:.2e} #'
              ' t={:8.2f}, p={:.2e} '.format(args.schemas[s], acc_dss_schema[:, s].mean()*100,
                                             acc_dss_schema[:, s].std()*100, acc_dsa_schema[:, s].mean()*100,
                                             acc_dsa_schema[:, s].std()*100, acc_isar_schema[:, s].mean()*100,
                                             acc_isar_schema[:, s].std()*100,
                                             *ttest_rel(acc_dss_schema[:, s], acc_isar_schema[:, s]),
                                             *ttest_rel(acc_dsa_schema[:, s], acc_isar_schema[:, s])))
    # [B] - Now do global
    acc_dss_global = np.divide(accuracy_dss.sum(axis=1), sizes_all.sum(axis=1))
    acc_dsa_global = np.divide(accuracy_dsa.sum(axis=1), sizes_all.sum(axis=1))
    acc_isar_global = np.divide(accuracy_isar.sum(axis=1), sizes_all.sum(axis=1))
    print('-----------------------------------------------------------------------------------------------------------------')
    print(' Global     | {:.1f}% +/- {:.2f}%  | {:.1f}% +/- {:.2f}% | {:.1f}% +/- {:.2f}% | t={:8.2f}, p={:.2e} # '
          't={:8.2f}, p={:.2e} '.format(acc_dss_global.mean()*100, acc_dss_global.std()*100, acc_dsa_global.mean()*100,
                                        acc_dsa_global.std()*100, acc_isar_global.mean()*100, acc_isar_global.std()*100,
                                        *ttest_rel(acc_dss_global, acc_isar_global),
                                        *ttest_rel(acc_dsa_global, acc_isar_global)))
    print('-----------------------------------------------------------------------------------------------------------------')

    # ---- Next Log Loss ---- #
    print('=========================================== Predictive Log-Likelihood ===========================================')
    # [A] - Per-Schema PLL
    pll_dss_schema = np.divide(-log_loss_dss, sizes_all)
    pll_dsa_schema = np.divide(-log_loss_dsa, sizes_all)
    pll_isar_schema = np.divide(-log_loss_isar, sizes_all)
    for s in range(len(args.schemas)):
        print(' Schema {:3} | {:6.2f} +/- {:4.2f}  | {:6.2f} +/- {:4.2f} | {:.2f} +/- {:.2f}  | t={:8.2f}, p={:.2e} # '
              't={:8.2f}, p={:.2e}'.format(args.schemas[s], pll_dss_schema[:, s].mean(), pll_dss_schema[:, s].std(),
                                           pll_dsa_schema[:, s].mean(), pll_dsa_schema[:, s].std(),
                                           pll_isar_schema[:, s].mean(), pll_isar_schema[:, s].std(),
                                           *ttest_rel(pll_dss_schema[:, s], pll_isar_schema[:, s]),
                                           *ttest_rel(pll_dsa_schema[:, s], pll_isar_schema[:, s])))
    # [B] - Now do global
    pll_dss_global = np.divide(-log_loss_dss.sum(axis=1), sizes_all.sum(axis=1))
    pll_dsa_global = np.divide(-log_loss_dsa.sum(axis=1), sizes_all.sum(axis=1))
    pll_isar_global = np.divide(-log_loss_isar.sum(axis=1), sizes_all.sum(axis=1))
    print('-----------------------------------------------------------------------------------------------------------------')
    print(' Global     | {:.2f} +/- {:.2f}  |  {:.2f} +/- {:.2f} | {:5.2f} +/- {:4.2f}  | t={:8.2f}, p={:.2e} # '
          't={:8.2f}, p={:.2e}'.format(pll_dss_global.mean(), pll_dss_global.std(), pll_dsa_global.mean(),
                                       pll_dsa_global.std(), pll_isar_global.mean(), pll_isar_global.std(),
                                       *ttest_rel(pll_dss_global, pll_isar_global),
                                       *ttest_rel(pll_dsa_global, pll_isar_global)))
    print('-----------------------------------------------------------------------------------------------------------------')
