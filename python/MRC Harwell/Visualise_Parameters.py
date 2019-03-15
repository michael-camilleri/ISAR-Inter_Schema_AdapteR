"""
Script to Visualise the Results of Parameter Learning using ISAR (in conjunction with Learn_Parameters)
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse

DEFAULTS = {'Results': '../../data/Parameters_ISAR.npz'}


if __name__ == '__main__':

    # ==== Parse Arguments: ==== #
    _arg_parse = argparse.ArgumentParser(description='Visualise the results of Training for Parameter Recover (MRC '
                                                     'Harwell Data)')
    _arg_parse.add_argument('-r', '--results', help='Result file. Default is: {}'.format(DEFAULTS['Results']),
                            default=DEFAULTS['Results'])
    _arg_parse.add_argument('-i', '--independent', help='If set, then compute the RAD independently per annotator.',
                            action='store_true')
    _arg_parse.set_defaults(independent=False)
    args = _arg_parse.parse_args()

    # ==== Load the Data: ==== #
    with np.load(args.results) as res_file:
        Pi_true = res_file['Pi_true']
        Pi_full = res_file['Pi_full']
        Pi_isar = res_file['Pi_isar']
        Psi_true = res_file['Psi_true']
        Psi_full = res_file['Psi_full']
        Psi_isar = res_file['Psi_isar']
        Data_sizes = res_file['Sizes']

    # ==== Compute RAD Values: ==== #
    sR, sI, sK = np.asarray(Psi_full.shape)[[0,1,3]]
    rad_full_pi = np.empty([sR, sI])
    rad_full_psi = np.empty([sR, sI, sK]) if args.independent else np.empty([sR, sI])
    rad_isar_pi = np.empty([sR, sI])
    rad_isar_psi = np.empty([sR, sI, sK]) if args.independent else np.empty([sR, sI])
    for run in range(Pi_full.shape[0]):
        for inc in range(Pi_full.shape[1]):
            # Compute first for Pi
            rad_full_pi[run, inc] = (np.abs(Pi_true[run] - Pi_full[run, inc]) * 100.0 / np.mean(Pi_true[run])).mean()
            rad_isar_pi[run, inc] = (np.abs(Pi_true[run] - Pi_isar[run, inc]) * 100.0 / np.mean(Pi_true[run])).mean()
            # Now Compute for Psi
            if args.independent:
                rad_full_psi[run, inc, :] = (np.abs(Psi_true[run] - Psi_full[run, inc]) * 100.0 /
                                             np.mean(Psi_true[run], axis=[0, 2], keepdims=True)).mean(axis=[0, 2])
                rad_isar_psi[run, inc, :] = (np.abs(Psi_true[run] - Psi_isar[run, inc]) * 100.0 /
                                             np.mean(Psi_true[run], axis=[0, 2], keepdims=True)).mean(axis=[0, 2])
            else:
                rad_full_psi[run, inc] = (np.abs(Psi_true[run] - Psi_full[run, inc]) * 100.0 /
                                          np.mean(Psi_true[run])).mean()
                rad_isar_psi[run, inc] = (np.abs(Psi_true[run] - Psi_isar[run, inc]) * 100.0 /
                                          np.mean(Psi_true[run])).mean()

    # ==== Now Plot ==== #
    plt.figure()
    plt.errorbar(Data_sizes.mean(axis=0), np.mean(rad_isar_pi, axis=0), yerr=np.std(rad_isar_pi, axis=0),
                 label='$\Pi$ (ISAR)', linewidth=2.0, color='b', linestyle='-')
    plt.errorbar(Data_sizes.mean(axis=0), np.mean(rad_full_pi, axis=0), yerr=np.std(rad_full_pi, axis=0),
                 label='$\Pi$ (Full)', linewidth=2.0, color='b', linestyle='--')
    plt.errorbar(Data_sizes.mean(axis=0), np.mean(rad_isar_psi, axis=0), yerr=np.std(rad_isar_psi, axis=0),
                 label='$\Psi$ (ISAR)', linewidth=2.0, color='orange', linestyle='-')
    plt.errorbar(Data_sizes.mean(axis=0), np.mean(rad_full_psi, axis=0), yerr=np.std(rad_full_psi, axis=0),
                 label='$\Psi$ (Full)', linewidth=2.0, color='orange', linestyle='--')
    plt.xscale('log')
    plt.show()



#
# if __name__ == '__main__':
#
#     pi_rad = np.zeros([len(RUNS), len(PERCENTAGES)])
#     # kl_pi = np.zeros([len(RUNS), len(PERCENTAGES)])
#     psi_rad = np.zeros([len(RUNS), len(PERCENTAGES)])
#     psi_rad_k = np.zeros([sK, len(RUNS), len(PERCENTAGES)])
#     # kl_psi = np.zeros([len(RUNS), len(PERCENTAGES), sK])
#
#     # Iterate over Runs
#     for r_idx, run in enumerate(RUNS):
#         with shelve.open(os.path.join(Const['Results.Scratch'], DUMP_NAME.format(run)), 'c') as input:
#             Pi = input['True']['Pi']
#             Psi = input['True']['Psi']
#             # Iterate over percentages
#             for p_idx, percent in enumerate(PERCENTAGES):
#                 # PI
#                 _Pi = input['Learnt']['Pi'][percent]
#                 pi_rad[r_idx, p_idx] = (np.abs(Pi - _Pi) * 100.0 / np.mean(Pi)).mean()
#                 # kl_pi[r_idx, p_idx] = entropy(_Pi, Pi)
#                 # PSI
#                 _Psi = input['Learnt']['Psi'][percent]
#                 psi_rad[r_idx, p_idx] = (np.abs(Psi - _Psi) * 100.0 / np.mean(Psi)).mean()
#                 for k in range(sK):
#                     psi_rad_k[k, r_idx, p_idx] = (np.abs(Psi[:,k,:] - _Psi[:,k,:]) * 100.0 / np.mean(Psi[:,k,:])).mean()
#                 # _kl_psi = np.zeros([sZ, sK])
#                 # for z in range(sZ):
#                 #     for k in range(sK):
#                 #         _kl_psi[z, k] = entropy(_Psi[z, k, :], Psi[z, k, :])
#                 # kl_psi[r_idx, p_idx, :] = np.average(_kl_psi, axis=0, weights=Pi)
#
#     # Now plot
#     plt.figure()
#     plt.errorbar(PERCENTAGES * sNT, np.mean(pi_rad, axis=0), yerr=np.std(pi_rad, axis=0), label='$\Pi$', linewidth=2.0)
#     plt.errorbar(PERCENTAGES * sNT, np.mean(psi_rad, axis=0), yerr=np.std(psi_rad, axis=0), label='$\Psi$', linewidth=2.0)
#     plt.gca().tick_params(labelsize=20)
#     plt.xlabel('Data-Set Size', fontsize=20)
#     plt.ylabel('% RAD', fontsize=20)
#     plt.legend(fontsize=20)
#     plt.figure()
#     for k in range(sK):
#         if k > sK*2/3:
#             plt.errorbar(PERCENTAGES * sNT, np.mean(psi_rad_k[k], axis=0), yerr=np.std(psi_rad_k[k], axis=0),
#                          label='{0} ({1:.0f}%)'.format(k, pAnnot[k]), linewidth=2.0, linestyle='dashed')
#         else:
#             plt.errorbar(PERCENTAGES * sNT, np.mean(psi_rad_k[k], axis=0), yerr=np.std(psi_rad_k[k], axis=0),
#                          label='{0} ({1:.0f}%)'.format(k, pAnnot[k]), linewidth=2.0)
#     plt.gca().tick_params(labelsize=20)
#     plt.xlabel('Data-Set Size', fontsize=20)
#     plt.ylabel('% RAD', fontsize=20)
#     plt.legend(fontsize=20)
#
#
#     # plt.figure()
#     # plt.errorbar(PERCENTAGES * sNT, np.mean(kl_pi, axis=0), yerr=np.std(kl_pi, axis=0), label='$\Pi$', linewidth=2.0)
#     # plt.errorbar(PERCENTAGES * sNT, np.mean(kl_psi, axis=(0, 2)), yerr=np.std(kl_psi, axis=(0, 2)), label='$\Psi$', linewidth=2.0)
#     # plt.gca().tick_params(labelsize=17)
#     # plt.xlabel('Data-Set Size', fontsize=17)
#     # plt.ylabel('KL-Divergence', fontsize=17)
#     # plt.legend(fontsize=17)
#     plt.show()

