"""
Script to Visualise the Results of Parameter Learning using ISAR (in conjunction with Learn_Parameters)
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse

DEFAULTS = {'Results': '../../data/Parameters_ISAR.npz', 'Independent': False}


if __name__ == '__main__':

    # ==== Parse Arguments: ==== #
    _arg_parse = argparse.ArgumentParser(description='Visualise the results of Training for Parameter Recover (MRC '
                                                     'Harwell Data)')
    _arg_parse.add_argument('-r', '--results', help='Result file. Default is: {}'.format(DEFAULTS['Results']),
                            default=DEFAULTS['Results'])
    _arg_parse.add_argument('-i', '--independent', help='If set, then compute the RAD independently per annotator.',
                            action='store_true')
    _arg_parse.set_defaults(independent=DEFAULTS['Independent'])
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
    sR, sI, sK = np.asarray(Psi_full.shape)[[0, 1, 3]]
    rad_full_pi = np.empty([sR, sI])
    rad_full_psi = np.empty([sR, sI])
    rad_isar_pi = np.empty([sR, sI])
    rad_isar_psi = np.empty([sR, sI])
    rad_isar_psi_k = np.empty([sR, sI, sK]) if args.independent else None
    for run in range(Pi_full.shape[0]):
        for inc in range(Pi_full.shape[1]):
            # Compute first for Pi
            rad_full_pi[run, inc] = (np.abs(Pi_true[run] - Pi_full[run, inc])*100.0 / np.mean(Pi_true[run])).mean()
            rad_isar_pi[run, inc] = (np.abs(Pi_true[run] - Pi_isar[run, inc])*100.0 / np.mean(Pi_true[run])).mean()
            # Compute now for Psi
            rad_full_psi[run, inc] = (np.abs(Psi_true[run] - Psi_full[run, inc])*100.0 / np.mean(Psi_true[run])).mean()
            rad_isar_psi[run, inc] = (np.abs(Psi_true[run] - Psi_isar[run, inc])*100.0 / np.mean(Psi_true[run])).mean()
            # Now Compute for Psi independently if needed
            if args.independent:
                rad_isar_psi_k[run, inc, :] = (np.abs(Psi_true[run] - Psi_isar[run, inc]) * 100.0 /
                                               np.mean(Psi_true[run], axis=(0, 2), keepdims=True)).mean(axis=(0, 2))

    # ==== Now Plot ==== #
    # --- Plot All Together --- #
    plt.figure()
    # [A] - Plot
    plt.errorbar(Data_sizes.mean(axis=0), np.mean(rad_isar_pi, axis=0), yerr=np.std(rad_isar_pi, axis=0),
                 label='$\Pi$ (ISAR)', linewidth=2.0, color='b', linestyle='-')
    plt.errorbar(Data_sizes.mean(axis=0), np.mean(rad_full_pi, axis=0), yerr=np.std(rad_full_pi, axis=0),
                 label='$\Pi$ (Full)', linewidth=2.0, color='b', linestyle='--')
    plt.errorbar(Data_sizes.mean(axis=0), np.mean(rad_isar_psi, axis=0), yerr=np.std(rad_isar_psi, axis=0),
                 label='$\Psi$ (ISAR)', linewidth=2.0, color='orange', linestyle='-')
    plt.errorbar(Data_sizes.mean(axis=0), np.mean(rad_full_psi, axis=0), yerr=np.std(rad_full_psi, axis=0),
                 label='$\Psi$ (Full)', linewidth=2.0, color='orange', linestyle='--')
    # [B] - Improve Plot
    plt.legend(fontsize=20)
    plt.xlabel('Number of samples used for training', fontsize=20)
    plt.ylabel('RAD', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True, which='major', axis='both')
    plt.xscale('log')
    # --- Will Plot Individually If requested --- #
    if args.independent:
        plt.figure()
        # [A] - Plot per Annotator
        for k in range(sK):
            plt.errorbar(Data_sizes.mean(axis=0), np.mean(rad_isar_psi_k[:, :, k], axis=0),
                         yerr=np.std(rad_isar_psi_k[:, :, k], axis=0), label='$\Psi$ (A {})'.format(k), linewidth=2.0)
        # [B] - Improve Plot
        plt.legend(fontsize=20)
        plt.xlabel('Number of samples used for training', fontsize=20)
        plt.ylabel('RAD', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.grid(True, which='major', axis='both')
        plt.xscale('log')
    plt.show()
