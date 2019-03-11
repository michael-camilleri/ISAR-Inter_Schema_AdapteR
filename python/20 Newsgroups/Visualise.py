"""
Script to Visualise the results of training
"""
# Load Python Specific Libraries
from matplotlib import pyplot as plt
import numpy as np
import argparse

DEFAULTS = {'Results': ['Results_02.npz'],  # List of the result files
            'Names': ['2.0%']}              # Names to accompany Result Files

if __name__=='__main__':

    # ==== Parse Arguments: ==== #
    _arg_parse = argparse.ArgumentParser(description='Visualise the results of Training')
    _arg_parse.add_argument('-r', '--results', help='List of Result files (separated by spaces). Defaults are: {}'.
                            format(DEFAULTS['Results']), default=DEFAULTS['Results'], nargs='*')
    _arg_parse.add_argument('-n', '--names', help='Names to accompany Result files: must be same number of elements as '
                                                  'the [results] list. Defaults to: {}'.format(DEFAULTS['Names']),
                            default=DEFAULTS['Names'], nargs='*')
    args = _arg_parse.parse_args()
    assert len(args.results) == len(args.names), 'Names must be of same length as Results'

    # ==== Load Files and Visualise Plots ==== #
    # --- Iterate through Experiments --- #
    for exp_name, exp_file in zip(args.names, args.results):
        _data = np.load(exp_file)
        _scores = _data['score']
        _sizes = _data['sizes']
        plt.errorbar(_sizes.mean(axis=0), _scores.mean(axis=0), _scores.std(axis=0), label=exp_name, linewidth=3.0)
    # --- Improve Plot --- #
    plt.legend(fontsize=18)
    plt.xlabel('Number of samples used for training', fontsize=18)
    plt.ylabel('Mean F1 Score', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()