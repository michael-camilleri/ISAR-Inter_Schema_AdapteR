"""
Script to Visualise the results of training
"""
# Load Python Specific Libraries
from matplotlib.ticker import MultipleLocator as MLoc
from matplotlib import pyplot as plt
import numpy as np
import argparse

DEFAULTS = {'Results': ['../../data/Results_02.npz', '../../data/Results_05.npz', '../../data/Results_10.npz',
                        '../../data/Results_20.npz'],  # List of result files
            'Names': ['  2.0%', '  5.0%', '10.0%', '20.0%']}              # Names to accompany Result Files

if __name__=='__main__':

    # ==== Parse Arguments: ==== #
    _arg_parse = argparse.ArgumentParser(description='Visualise the results of Training')
    _arg_parse.add_argument('-r', '--results', help='List of Result files (separated by spaces). Defaults are: {}'.
                            format(DEFAULTS['Results']), default=DEFAULTS['Results'], nargs='*')
    _arg_parse.add_argument('-n', '--names', help='Names to accompany Result files: must be same number of elements as '
                                                  'the [results] list. Defaults to: {}'.format(DEFAULTS['Names']), # 
                            default=DEFAULTS['Names'], nargs='*')
    args = _arg_parse.parse_args()
    assert len(args.results) == len(args.names), 'Names must be of same length as Results'

    # ==== Load Files and Visualise Plots ==== #
    # --- Iterate through Experiments --- #
    for exp_name, exp_file in zip(args.names, args.results):
        with np.load(exp_file) as _data:
            _scores = _data['score']
            _sizes = _data['sizes']
        plt.errorbar(_sizes.mean(axis=0), _scores.mean(axis=0), _scores.std(axis=0), label=exp_name, linewidth=3.0)
    # --- Improve Plot --- #
    plt.legend(fontsize=20, loc=4)
    plt.xlabel('Number of samples used for training', fontsize=20)
    plt.ylabel('Mean F1 Score', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True, which='major', axis='both')
    plt.axes().yaxis.set_minor_locator(MLoc(0.025))
    plt.grid(True, which='minor', axis='y', color='c', linestyle='--')
    plt.show()
