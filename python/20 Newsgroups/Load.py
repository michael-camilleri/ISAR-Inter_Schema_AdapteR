"""
 Run this Script to Generate the Data Required for the 20-Newsgroups Experiments

 This Script will get the 20Newsgroup Data-Set, using the raw data. This is so that it allows us to
  a) Remove Stop-Words
  b) Control the Random/Shuffle State
  c) Keep only the top K Features...
  d) Remove some labels
  e) Group the data as required...
"""

import numpy as np
import argparse
import sys

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Need to append parent folder to path
sys.path.append('../')
from Tools import npext

DEFAULTS = {'Name': '20NG_BOW',     # Name of the output data-file
            'Random': '1010',       # Random Seed State
            'Features': '5000',     # Feature-Set size
            'TestSize': '0.2'}      # Test-Set Proportion

# We specify the following Groups for the Hierarchical Structure: This is the the same grouping as specified in the
#    original collection of the data: http://qwone.com/~jason/20Newsgroups/
#    Note, that we drop the misc.forsale groups, as this is a bit of a loner...
#    Since this is python 3.6, I am assuming that the insertion order is preserved...
_groups = {'comp': ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                    'comp.windows.x'],
           'rec': ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
           'politics': ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'],
           'sci': ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
           'religion': ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']}


if __name__ == '__main__':

    # ==== Parse Arguments: ==== #
    _arg_parse = argparse.ArgumentParser(description='Load and Format the 20 Newsgroup Data-Set for processing')
    _arg_parse.add_argument('-o', '--output', help='Output File-Name: default is {}'.format(DEFAULTS['Name']),
                            default=DEFAULTS['Name'])
    _arg_parse.add_argument('-r', '--random', help='Seed for all Random States: ensures repeatibility. Defaults to {}'
                            .format(DEFAULTS['Random']), default=DEFAULTS['Random'])
    _arg_parse.add_argument('-f', '--features', help='Number of features to retain (ordered by TF): default is {}'
                            .format(DEFAULTS['Features']), default=DEFAULTS['Features'])
    _arg_parse.add_argument('-t', '--testsize', help='Proportion of the Data to use for the Test-Set: defaults to {}'
                            .format(DEFAULTS['TestSize']), default=DEFAULTS['TestSize'])
    _args = _arg_parse.parse_args()
    rs = int(_args.random)
    ts = float(_args.testsize)
    fs = int(_args.features)

    # ===== Load the Data: ===== #
    print('Loading the Data...')
    # Generate groups to get
    _to_get = [c_ for sublist in _groups.values() for c_ in sublist]
    #  I decided to shuffle the data, but with a known random-state.
    _data = fetch_20newsgroups(subset='all', remove=('headers', 'footers'), categories=_to_get, shuffle=True,
                               random_state=rs)
    # Now, due to presentation reasons, I will reorder the label indices.
    _mapper = [_to_get.index(_) for _ in _data.target_names]
    _y_all = npext.value_map(_data.target, _to=_mapper, shuffle=True)
    # Finally, create the coarse-label list
    _course_labels = np.asarray(list(_groups.keys()))
    print('... Done.\n')

    # ==== Create Vectorizer and Extract the TF-IDF Features in the Data: ==== #
    print('Vectorizing Data...')
    _vectorizer = CountVectorizer(max_features=fs, stop_words='english', token_pattern='(?u)\\b[A-z]{2,}\\b')
    _X_all = _vectorizer.fit_transform(_data.data).todense()
    print('... Done.\n')

    # ==== Perform the Train/Test Split ==== #
    print('Splitting Data...')
    X_train, X_test, y_train, y_test = train_test_split(_X_all, _y_all, test_size=ts, stratify=_y_all, random_state=rs)
    print('... Done.\n')

    # ==== Extract the Hierarchical Grouping ==== #
    print('Generating Coarse-Labels...')
    # Generate Mapper
    _mapper = np.empty(len(_to_get), dtype=int)
    for grp_idx, grp in enumerate(_groups.values()):         # Iterate over Hierarchical Labels
        for lbl in grp:                                      # Iterate over fine-grained labels
            _mapper[_to_get.index(lbl)] = grp_idx            # Encode 'inverse'-mapping
    # Map the target labels
    g_train = npext.value_map(y_train, _to=_mapper, shuffle=True)
    g_test = npext.value_map(y_test, _to=_mapper, shuffle=True)
    print('... Done.\n')

    # ==== Store the Data ==== #
    print('Saving to file (compressed)...')
    vocabulary = {v: k for k, v in _vectorizer.vocabulary_.items()}
    np.savez_compressed(_args.output, full_labelset=np.asarray(_to_get), group_labels=_course_labels, mapper=_mapper,
                        voc=[_2 for (_1, _2) in sorted(vocabulary.items())], X_train=X_train, y_train=y_train,
                        g_train=g_train, X_test=X_test, y_test=y_test, g_test=g_test)
    print('... Done.\n')





