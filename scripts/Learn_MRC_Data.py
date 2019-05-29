"""
This Script may be used to train the DS and ISAR models on the real data. Due to the nature of our collaboration with
MRC Harwell, we are not allowed to post the data publicly: however, we release the code for posterity.

The Dataset is assumed to be stored as a MessagePack Pandas DataFrame. The index columns are:
 * The Fold Index (Letters, A through K)
 * The Run Number: This splits the samples into contiguous runs
 * The Mouse Identifier (RFID)
 * Time-Stamp (absolute)
As regards Data Columns:
 * Schema: set in I, II, III, V [This is a remnant from other tests]
 * Segment: A 3-Letter alphabetical segment identifier
 * Annotator Labels: Albl.1 through Albl.12 (skipping 11). These are CDType 0 through 13 (12 behaviours + NIS). The NIS
    distinguishes between informative unlabelling and missing data according to Section 3 (pg 7)
 * [Other columns not relevant to our study]
"""
from pandas.api.types import CategoricalDtype as CDType
from mpctools.extensions import pdext, npext, utils
import pandas as pd
import numpy as np
import argparse
import sys

# Load own packages
sys.path.append('..')
from isar.models import DawidSkeneIID, InterSchemaAdapteRIID

# Fixed Constants
ANNOTATORS = ['Albl.{}'.format(i) for i in range(1, 13) if i != 11]  # The annotator columns
SCHEMA_VALUES = ([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],   # I
                 [0, 1, 5, 6, 7, 10, 12, 13],                   # II
                 [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 13],          # III
                 [0, 1, 2, 5, 6, 7, 10, 11, 13])                # V
FULL_SET = 13
SCHEMA_NAMES = ['I', 'II', 'III', 'V']
sS = len(SCHEMA_VALUES)
sK = len(ANNOTATORS)
DS, ISAR = (0, 1)   # Position (index) into arrays

# User Settings
DEFAULTS = \
    {'Input': '../data/mrc_data.df',                        # File-Name containing the Data (Pandas DataFrame)
     'Output': ['../data/Learn_DS', 'none'],  # Output file Names. ../data/Learn_ISAR
     'Random': '0'}                                         # Random Seed


if __name__ == '__main__':

    # ==== Parse Arguments ==== #
    arg_parse = argparse.ArgumentParser(description='Train DS/ISAR Models on the MRC Harwell Dataset')
    arg_parse.add_argument('-i', '--inputs', help='Input Pandas DataFrame File containing the MRC Dataset. Defaults '
                                                  'to {}'.format(DEFAULTS['Input']), default=DEFAULTS['Input'])
    arg_parse.add_argument('-o', '--output', help='Output Result files, one each for the results from the DS (trained '
                                                  'holistically) and ISAR models. Put "None" for any that you do not '
                                                  'want to simulate. Defaults to {}'.format(DEFAULTS['Output']),
                           default=DEFAULTS['Output'], nargs=2)
    arg_parse.add_argument('-r', '--random', help='Seed (offset) for all Random States: ensures repeatibility. '
                                                  'Defaults to {}'.format(DEFAULTS['Random']),
                           default=DEFAULTS['Random'])
    args = arg_parse.parse_args()
    args.random = int(args.random)

    # ==== Read and Prepare the DataSet/Values ==== #
    print('Loading Data...')
    df = pd.read_msgpack(args.inputs)[['Schema', 'Segment', *ANNOTATORS]]
    segments = sorted(df['Segment'].unique().tolist())
    folds = set(df.index.get_level_values(0))
    with np.load('../data/model.mrc.npz') as _data:
        omega = _data['omega']

    # ==== Iterate through the Folds of the DataSet ==== #

    # ---- Prepare Placeholders ---- #
    ell_ds = np.full([len(segments), sS], fill_value=np.NaN) if args.output[DS].lower() != 'none' else None
    ell_isar = np.full([len(segments), sS], fill_value=np.NaN) if args.output[ISAR].lower() != 'none' else None

    # ---- Iterate through Folds ---- #
    for fold_idx, fold_value in enumerate(sorted(folds)):
        print('---------- Training on Fold Index {} ----------'.format(fold_idx))

        # ++++ Prepare which folds we will be training on ++++ #
        train_folds = folds.difference(fold_value)
        valid_folds = fold_value
        train_data = pdext.dfmultiindex(df, lvl=0, vals=train_folds)
        valid_data = pdext.dfmultiindex(df, lvl=0, vals=valid_folds)

        # ++++ Train ISAR Model (Holistically) ++++ #
        if args.output[ISAR].lower() != 'none':
            print(' +++ Training ISAR Model:')

            # [A] Extract Training Data
            Y_train = train_data[ANNOTATORS].astype(float).values
            S_train = train_data['Schema'].cat.codes.values.astype(int)
            sZ = FULL_SET; sU = FULL_SET
            priors = [np.ones(sZ) * 2, np.ones([sZ, sK, sU]) * 2]
            starts = [(npext.sum_to_one(np.ones(sZ)),
                       np.tile(npext.sum_to_one(np.eye(sZ, sU) + np.full([sZ, sU], fill_value=0.01), axis=1)[:, np.newaxis, :], [1, sK, 1]))]

            # [B] Build ISAR Model & Train on Training Set
            isar_model = InterSchemaAdapteRIID([sZ, sK, sS], omega, random_state=args.random, sink=sys.stdout)
            isar_model.fit(Y_train, S_train, priors, starts)

            # [C] Validate Model on a per-segment basis
            for seg_name, seg_data in valid_data.groupby(['Segment']):
                sN = len(seg_data)
                sch_idx = SCHEMA_NAMES.index(seg_data['Schema'][0])  # Get the Schema Value
                seg_idx = segments.index(seg_name)                  # Get the Segment Value (numeric)
                Y_valid = seg_data[ANNOTATORS].astype(float).values         # Get the Y-values
                S_valid = seg_data['Schema'].cat.codes.values.astype(int)   # Get the Schemas
                ell_isar[seg_idx, sch_idx] = isar_model.evidence_log_likelihood(Y_valid, S_valid)/sN

        # ++++ Train DS Model (on a Per-Schema Basis) ++++ #
        if args.output[DS].lower() != 'none':
            print(' +++ Training DS Model:')
            for schema in valid_data['Schema'].unique():  # Iterate only over the schemas existing in the validation set
                print(' .... On Schema {}'.format(schema))
                sch_idx = SCHEMA_NAMES.index(schema)                 # Index into Schema List
                sch_map = utils.dict_invert(SCHEMA_VALUES[sch_idx])  # Mapping from full to intermediate schema

                # [A] - Extract Training Data: we need to recategorise and map
                train_schema = train_data[train_data['Schema'] == schema].copy()
                pdext.recategorise(train_schema, CDType(np.arange(len(sch_map))), ANNOTATORS, _map=sch_map)
                U_train = train_schema[ANNOTATORS].astype(float).values
                sZU = len(SCHEMA_VALUES[sch_idx])
                priors = [np.ones(sZU) * 2, np.ones([sZU, sK, sZU]) * 2]
                starts = [(npext.sum_to_one(np.ones(sZU)),
                           np.tile(npext.sum_to_one(np.eye(sZU, sZU) + np.full([sZU, sZU], fill_value=0.01), axis=1)[:, np.newaxis, :], [1, sK, 1]))]

                # [B] Build DS Model & Train on Training Set
                ds_model = DawidSkeneIID([sZU, sK], random_state=args.random, sink=sys.stdout)
                ds_model.fit(U_train, priors=priors, starts=starts)

                # [C] Validate Model on a per-segment basis
                valid_schema = valid_data[valid_data['Schema'] == schema].copy()
                pdext.recategorise(valid_schema, CDType(np.arange(len(sch_map))), ANNOTATORS, _map=sch_map)
                for seg_name, seg_data in valid_schema.groupby(['Segment']):
                    sN = len(seg_data)
                    seg_idx = segments.index(seg_name)  # Get the Segment Value (numeric)
                    U_valid = seg_data[ANNOTATORS].astype(float).values
                    ell_ds[seg_idx, sch_idx] = ds_model.evidence_log_likelihood(U_valid)/sN

    # ===== Finally store the results to File: ===== #
    print('Storing Results to file ... ')
    if args.output[DS].lower() != 'none':
        np.savez_compressed(args.output[DS], ell=ell_ds)
    if args.output[ISAR].lower() != 'none':
        np.savez_compressed(args.output[ISAR], ell=ell_isar)
