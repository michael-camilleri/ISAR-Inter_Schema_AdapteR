"""
Script to analyse the ability of ISAR to learn the true parameters the data-generating process.

Note that while the script allows some flexibility, it should probably not combine sizes < [13, 11] with non extreme
(one-vs-rest) scenarios.

Note also that in these simulations, it is (again) assumed that each annotator either labels a segment or puts NIS.
This stems from our modelling behaviour, in that if the annotator does not label the sample it is solely because it is
not in-schema (i.e. we do not model missing-at-random behaviour.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
http://www.gnu.org/licenses/.

Author: Michael P. J. Camilleri
"""

from mpctools.extensions import npext
import numpy as np
import argparse
import sys

# Load own packages
sys.path.append("..")
from isar.models import InterSchemaAdapteRIID, DawidSkeneIID

# Default Parameters
DEFAULTS = {
    "Output": "../data/Parameters_ISAR",  # Result File
    "Random": "0",  # Random Seed offset
    "Numbers": ["0", "20"],  # Range: start index, number of runs
    "Lengths": ["60", "5400"],  # Number and length of segments
    "Sizes": ["13", "11"],  # Dimensionality of the data: sZ/sK
    "Steps": ["0.001", "0.005", "0.01", "0.05", "0.1", "0.5", "1.0"],  # Step Sizes
    "Extreme": False,  # One-V-Rest?
    "Different": False,  # Different schemas within Sample?
}
nA = 3

if __name__ == "__main__":

    # ==== Parse Arguments ==== #
    _arg_parse = argparse.ArgumentParser(
        description="Simulate and Train ISAR to evaluate Parameter Learning"
    )
    _arg_parse.add_argument(
        "-o",
        "--output",
        help="Output Result file: defaults to {}".format(DEFAULTS["Output"]),
        default=DEFAULTS["Output"],
    )
    _arg_parse.add_argument(
        "-r",
        "--random",
        help="Seed (offset) for all Random States: ensures repeatibility. "
        "Defaults to {}".format(DEFAULTS["Random"]),
        default=DEFAULTS["Random"],
    )
    _arg_parse.add_argument(
        "-n",
        "--numbers",
        help="Range of Runs to simulate: tuple containing the start index and "
        "number of runs. Defaults to [{} {}]".format(*DEFAULTS["Numbers"]),
        nargs=2,
        default=DEFAULTS["Numbers"],
    )
    _arg_parse.add_argument(
        "-l",
        "--lengths",
        help="Number and length of each segment. Default {}".format(DEFAULTS["Lengths"]),
        default=DEFAULTS["Lengths"],
        nargs=2,
    )
    _arg_parse.add_argument(
        "-s",
        "--sizes",
        help="Dimensionality of the Problem in terms of number of latent states "
        "|Z| and number of annotators |K|. Default is {} (i.e. the same as in"
        " the real data). Note that the values cannot exceed these defaults, "
        "since the parameters are extracted as sub-matrices of the true ones."
        " Note also that the behaviour is undefined if |Z| is not {} when "
        "using MRC Harwell schemas.".format(DEFAULTS["Sizes"], DEFAULTS["Sizes"][0]),
        default=DEFAULTS["Sizes"],
        nargs=2,
    )
    _arg_parse.add_argument(
        "-i",
        "--increments",
        help="Fractions of the data-set to evaluate the parameters at. Must "
        "range from >0 to 1.0",
        nargs="*",
        default=DEFAULTS["Steps"],
    )
    _arg_parse.add_argument(
        "-d",
        "--different",
        help="If set, then annotators may label using different schemas per "
        "sample: otherwise, within the same sample, all annotators use "
        "the same schema. Default is False.",
        action="store_true",
    )
    _arg_parse.add_argument(
        "-e",
        "--extreme",
        help="If set, then there are |Z| schemas corresponding to |Z| latent "
        "states in a one-vs-rest configuration: otherwise (default) use "
        "the MRC Harwell schema set. Note that in the former case, the "
        "probability over Schemas and Annotators is uniform.",
        action="store_true",
    )

    _arg_parse.set_defaults(different=DEFAULTS["Different"])
    _arg_parse.set_defaults(extreme=DEFAULTS["Extreme"])
    # ---- Parse Arguments and transform as required ---- #
    args = _arg_parse.parse_args()
    args.random = int(args.random)
    run_offset = int(args.numbers[0])
    run_length = int(args.numbers[1])
    args.increments = [float(s) for s in args.increments]
    sN, sT = [int(l) for l in args.lengths]
    sZ, sK = [int(s) for s in args.sizes]
    sU = sZ
    NIS = sZ  # NIS Label

    # ==== Load/Prepare the Data ==== #
    with np.load("../data/model.mrc.npz") as _data:
        # ---- Load Baseline Components ---- #
        pi = npext.sum_to_one(_data["pi"][:sZ])  # Note that we have to ensure normalisation
        psi = npext.sum_to_one(_data["psi"][:sZ, :sK, :sU], axis=-1)
        # ---- Now Handle Omega ---- #
        if args.extreme:
            sS = sZ
            sY = sZ + 1
            omega = np.zeros([sS, sU, sY])
            for s in range(sS):
                for u in range(sU):
                    for y in range(sY):
                        if (y == u) and (u == s):
                            omega[s, u, y] = 1.0
                        elif (y == sY - 1) and (u != s):
                            omega[s, u, y] = 1.0
        else:
            omega = _data["omega"]
            sS, _, sY = omega.shape
        # ---- Also Handle Probabilities ---- #
        if args.extreme:
            PDF_ANNOT = npext.sum_to_one(np.ones(sK))  # Probability over Annotators
            PDF_SCHEMA = npext.sum_to_one(np.ones(sS))  # Probability over Schemas
        else:
            PDF_ANNOT = npext.sum_to_one(
                [49, 31, 11, 4, 10, 25, 9, 7, 6, 3, 3][:sK]
            )  # Probability over Annotators
            PDF_SCHEMA = npext.sum_to_one([13, 15, 17, 10])  # Probability over Schemas

    # ==== Simulate (and Learn) Model(s) ==== #
    # ---- Prepare Storage ---- #
    # Basically, for each parameter, we index by run, then progression in number of steps and finally the dimensions of
    #  the parameter itself.
    pi_true = np.empty([run_length, sZ])  # True Values
    pi_isar = np.empty([run_length, len(args.increments), sZ])  # As learnt using ISAR
    pi_full = np.zeros(
        [run_length, len(args.increments), sZ]
    )  # As learnt from fully-observeable data
    psi_true = np.empty([run_length, sZ, sK, sU])
    psi_isar = np.empty([run_length, len(args.increments), sZ, sK, sU])
    psi_full = np.zeros([run_length, len(args.increments), sZ, sK, sU])
    sizes = np.zeros([run_length, len(args.increments)], dtype=int)  # Sizes of the Data

    # --- Iterate over Runs ---- #
    for run in range(run_offset, run_offset + run_length):
        print("Executing Simulation Run: {} ...".format(run))

        # Seed with Random Offset + run value
        np.random.seed(args.random + run)

        # [A] - Generate Data
        print(" - Generating Data:")
        # Add some noise to Parameters to introduce some randomness
        if run == run_offset:
            Pi = pi
            Psi = psi
        else:
            Pi = npext.sum_to_one(pi + np.random.uniform(0.0, 0.05, size=sZ))
            Psi = npext.sum_to_one(psi + np.random.uniform(0.0, 0.05, size=[sZ, sK, sU]), axis=-1)
        # Keep Track of Parameters
        pi_true[run, :] = Pi
        psi_true[run, :, :, :] = Psi
        sampler = InterSchemaAdapteRIID(
            [sZ, sK, sS], omega, (Pi, Psi), random_state=args.random + run, sink=None
        )
        Z, S, A, Y, U = sampler.sample(sN, sT, nA, PDF_SCHEMA, PDF_ANNOT, not args.different, True)
        # And Shuffle all to avoid same sizes when
        permutation = np.random.permutation(sN * sT)
        Z = Z[permutation]
        S = S[permutation, :] if args.different else S[permutation]
        U = U[permutation, :]
        Y = Y[permutation, :]

        # [B] - Learn using fully-labelled Data: this is basically DS Model using supervised learning
        print(" - Learning using fully-observed Data")
        for i, inc in enumerate(args.increments):
            # Extract Sub-Set to operate on
            sizes[run, i] = int(inc * sN * sT)
            z_i = Z[: sizes[run, i]]
            u_i = U[: sizes[run, i], :]
            idcs, cnts = np.unique(z_i, return_counts=True)

            supervised = DawidSkeneIID((sZ, sK)).fit(
                u_i, z_i, priors=[np.ones(sZ) * 2, np.ones([sZ, sK, sU]) * 2]
            )
            pi_full[run, i, :] = supervised.Pi
            psi_full[run, i, :, :, :] = supervised.Psi

        # [C] - Learn using ISAR
        print(" - Learning using ISAR")
        for i, inc in enumerate(args.increments):
            # Extract Sub-Set(s) to operate on
            y_i = Y[: sizes[run, i], :]
            s_i = S[: sizes[run, i], :] if args.different else S[: sizes[run, i]]
            priors = [np.ones(sZ) * 2, np.ones([sZ, sK, sU]) * 2]
            starts = [
                (
                    npext.sum_to_one(np.ones(sZ)),
                    np.stack(
                        [
                            npext.sum_to_one(
                                np.eye(sZ, sZ) + np.full([sZ, sZ], fill_value=0.01), axis=1
                            )
                            for _ in range(sK)
                        ]
                    ).swapaxes(0, 1),
                )
            ]
            # Train ISAR Model
            isar_model = InterSchemaAdapteRIID(
                [sZ, sK, sS], omega, max_iter=200, random_state=args.random, sink=sys.stdout
            )
            isar_model.fit(y_i, s_i, priors, starts)
            # Store results
            pi_isar[run, i, :] = isar_model.Pi
            psi_isar[run, i, :, :, :] = isar_model.Psi

    # ===== Finally store the results to File: ===== #
    print("Storing Results to file ... ")
    np.savez_compressed(
        args.output,
        Pi_true=pi_true,
        Pi_full=pi_full,
        Pi_isar=pi_isar,
        Psi_true=psi_true,
        Psi_full=psi_full,
        Psi_isar=psi_isar,
        Sizes=sizes,
    )
