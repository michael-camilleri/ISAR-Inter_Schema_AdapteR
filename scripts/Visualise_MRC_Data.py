"""
Visualiser of the Results

Note that I assume that the segments are of equal length (which is mostly true)

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

DEFAULTS = {
    "Results": ["../data/Learn_DS.npz", "../data/Learn_ISAR.npz"],
    "Schemas": ["I", "II", "III", "IV"],
}

if __name__ == "__main__":

    # ==== Parse Arguments: ==== #
    _arg_parse = argparse.ArgumentParser(
        description="Visualise the results of Training (MRC Harwell Data)"
    )
    _arg_parse.add_argument(
        "-r",
        "--results",
        help="List of (2) Result files for the DS/ISAR models respectively "
        "(separated by spaces). Defaults are: {}".format(DEFAULTS["Results"]),
        default=DEFAULTS["Results"],
        nargs=2,
    )
    _arg_parse.add_argument(
        "-s",
        "--schemas",
        help="The Schema Names: Default is {}".format(DEFAULTS["Schemas"]),
        default=DEFAULTS["Schemas"],
        nargs="*",
    )
    args = _arg_parse.parse_args()

    # ==== Load the Data: ==== #
    with np.load(args.results[0]) as res_ds:
        ell_ds = res_ds["ell"]
    with np.load(args.results[1]) as res_isar:
        ell_isar = res_isar["ell"]

    # ==== Now Print: ==== #
    np.set_printoptions(linewidth=120)
    print("            |       DS       |      ISAR      | ------- T-Test ------")
    print("            |                |                |       DS vs ISAR")
    print("====================== Evidence Log-Likelihood =======================")

    # [A] - Per-Schema ELL: Note that we need to handle mean appropriately
    for s in range(len(args.schemas)):
        print(
            " Schema {:3} | {:.2f} +/- {:4.2f} | {:.2f} +/- {:4.2f} | t={:8.2f}, p={:.2e}".format(
                args.schemas[s],
                np.nanmean(ell_ds[:, s]),
                np.nanstd(ell_ds[:, s]),
                np.nanmean(ell_isar[:, s]),
                np.nanstd(ell_isar[:, s]),
                *ttest_rel(
                    ell_ds[~np.isnan(ell_ds[:, s]), s], ell_isar[~np.isnan(ell_isar[:, s]), s]
                )
            )
        )
    # [B] - Now do global
    ell_ds_global = np.nansum(ell_ds, axis=-1)
    ell_isar_global = np.nansum(ell_isar, axis=-1)
    print("----------------------------------------------------------------------")
    print(
        " Global     | {:.2f} +/- {:4.2f} | {:.2f} +/- {:.2f} | t={:8.2f}, p={:.2e}".format(
            ell_ds_global.mean(),
            ell_ds_global.std(),
            ell_isar_global.mean(),
            ell_isar_global.std(),
            *ttest_rel(ell_ds_global, ell_isar_global)
        )
    )
    print("----------------------------------------------------------------------")
