"""
This is the Dawid-Skene Model, for comparing the ISAR model against.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
http://www.gnu.org/licenses/.

Author: Michael P. J. Camilleri
"""

from mpctools.parallel import IWorker, WorkerHandler
from sklearn.metrics import confusion_matrix
from mpctools.extensions import npext
from collections import namedtuple
from numba import jit, float64
import numpy as np
import time as tm
import warnings
import copy


class DawidSkeneIID(WorkerHandler):
    """
    This Class implements the Noisy Annotator Model, following the Formulation of Dawid-Skene in an
    IID fashion.
    """

    DSIIDResult_t = namedtuple("DSIIDResult_t", ["Pi", "Psi", "Converged", "Best", "LLEvolutions"])

    def __init__(
        self,
        dims,
        params=None,
        max_iter=100,
        converge_tol=1e-4,
        predict_tol=0.0,
        n_jobs=-1,
        random_state=None,
        sink=None,
    ):
        """
        Initialises the Model

        :param dims: The Dimensions of the Model, basically Z/U, K (See Paper: We assume Z == U)
        :param params: If not None, specifies a value for the parameters: otherwise, these are
                       initialised to completely perfect distributions (i.e. perfect annotators)
        :param max_iter: Maximum number of iterations in EM computation
        :param converge_tol: Tolerance for convergence in EM Computation
        :param predict_tol: Tolerance when computing predictions: if less than this, will not
                       return a prediction (np.NaN). By default, this is disabled (set to 0)
        :param n_jobs: Number of jobs: see documentation for mpctools.WorkerHandler
        :param random_state: If set, ensures reproducible results
        :param sink: For debugging, outputs progress etc..
        """

        # Call Super-Class
        super(DawidSkeneIID, self).__init__(n_jobs, sink)

        # Initialise own-stuff
        self.sZU, self.sK = dims
        self.__rand = random_state if random_state is not None else int(tm.time())
        self.__max_iter = max_iter
        self.__conv_toler = converge_tol
        self.__pred_toler = predict_tol

        # Prepare a valid probability for the model
        if params is None:
            self.Pi = np.full(self.sZU, 1 / self.sZU)
            self.Psi = np.tile(np.eye(self.sZU)[:, np.newaxis, :], [1, self.sK, 1])
        else:
            self.Pi, self.Psi = params

    def sample(self, n_runs, n_times, nA, pA):
        """
        Sampler for the Model
        :param n_runs:  Number of Runs to generate. Schemas/Annotators only vary between runs!
        :param n_times: Length of each run
        :param nA:      Number of Annotators to assign per Run: this allows leaving out certain annotators in runs.
        :param pA:      Probability over Annotators.
        :return:
        """
        # First Seed the random number generator
        np.random.seed(self.__rand)

        # Generate Z at one go
        Z = np.random.choice(self.sZU, size=n_runs * n_times, p=self.Pi)  # Latent State

        # With regards to the observations, have to do on a sample-by-sample basis.
        A = np.empty([n_runs * n_times, nA], dtype=int)  # Annotator Selection Matrix
        U = np.full([n_runs * n_times, self.sK], fill_value=np.NaN)

        # Iterate over all segments
        for n in range(n_runs):
            # Pick Annotators for this segment
            A[n * n_times : (n + 1) * n_times, :] = np.random.choice(
                self.sK, size=nA, replace=False, p=pA
            )
            # Iterate over time-instances in this Segment
            for nt in range(n * n_times, (n + 1) * n_times):
                for k in A[nt]:  # Iterate over Annotators for this segment
                    U[nt, k] = np.random.choice(
                        self.sZU, p=self.Psi[Z[nt], k, :]
                    )  # Annotator Emission (confusion)

        # Return Data
        return Z, A, U

    def fit(self, U, z=None, priors=None, starts=1, return_diagnostics=False, learn_prior=True):
        """
        Fit the Parameters Pi/Psi to the data, in either a supervised or unsupervised manner.

        :param U:        Observations:  Size N x K. Note that where the annotator does not label a sample, this should
                         be represented by NaN
        :param z:        If not None, then this amounts to supervised learning: otherwise, we use EM in an unsupervised
                         manner.
        :param priors:   Prior Probabilities for Pi and Psi [|Z|, |Z| x K x |U|]. Note that if none, then this amounts to
                         0 counts (alpha = 1)
        :param starts:   This must be the list of Starting points (each starting point should be a tuple/list,
                         containing the starting pi/psi matrices). Note that in this case, they may be modified. If not
                         provided, then a single start with random (dirichlet) counts (from prior) is generated.
        :param return_diagnostics: If true, return a named tuple of type DSIIDResult_t containing diagnostic information
                         Note, that this is only applicable if z is None (i.e. unsupervised)
        :param learn_prior: Only applicable when performing supervised learning. If False, then instead of using the
                         data to learn the distribution over Z, this is taken to be the prior probability.
        :return:  Self, (Diagnostics) : Self for chaining, Diagnostics if requested. These contain:
                        * Pi : Fitted Prior over Targets
                        * Psi: Annotator Confusion Matrix
                        * Converged: Which runs in the EM scheme converged
                        * Best: The index of the run with the best log-likelihood
                        * LLEvolutions: The evolution of the log-likelihood for all runs.
        """
        # Branch based on whether we are doing this in a supervised or unsupervised fashion.
        if z is not None:
            # --- Indicate branch --- #
            self._print("Fitting DS Model in a supervised fashion.")

            # --- Handle Prior Smoothing --- #
            if priors is None:
                assert learn_prior is True, "If Not Learning Prior, then you must define the Priors"
                self.Pi = np.zeros_like(self.Pi)
                self.Psi = np.zeros_like(self.Psi)
            else:
                self.Pi = priors[0] - 1
                self.Psi = priors[1] - 1

            # --- Construct prior on z: --- #
            #  --- Note, that if not specified, then prior
            if learn_prior:
                val, cnts = np.unique(
                    z.astype(int), return_counts=True
                )  # Guard against some values not appearing!
                self.Pi[val] += cnts
            self.Pi = npext.sum_to_one(self.Pi)

            # --- Construct Conditional on U from confusion matrices --- #
            for k in range(self.sK):
                # Mask against NaN
                valid_u = ~np.isnan(U[:, k])
                if np.any(valid_u):
                    self.Psi[:, k, :] += confusion_matrix(
                        z[valid_u], U[valid_u, k], labels=np.arange(len(self.Pi))
                    )
                # Otherwise, will just retain prior
            self.Psi = npext.sum_to_one(self.Psi, axis=-1)

            # --- Return Self (chaining) --- #
            return self

        else:
            # --- Indicate branch --- #
            self._print("Fitting DS Model using Expectation Maximisation")

            # --- Handle Priors first --- #
            if priors is None:
                priors = (np.ones_like(self.Pi), np.ones_like(self.Psi))

            # --- Now Handle Starting Points --- #
            np.random.seed(self.__rand)
            if hasattr(starts, "__len__"):
                num_work = len(starts)
            else:
                num_work = starts
                starts = [
                    (npext.Dirichlet(priors[0]).sample(), npext.Dirichlet(priors[1]).sample())
                    for _ in range(num_work)
                ]

            # Perform EM (and format output)
            self._print(
                "Running {0} starts for (max) {1} iterations.".format(num_work, self.__max_iter)
            )
            self.start_timer("global")
            results = self.run_workers(
                num_work,
                self._EMWorker,
                _configs=self._EMWorker.ComputeParams_t(
                    self.__max_iter, self.__conv_toler, U, *priors
                ),
                _args=starts,
            )
            # Stop Main timer
            self.stop_timer("global")

            # Check that there was something to return:
            if results is not None:
                # Consolidate Data
                self.Pi = results.Pi
                self.Psi = results.Psi

                # Display some statistics
                self._write(
                    "DS Model was fit in {0:1.5f}s ({1:1.5f}s/run):\n".format(
                        self.elapsed("global"), self.elapsed("global") / num_work
                    )
                )

            # Build (and return) Information Structure
            return (self, results) if return_diagnostics else self

    def evidence_log_likelihood(self, U, prior=None):
        """
        Compute the (Marginalised Z) Evidence Log-Likelihood of the model on the data

        :param U:       The Observations: Size N x K. Note that where the annotator does not label a sample, this should
                        be represented by NaN
        :param prior:   Any prior probabilities on parameters. Can be None
        :return:        Log-Likelihood
        """
        if prior is not None:
            return (
                npext.Dirichlet(prior[0]).logsumpdf(self.Pi)
                + npext.Dirichlet(prior[1]).logsumpdf(self.Psi)
                + self._EMWorker._compute_responsibilities(U, self.Pi, self.Psi).log_likelihood
            )
        else:
            return self._EMWorker._compute_responsibilities(U, self.Pi, self.Psi).log_likelihood

    def predict(self, U):
        """
        Predict the latent state Z given the observations. Note that because of the prediction
        tolerance, the returned values are float.

        :param U:  Observations: Size N by K (of domain |U| with missing data as NaN)
        :return:   Predictions over the latent states
        """
        resp = self._EMWorker._compute_responsibilities(U, self.Pi, self.Psi).gamma
        pred = np.argmax(resp, axis=-1).astype(float)
        pred[resp[np.arange(U.shape[0]), pred.astype(int)] < self.__pred_toler] = np.NaN
        return pred

    def predict_proba(self, U):
        """
        Get the distribution over the latent state Z.

        :param U:  Observations: Size N by K (of domain |U| with missing data as NaN)
        :return:   Distribution over latent state Z
        """
        return self._EMWorker._compute_responsibilities(U, self.Pi, self.Psi).gamma

    def _aggregate_results(self, results):
        """
        Maximise parameters over runs: for comparison's sake, it assumes that the pi's and psi's are sorted in a
        consistent order (for comparison).

        :param results: the ApplyResult object
        :return: Dictionary, containing all parameters
        """

        # Initialise Placeholders
        _final_pis = []  # Final Pi for each EM run
        _final_psis = []  # Final Psi for each EM run
        _final_llik = []  # Final Likelihood, for each EM run
        _converged = []  # Whether the run converged:
        _evol_llikel = []  # Evolutons of log-likelihoods

        # Iterate over results from each worker:
        for result in results:
            _final_pis.append(result.Pi)
            _final_psis.append(result.Psi)
            _final_llik.append(result.LLEvolutions[-1])
            _evol_llikel.append(result.LLEvolutions)
            _converged.append(result.Converged)

        # Convert to Numpy Arrays for Indexing
        _final_llik = np.asarray(_final_llik)
        _final_pis = np.asarray(_final_pis)
        _final_psis = np.asarray(_final_psis)

        # Check that we got something, and if not:
        if len(_converged) < 1:
            warnings.warn("The Runs failed completely - The parameters have not been changed.")
            return None

        # Check whether at least one converged, and warn if not...
        if np.any(_converged):
            _masked_likelihood = np.ma.masked_array(_final_llik, np.logical_not(_converged))
        else:
            warnings.warn("None of the Runs Converged - Parameters have been updated but may not "
                          "represent optimal values. It is advised to rerun with a larger maximum "
                          "number of iterations.")
            _masked_likelihood = _final_llik

        # Find the best one out of those converged, or out of all, if none converged
        _best_index = _masked_likelihood.argmax().squeeze()

        # Return Results in Dictionary:
        return self.DSIIDResult_t(
            Pi=_final_pis[_best_index],
            Psi=_final_psis[_best_index],
            Best=_best_index,
            Converged=np.asarray(_converged),
            LLEvolutions=_evol_llikel,
        )

    # ========================== Private Nested Implementations ========================== #
    class _EMWorker(IWorker):
        """
        (Private) Nested class for running the EM Algorithm

        Note that in this case, the index order is n,z,k,u (i.e. Sample, Latent, Annotator, Label)
        """

        ComputeParams_t = namedtuple(
            "ComputeParams_t", ["max_iter", "tolerance", "U", "prior_pi", "prior_psi"]
        )
        Responsibilities_t = namedtuple("Responsibilities_t", ["gamma", "log_likelihood"])
        ComputeResult_t = namedtuple("ComputeResult_t", ["Pi", "Psi", "Converged", "LLEvolutions"])

        def __init__(self, _id, _handler):
            """
            Initialiser

            :param _id:         Identifier - allows different seeds for random initialisation
            :param _handler:    The Worker Handler
            """
            # Initialise Super-Class
            super(DawidSkeneIID._EMWorker, self).__init__(_id, _handler)

        def parallel_compute(self, _common, _data):
            """
            Implementation of Parallel Computation
            :param _common: NamedTuple of type ComputeParams_t
                            > max_iter:     Maximum number of iterations to compute for
                            > tolerance:    Tolerance Parameter for early stopping: if Likelihood does not change more
                                            than this amount between Iterations, then stop.
                            > U:            The Data, of size N by K (values in {0, ..., sZ-1, np.NaN})
                            > prior_pi:     Prior Probabilities for the Pi Distributions (over Z), of size |Z|. Note
                                            that this should be the true prior )alpha) and not laplacian smoothing
                                            counts.
                            > prior_psi:    Prior Probabilities for the Psi Distributions (over U), of size  |Z| by K by
                                            |U|. Again these should be the true priors, (alpha) and not laplacian
                                            smoothing counts.
            :param _data:   None or [pi, psi]
                            > pi - Initial value for pi
                            > psi - Initial value for psi
            :return: Named Tuple containing
                            > Pi (optimised): size |Z|
                            > Psi (optimised): size |Z| by K by |U|
                            > Whether the run converged or not within max-iterations
                            > Evolution of (evidence) log-likelihood through iterations.
            """
            # Initialise parameters
            # max_iter, toler, sZ, U, pPi, pPsi, sort = _common

            # Initialise Random Points (start)
            pi, psi = _data

            # Initialise Dirichlets
            pi_dir = npext.Dirichlet(_common.prior_pi, ignore_zeros=False)
            psi_dir = npext.Dirichlet(_common.prior_psi, ignore_zeros=False)

            # Prepare for Loop
            iterations = 0  # Iteration Count
            loglikelihood = []  # Observed Data Log-Likelihood (evolutions)

            # Start the Loop
            while iterations < _common.max_iter:
                # -------------- E-Step: -------------- #
                # ++++ Compute Responsibilities ++++ #
                # Pre-Allocate Gamma
                msg = self._compute_responsibilities(_common.U, pi, psi)
                # ++++ Now compute log-likelihood ++++ #
                loglikelihood.append(
                    pi_dir.logpdf(pi) + psi_dir.logsumpdf(psi) + msg.log_likelihood
                )

                # --------- Likelihood-Check: --------- #
                # Check for convergence!
                if self._converged(loglikelihood, _common.tolerance):
                    break

                # -------------- M-Step: -------------- #
                # First Compute Pi
                pi = npext.sum_to_one(np.sum(msg.gamma, axis=0).squeeze() + _common.prior_pi - 1)
                # Finally Compute Psi
                psi = copy.deepcopy(_common.prior_psi - 1)
                self.__update_psi(psi, _common.U, msg.gamma)
                psi = npext.sum_to_one(psi, axis=-1)

                # -------------- Iteration Control -------------- #
                iterations += 1
                self.update_progress(iterations * 100.0 / _common.max_iter)

                # Clean up
            self.update_progress(100.0)
            converged = self._converged(loglikelihood, _common.tolerance)

            # Return Result
            return DawidSkeneIID._EMWorker.ComputeResult_t(pi, psi, converged, loglikelihood)

        @staticmethod
        def _compute_responsibilities(U, pi, psi):
            """
            Compute the Responsibilities

            :param U:       The Data, with Unlabelled/Missing data indicated by np.NaN
            :param pi:      The Prior Probabilities
            :param psi:     The Emission Probabilities
            :return:        The Responsibilities and the Log-Likelihood
            """
            # Compute Unnormalised Gamma using JIT
            gamma = np.tile(pi[np.newaxis, :], [len(U), 1])
            DawidSkeneIID._EMWorker.__gamma(U, pi, psi, gamma)

            # Normalise to sum to 1, and at the same, through the normaliser, compute observed log-likelihood
            gamma, normaliser = npext.sum_to_one(gamma, axis=-1, norm=True)
            log_likelihood = -np.log(normaliser).sum()

            # Return
            return DawidSkeneIID._EMWorker.Responsibilities_t(gamma, log_likelihood)

        @staticmethod
        def _converged(likelihoods, tolerance):
            """
            Convergence Check

            Returns True only if converged, within tolerance
            :param likelihoods: Array of Log-Likelihood
            :param tolerance:   Tolerance Parameter
            :return:
            """
            if len(likelihoods) < 2:
                return False
            elif likelihoods[-1] < likelihoods[-2]:
                warnings.warn("Drop in Log-Likelihood Observed! Results are probably wrong.")
            else:
                return abs((likelihoods[-1] - likelihoods[-2]) / likelihoods[-2]) < tolerance

        @staticmethod
        @jit(
            signature_or_function=(float64[:, :], float64[:], float64[:, :, :], float64[:, :]),
            nopython=True,
        )
        def __gamma(U, pi, psi, gamma):
            """
            Convenience Wrapper for JIT compilation of Gamma Computation (unnormalised)

            :param U:       The Data, with Unlabelled/Missing data indicated by np.NaN
            :param pi:      The Prior Probabilities
            :param psi:     The Emission Probabilities
            :param gamma:   The gamma placeholder (modified in place). Note that these are unnormalised!
            :return         None: the gamma is modified in place.
            """
            sN, sK = U.shape
            sZU = pi.shape[0]

            # Compute - this is basically an iteration over samples and states
            for n in range(sN):
                for z in range(sZU):
                    for k in range(sK):
                        if not (np.isnan(U[n, k])):
                            gamma[n, z] *= psi[z, k, int(U[n, k])]

        @staticmethod
        @jit(signature_or_function=(float64[:, :, :], float64[:, :], float64[:, :]), nopython=True)
        def __update_psi(psi, U, gamma):
            """
            Convenience wrapper for updating Psi using JIT

            :param psi:       The initialisation for Psi (typically prior counts): Modified in place
            :param U:         The observations
            :param gamma:     The posterior probabilities
            """
            for n in range(len(U)):
                for k in range(psi.shape[1]):
                    if not (np.isnan(U[n, k])):
                        psi[:, k, int(U[n, k])] += gamma[n, :]
