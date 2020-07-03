"""
This is the ISAR Model, applied to the multi-annotator task where P(U|Z) is a mixture of categorical distributions
according to the Dawid-Skene Model.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
http://www.gnu.org/licenses/.

Author: Michael P. J. Camilleri
"""

from mpctools.parallel import WorkerHandler, IWorker
from mpctools.extensions import npext
from numba import jit, float64, int64
from collections import namedtuple
import numpy as np
import time as tm
import warnings


class InterSchemaAdapteRIID(WorkerHandler):
    """
    This Class implements the Inter-Schema Adapter model, based on a mixture of categorical distributions,
    which are in turn emitted via a deterministic schema translation. It is implemented in a multi-processing framework
    allowing multiple independent restarts to happen in parallel.

    Refer to the Paper, esp Appendix for derivations/definitions of the equations
    """

    # ============================== Data Type Definitions =============================== #
    ISARIIDResult_t = namedtuple(
        "ISARIIDResults_t", ["Pi", "Psi", "Converged", "Best", "LLEvolutions"]
    )

    # ========================== Initialisers ========================== #
    def __init__(
        self,
        dims,
        omega,
        params=None,
        max_iter=100,
        converge_tol=1e-4,
        predict_tol=0.0,
        n_jobs=-1,
        max_mem=200000,
        random_state=None,
        sink=None,
    ):
        """
        Initialiser

        :param dims: Dimensionality of the problem: basically, Z/U, K, S (See Paper: We assume Z==U)
        :param omega: Fixed ISAR Matrix. This must be an NdArray of size |S| by |U| by |Y|
        :param params: The Pi/Psi Parameters. If None, will be initialised assuming perfect
                       annotators.
        :param max_iter: Maximum number of iterations in EM computation
        :param converge_tol: Tolerance for convergence in EM Computation
        :param predict_tol: Tolerance when computing predictions: if less than this, will not
                       return a prediction (np.NaN). By default, this is disabled (set to 0)
        :param n_jobs: Number of jobs: see documentation for mpctools.WorkerHandler
        :param max_mem: Maximum Memory to execute with (number of samples). When using
                        multiprocessing, this will be divided by the number of jobs to prevent
                        issues.
        :param random_state: If set, ensures reproducible results
        :param sink: For debugging, outputs progress etc..
        """

        # Call Super-Class
        super(InterSchemaAdapteRIID, self).__init__(n_jobs, sink)

        # Initialise own-stuff
        self.sZU, self.sK, self.sS = dims
        self.__rand = random_state if random_state is not None else int(tm.time())
        self.__max_iter = max_iter
        self.__conv_toler = converge_tol
        self.__pred_toler = predict_tol
        self.__max_mem = max_mem
        self.__jobs = max(
            1, n_jobs
        )  # Keep track of whether we are doing multiprocessing or just threading!

        # Now the Parameters
        self.Omega = omega
        if params is None:
            self.Pi = np.full(self.sZU, 1 / self.sZU)
            self.Psi = np.tile(np.eye(self.sZU)[:, np.newaxis, :], [1, self.sK, 1])
        else:
            self.Pi, self.Psi = params

    def sample(self, n_runs, n_times, nA, pS, pA, same_schema_per_annotator=True, return_U=False):
        """
        Generate Samples from the Model. This currently only supports equal-sized runs.

        :param n_runs:  Number of Runs to generate. Schemas/Annotators only vary between runs!
        :param n_times: Length of each run
        :param nA:      Number of Annotators to assign per Run
        :param pS:      Probability over Schemas
        :param pA:      Probability over Annotators.
        :param same_schema_per_annotator: If False, then it is possible for different annotators to use a different
                        schema
        :param return_U: If True, then also keep track of U
        :return:        Tuple containing Z, S, A, Y
        """
        # First Seed the random number generator
        np.random.seed(self.__rand)

        # Generate Z and S at one go
        Z = np.random.choice(self.sZU, size=n_runs * n_times, p=self.Pi)  # Latent State
        if same_schema_per_annotator:
            S = np.repeat(np.random.choice(self.sS, size=n_runs, p=pS), n_times)  # Schema
        else:
            S = np.repeat(np.random.choice(self.sS, size=[n_runs, self.sK], p=pS), n_times, axis=0)

        # With regards to the observations, have to do on a sample-by-sample basis.
        Y = np.full([n_runs * n_times, self.sK], fill_value=np.NaN)  # Observations Matrix
        A = np.empty([n_runs * n_times, nA], dtype=int)  # Annotator Selection Matrix
        U = np.full([n_runs * n_times, self.sK], fill_value=np.NaN) if return_U else None
        if same_schema_per_annotator:
            self.__generate_samples_same(n_runs, n_times, pA, self.Psi, self.Omega, Z, S, A, Y, U)
        else:
            self.__generate_samples_different(
                n_runs, n_times, pA, self.Psi, self.Omega, Z, S, A, Y, U
            )

        # Return as a Tuple
        return (Z, S, A, Y, U) if return_U else (Z, S, A, Y)

    def fit(self, Y, S, priors=None, starts=1, return_diagnostics=False):
        """
        Fit the Parameters Pi/Psi from the data. Note that unlike the base DS Model, we do not currently support fully
        supervised learning.

        :param Y:        Annotator provided labels [N by |K|]
        :param S:        Schema for each sample: [N] or [N by |K|] if using individual schema per annotator.
        :param priors:   Prior Probabilities for Pi and Psi (|Z|, [|Z| by K by |U|]): Note these must be the true alpha
                         values in the Dirichlet prior formulation (and not raw counts). If not provided, then
                         alpha=1 priors are implied.
        :param starts:   This can be either:
                            * Integer - Number of random starts to perform
                            * List of Starting points. Each starting point should be a tuple/list, containing the
                              starting pi/psi matrices. N.B. These will be modified (so pass a copy if need to preserve)
        :param return_diagnostics: If true, return a named tuple of type ISARIIDResult_t containing diagnostic
                                   information.
        :return:    Self, (Diagnostics) : Self for chaining, Diagnostics if requested. These contain:
                        * Pi : Fitted Prior over Targets
                        * Psi: Annotator Confusion Matrix
                        * Converged: Which runs in the EM scheme converged
                        * Best: The index of the run with the best log-likelihood
                        * LLEvolutions: The evolution of the log-likelihood for all runs.
        """
        self._print("Fitting ISAR Model using Expectation Maximisation")

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

        # Indicate Start of Fit
        self._print(
            "Running {0} starts for (max) {1} iterations.".format(num_work, self.__max_iter)
        )
        self.start_timer("global")

        # First Generate the M_Xi message
        self._write("Generating Latent-State Message (M_Omega)")
        self._flush()
        self.start_timer("m_omega")
        _M_Omega = self._compute_omega_msg(self.Omega, Y, S)
        self.stop_timer("m_omega")
        self._print("... Done")

        # Now Run EM on the Data
        self._write("Running EM:\n")
        self.start_timer("em")
        results = self.run_workers(
            num_work,
            self._EMWorker,
            _configs=self._EMWorker.ComputeParams_t(
                max_iter=self.__max_iter,
                tolerance=self.__conv_toler,
                m_omega=_M_Omega,
                prior_pi=priors[0],
                prior_psi=priors[1],
                mem=self.__jobs,
            ),
            _args=starts,
        )
        # Stop Timers
        self.stop_timer("em")
        self.stop_timer("global")

        if results is not None:
            self.Pi = results.Pi
            self.Psi = results.Psi

            # Display some statistics
            self._write("ISAR Model was fit in {0:1.5f}s of which:\n".format(self.elapsed("global")))
            self._write(
                "\t\tGenerating Emission Messages : {0:1.3f}s\n".format(self.elapsed("m_omega"))
            )
            self._print(
                "\t\tExpectation Maximisation     : {0:1.3f}s ({1:1.5f}s/run)".format(
                    self.elapsed("em"), self.elapsed("em") / num_work
                )
            )

        # Build (and return) Information Structure
        return (self, results) if return_diagnostics else self

    def predict(self, Y, S):
        """
        Predict the latent state Z given the observations (Y/S). Note that because of the prediction
        tolerance, the returned values are float.

        :param Y: Annotator Labels: np.NaN if unlabelled [N by |K|]
        :param S: The Schema used by the Annotator [N by |K|] or [N] if S is common per annotator.
        :return:  The predictions over Z
        """
        resp = self._EMWorker._compute_responsibilities(
            self._compute_omega_msg(self.Omega, Y, S), self.Pi, self.Psi, self.__max_mem
        ).gamma
        pred = np.argmax(resp, axis=-1).astype(float)
        pred[resp[np.arange(Y.shape[0]), pred.astype(int)] < self.__pred_toler] = np.NaN
        return pred

    def predict_proba(self, Y, S):
        """
        Compute the MAP estimate for (Z|Y, S)

        :param Y: Annotator Labels: np.NaN if unlabelled [N by |K|]
        :param S: The Schema used by the Annotator [N by |K|] or [N] if S is common per annotator.
        :return:  The posterior over Z (N by |Z|)
        """
        return self._EMWorker._compute_responsibilities(
            self._compute_omega_msg(self.Omega, Y, S), self.Pi, self.Psi, self.__max_mem
        ).gamma

    def evidence_log_likelihood(self, Y, S, prior=None):
        """
        Compute the (Marginalised Z) Evidence Log-Likelihood of the model on the data

        :param Y: Annotator Labels: np.NaN if unlabelled [N by |K|]
        :param S: The Schema used by the Annotator [N by |K|] or [N] if S is common per annotator.
        :param prior: Any prior probabilities on parameters. Can be None
        :return:  Log-Likelihood
        """
        m_omega = self._compute_omega_msg(self.Omega, Y, S)
        if prior is not None:
            return (
                npext.Dirichlet(prior[0]).logsumpdf(self.Pi)
                + npext.Dirichlet(prior[1]).logsumpdf(self.Psi)
                + self._EMWorker._compute_responsibilities(
                    m_omega, self.Pi, self.Psi, self.__max_mem
                ).log_likelihood
            )
        else:
            return self._EMWorker._compute_responsibilities(
                m_omega, self.Pi, self.Psi, self.__max_mem
            ).log_likelihood

    def _aggregate_results(self, results):
        """
        Maximise parameters over runs:

        :param results: the ApplyResult object
        :return: NamedTuple of type WorkerResults:
                    > Pi:               Best Pi Vector
                    > Psi:              Best Psi Tensor
                    > LogLikelihood:    Best Log-Likelihood
                    > Converged:        Boolean Array of which runs converged
                    > Best:             Index of the Best Run (absolute)
        """

        # Initialise Placeholders
        _final_pis = []  # Final Pi for each EM run
        _final_psis = []  # Final Psi for each EM run
        _final_llik = []  # Final Likelihood, for each EM run
        _evol_llik = []  # List of Likelihood Evolutions (for debugging)
        _converged = []  # Whether the run converged:

        # Iterate over results from each worker:
        for result in results:
            _final_pis.append(result.Pi)
            _final_psis.append(result.Psi)
            _final_llik.append(result.LLEvolutions[-1])
            _evol_llik.append(result.LLEvolutions)
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
        return self.ISARIIDResult_t(
            Pi=_final_pis[_best_index],
            Psi=_final_psis[_best_index],
            Best=_best_index,
            Converged=np.asarray(_converged),
            LLEvolutions=_evol_llik,
        )

    @staticmethod
    def _compute_omega_msg(omega, Y, S):
        """
        Generate M_Omega Message:

        :param omega: Schema-Specific Emission Probabilities [|S| by |U| by |Y|]
        :param Y: Annotator Labels: np.NaN if unlabelled [N by |K|]
        :param S: The Schema used by the Annotator [N by |K|] or [N] if S is common per annotator.
        :return:  N by K by |U| matrix
        """
        # Get the Dimensionality of the Problem
        sN, sK = Y.shape
        sU = omega.shape[1]
        sch_per_ann = len(S.shape) == 2
        m_omega = np.empty([sN, sK, sU])

        # Now iterate over samples
        if sch_per_ann:
            InterSchemaAdapteRIID.__compute_omega_sperk(Y, S, omega.astype(float), m_omega)
        else:
            InterSchemaAdapteRIID.__compute_omega_same(Y, S, omega.astype(float), m_omega)

        # Return
        return m_omega

    @staticmethod
    @jit(
        signature_or_function=(float64[:, :], int64[:, :], float64[:, :, :], float64[:, :, :]),
        nopython=True,
    )
    def __compute_omega_sperk(Y, S, omega, m_omega):
        """
        JIT Compiled computation of Omega Matrix [Case 1: Independent Schema per Annotator]

        :param Y:      Annotator Labels: np.NaN if unlabelled [N by |K|]
        :param S:      The Schema used by the Annotator [N by |K|]
        :param omega:  The omega mapper
        :param m_omega: Placeholder where to return
        :return:
        """
        sN, sK, sU = m_omega.shape
        for n in range(sN):
            for k in range(sK):
                m_omega[n, k, :] = (
                    np.ones(sU) if np.isnan(Y[n, k]) else omega[int(S[n, k]), :, int(Y[n, k])]
                )

    @staticmethod
    @jit(
        signature_or_function=(float64[:, :], int64[:], float64[:, :, :], float64[:, :, :]),
        nopython=True,
    )
    def __compute_omega_same(Y, S, omega, m_omega):
        """
        JIT Compiled computation of Omega Matrix [Case 2: Same Schema per Annotator]

        :param Y:      Annotator Labels: np.NaN if unlabelled [N by |K|]
        :param S:      The Schema used by the Annotator [N]
        :param omega:  The omega mapper
        :param m_omega: Placeholder where to return
        :return:
        """
        sN, sK, sU = m_omega.shape
        for n in range(sN):
            for k in range(sK):
                m_omega[n, k, :] = (
                    np.ones(sU) if np.isnan(Y[n, k]) else omega[int(S[n]), :, int(Y[n, k])]
                )

    @staticmethod
    def __generate_samples_same(n_runs, n_times, pA, psi, omega, Z, S, A, Y, U):
        """
        Wrapper Function for generating samples, in case random number generation will be
        supported by Numba in the future.
        """
        # Compute some Sizes
        sZU, sK, _ = psi.shape
        nA = A.shape[1]

        # Iterate over all segments
        for n in range(n_runs):
            # Pick Annotators for this segment
            A[n * n_times : (n + 1) * n_times, :] = np.random.choice(
                sK, size=nA, replace=False, p=pA
            )
            # Iterate over time-instances in this Segment
            for nt in range(n * n_times, (n + 1) * n_times):
                for k in A[nt]:  # Iterate over Annotators for this segment
                    u_k = np.random.choice(
                        sZU, p=psi[Z[nt], k, :]
                    )  # Compute Annotator Emission (confusion)
                    Y[nt, k] = (
                        u_k if omega[S[nt], u_k, u_k] == 1 else sZU
                    )  # Project Observation or NIS
                    if U is not None:
                        U[nt, k] = u_k

    @staticmethod
    def __generate_samples_different(n_runs, n_times, pA, psi, omega, Z, S, A, Y, U):
        """
        Wrapper Function for generating samples, in case it will be supported by Numba in the future. This variant
        supports using a different schema per-annotator.

        :param n_runs:
        :param n_times:
        :param pA:
        :param psi:
        :param omega:
        :param Z:
        :param S:
        :param A:
        :param Y:
        :return:
        """
        # Compute some Sizes
        sZU, sK, _ = psi.shape
        nA = A.shape[1]

        # Iterate over all segments
        for n in range(n_runs):
            # Pick Annotators for this segment
            A[n * n_times : (n + 1) * n_times, :] = np.random.choice(
                sK, size=nA, replace=False, p=pA
            )
            # Iterate over time-instances in this Segment
            for nt in range(n * n_times, (n + 1) * n_times):
                for k in A[nt]:  # Iterate over Annotators for this segment
                    u_k = np.random.choice(
                        sZU, p=psi[Z[nt], k, :]
                    )  # Compute Annotator Emission (confusion)
                    Y[nt, k] = (
                        u_k if (omega[S[nt, k], int(u_k), int(u_k)] == 1) else sZU
                    )  # Project Observation or NIS
                    if U is not None:
                        U[nt, k] = u_k

    # ========================== Private Nested Implementations ========================== #
    class _EMWorker(IWorker):
        """
        (Private) Nested class for running the EM Algorithm
        """

        # Definition of Named Data-Types
        ComputeParams_t = namedtuple(
            "ComputeParams_t", ["max_iter", "tolerance", "m_omega", "prior_pi", "prior_psi", "mem"]
        )
        Responsibilities_t = namedtuple(
            "Responsibilities_t", ["gamma", "rho_sum", "log_likelihood"]
        )
        ComputeResult_t = namedtuple("ComputeResult_t", ["Pi", "Psi", "Converged", "LLEvolutions"])

        def __init__(self, _id, _handler):
            """
            Initialiser

            :param _id:         Identifier
            :param _handler:    The Worker Handler
            """
            # Initialise Super-Class
            super(InterSchemaAdapteRIID._EMWorker, self).__init__(_id, _handler)

        def parallel_compute(self, _common: ComputeParams_t, _data):
            """
            Implementation of Parallel Computation
            :param _common: NamedTuple of type ComputeCommon
                            > max_iter:     Maximum number of iterations to compute for
                            > tolerance:    Tolerance Parameter for early stopping: if Likelihood does not change more
                                            than this amount between Iterations, then stop.
                            > m_omega:      The m_omega message, of size N by K by |U|
                            > prior_pi:     Prior (Smoothing) over the Latent Distribution Pi, size |Z|. Should be raw
                                            counts (equal to [alpha-1] in the Dirichlet sense).
                            > prior_psi:    Prior (Smoothing) over the Inter-Annotator Emission, size |Z| x K x |U|.
                                            Again, should be raw counts (equal to [alpha-1] in the Dirichlet sense).
            :param _data:   Tuple of starting points for Pi/Psi (in order)
            :return: NamedTuple of type ComputeResult:
                            > Pi (optimised): size |Z|
                            > Psi (optimised): size |Z| by K by |U|
                            > Likelihood at each iteration: Note that this is the Expected Completed Penalised
                              log-likelihood
                            > Whether converged or not within max-iterations
            """
            # Initialise Starting Points
            pi, psi = _data

            # Initialise some precomputations
            pi_dir = npext.Dirichlet(_common.prior_pi, ignore_zeros=False)
            psi_dir = npext.Dirichlet(_common.prior_psi, ignore_zeros=False)

            # Prepare for Loop
            iterations = 0  # Iteration Count
            loglikelihood = []  # Observed Data Log-Likelihood (evolutions)

            # Start the Loop: we make heavy use of Message Passing (Pg 23)
            #   Note that due this is optimised so that we minimise repeated operations, especially since the E-Step is
            #       also conducive to computing the log-likelihood. It also ensures that we store the first likelihood
            #       at the start (before doing the first maximisation) and the likelihood computation is consistent. As
            #       a side-effect, the E-Step will be computed once more than the M-step, but its update will not effect
            #       the parameters, since we break if the likelihood did not change! It also implies that if the system
            #       is already at the local optimum, it should not change much from it!
            while iterations < _common.max_iter:
                # -------------- E-Step: -------------- #
                # ++++ Compute Responsibilities ++++ #
                msg = self._compute_responsibilities(_common.m_omega, pi, psi, _common.mem)
                # ++++ Now compute log-likelihood ++++ #
                loglikelihood.append(
                    pi_dir.logpdf(pi) + np.sum(psi_dir.logpdf(psi)) + msg.log_likelihood
                )
                # --------- Likelihood-Check: --------- #
                # Check for convergence!
                if self._converged(loglikelihood, _common.tolerance):
                    break
                # -------------- M-Step: -------------- #
                # Now Compute Pi
                pi = npext.sum_to_one(
                    np.sum(msg.gamma, axis=0).squeeze() + (_common.prior_pi - 1), axis=-1
                )
                # Finally Compute Psi
                psi = npext.sum_to_one(msg.rho_sum + (_common.prior_psi - 1), axis=-1)

                # -------------- Iteration Control -------------- #
                iterations += 1
                self.update_progress(iterations * 100.0 / _common.max_iter)

            # Clean up
            self.update_progress(100.0)
            converged = self._converged(loglikelihood, _common.tolerance)

            # Return Result
            return self.ComputeResult_t(
                Pi=pi, Psi=psi, Converged=converged, LLEvolutions=loglikelihood
            )

        @staticmethod
        def _compute_responsibilities(m_omega: np.ndarray, pi, psi, mm):
            """
            Compute the Responsibilities and subsequently the Log-Likelihood

            :param m_omega: Omega Message
            :param pi:      Latent Probabilities
            :param psi:     Emission Probabilities
            :param mm:      Maximum Memory to operate with
            :return:        Responsibilities_t Named Tuple
            """
            # ---- Define the Sizes we will work with ---- #
            sZ, sK, sU = psi.shape
            sN = m_omega.shape[0]

            # ---- Prepare Space ---- #
            rho_sum = np.zeros_like(psi)
            gamma = np.empty([sN, sZ])
            log_likelihood = 0
            pi = pi[np.newaxis, :, np.newaxis, np.newaxis]
            psi = psi[np.newaxis, :, :, :]

            # ---- Now Iterate over N ---- #
            cN = 0
            while cN < sN:
                # ++++ Identify indices to work with ++++ #
                max_idx = min(cN + mm, sN)
                omega_subset = m_omega[cN:max_idx, np.newaxis, :, :]
                # ++++ Compute Messages ++++ #
                m_omega_star = np.multiply(omega_subset, psi)  # [N, |Z|, K, |U|]
                m_omega_star_sum_u = np.sum(m_omega_star, axis=3, keepdims=True)  # [N, |Z|, K,  1 ]
                m_psi_pi = np.multiply(
                    np.prod(m_omega_star_sum_u, axis=2, keepdims=True), pi
                )  # [N, |Z|, 1,  1 ]
                m_psi_pi_sum_z = np.sum(m_psi_pi, axis=1, keepdims=True)  # [N,  1,  1,  1 ]
                # ++++ Update Gamma ++++ #
                gamma_subset = np.divide(m_psi_pi, m_psi_pi_sum_z)
                gamma[cN:max_idx, :] = gamma_subset.squeeze()
                # ++++ Update Rho_sum ++++ #
                rho_sum += np.multiply(
                    np.divide(gamma_subset, m_omega_star_sum_u), m_omega_star
                ).sum(axis=0)
                # ++++ Update Log-Likelihood ++++ #
                log_likelihood += np.sum(np.log(m_psi_pi_sum_z))
                # ++++ Update Indexing ++++ #
                cN = max_idx

            # ---- Return the Result ---- #
            return InterSchemaAdapteRIID._EMWorker.Responsibilities_t(
                gamma, rho_sum, log_likelihood
            )

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
