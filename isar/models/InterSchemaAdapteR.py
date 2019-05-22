"""
This is the ISAR Model, applied to the Crowd-sourcing task where P(U|Z) is a mixture of categorical distributions
according to the Dawid-Skene Model.

TODO: Use JiT Compilation on some of the for-loops

N.B. This code is copied into my M.Sc Project -- If I do any changes, I should also change there.
"""
from mpctools.parallel import WorkerHandler, IWorker
from mpctools.extensions import npext
from collections import namedtuple
import numpy as np
import warnings

MAX_MEMORY = 200000


class AnnotISAR(WorkerHandler):
    """
    This Class implements the Inter-Schema Adapter model, based on a mixture of categorical distributions,
    which are in turn emitted via a deterministic schema translation. It is implemented in a multi-processing framework
    allowing multiple independent restarts to happen in parallel.

    Refer to the Paper, esp Appendix for derivations/definitions of the equations
    """
    # ============================== Data Type Definitions =============================== #
    WorkerResults_t = namedtuple('WorkerResults_t', ['Pi', 'Psi', 'LogLikelihood', 'Converged', 'Best', 'LLEvolutions'])
    AISARResults_t = namedtuple('AISARResults_t', ['ModelDims', 'DataDims', 'Pi', 'Psi', 'BestRun', 'Converged',
                                                   'LogLikelihood', 'Times', 'LLEvolutions'])

    # ========================== Private Nested Implementations ========================== #
    class EMWorker(IWorker):
        """
        (Private) Nested class for running the EM Algorithm
        """
        # Definition of Named Data-Types
        ComputeCommon_t = namedtuple('ComputeParams_t', ['max_iter', 'tolerance', 'm_omega', 'prior_pi', 'prior_psi'])
        Responsibilities_t = namedtuple('Responsibilities_t', ['gamma', 'rho_sum', 'log_likelihood'])
        ComputeResult_t = namedtuple('ComputeResult_t', ['Pi', 'Psi', 'LogLikelihood', 'Converged', 'LLEvolutions'])

        def __init__(self, _id, _handler):
            """
            Initialiser

            :param _id:         Identifier
            :param _handler:    The Worker Handler
            """
            # Initialise Super-Class
            super(AnnotISAR.EMWorker, self).__init__(_id, _handler)

        def parallel_compute(self, _common: ComputeCommon_t, _data):
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
                            > Likelihood at each iteration: Note that this is the Expected Completed Penalised log-likelihood
                            > Whether converged or not within max-iterations
            """
            # Initialise parameters (and compute lengths)
            sN, sK, sU = _common.m_omega.shape

            # Initialise Starting Points
            pi, psi = _data

            # Initialise some precomputations
            pi_denom = np.sum(_common.prior_pi) + sN                      # Denominator for Phi
            psi_den_p = np.sum(_common.prior_psi, axis=2, keepdims=True)  # Partial Denominator for Psi [|Z|, K, 1]
            pi_dir = npext.Dirichlet(_common.prior_pi + 1.0)
            psi_dir = npext.Dirichlet(_common.prior_psi + 1.0)

            # Prepare for Loop
            iterations = 0        # Iteration Count
            loglikelihood = []    # Observed Data Log-Likelihood (evolutions)

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
                msg = self._compute_responsibilities(_common.m_omega, pi, psi)
                # ++++ Now compute log-likelihood ++++ #
                loglikelihood.append(pi_dir.logpdf(pi) + np.sum(psi_dir.logpdf(psi)) + msg.log_likelihood)

                # --------- Likelihood-Check: --------- #
                # Check for convergence!
                if self._converged(loglikelihood, _common.tolerance):
                    break

                # -------------- M-Step: -------------- #
                # Now Compute Pi
                pi = (np.sum(msg.gamma, axis=0).squeeze() + _common.prior_pi)/pi_denom
                # Finally Compute Psi
                psi = np.divide(msg.rho_sum + _common.prior_psi, np.sum(msg.rho_sum, axis=2, keepdims=True) + psi_den_p)

                # -------------- Iteration Control -------------- #
                iterations += 1
                self.update_progress(iterations * 100.0 / _common.max_iter)

            # Clean up
            self.update_progress(100.0)
            converged = self._converged(loglikelihood, _common.tolerance)

            # Return Result
            return self.ComputeResult_t(Pi=pi, Psi=psi, LogLikelihood=loglikelihood[-1], Converged=converged,
                                        LLEvolutions=loglikelihood)

        @staticmethod
        def _compute_responsibilities(m_omega: np.ndarray, pi, psi):
            """
            Compute the Responsibilities and subsequently the Log-Likelihood

            :param m_omega: Omega Message
            :param pi:      Latent Probabilities
            :param psi:     Emission Probabilities
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
                max_idx = min(cN + MAX_MEMORY, sN)
                omega_subset = m_omega[cN:max_idx, np.newaxis, :, :]

                # ++++ Compute Messages ++++ #
                m_omega_star = np.multiply(omega_subset, psi)                                   # [N, |Z|, K, |U|]
                m_omega_star_sum_u = np.sum(m_omega_star, axis=3, keepdims=True)                # [N, |Z|, K,  1 ]
                m_psi_pi = np.multiply(np.prod(m_omega_star_sum_u, axis=2, keepdims=True), pi)  # [N, |Z|, 1,  1 ]
                m_psi_pi_sum_z = np.sum(m_psi_pi, axis=1, keepdims=True)                        # [N,  1,  1,  1 ]

                # ++++ Update Gamma ++++ #
                gamma_subset = np.divide(m_psi_pi, m_psi_pi_sum_z)
                gamma[cN:max_idx, :] = gamma_subset.squeeze()

                # ++++ Update Rho_sum ++++ #
                rho_sum += np.multiply(np.divide(gamma_subset, m_omega_star_sum_u), m_omega_star).sum(axis=0)

                # ++++ Update Log-Likelihood ++++ #
                log_likelihood += np.sum(np.log(m_psi_pi_sum_z))

                # ++++ Update Indexing ++++ #
                cN = max_idx

            # ---- Return the Result ---- #
            return AnnotISAR.EMWorker.Responsibilities_t(gamma, rho_sum, log_likelihood)

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
                warnings.warn('Drop in Log-Likelihood Observed! Results are probably wrong.')
            else:
                return abs((likelihoods[-1] - likelihoods[-2])/likelihoods[-2]) < tolerance

    # ========================== Initialisers ========================== #
    def __init__(self, omega, _num_proc, _max_iter, _toler=1e-4, sink=None):
        """
        Initialiser

        :param omega:       Fixed ISAR Matrix. This must be an NdArray of size |S| by |U| by |X|
        :param _num_proc:   Number of processes to run in - see MultiProgramming
        :param _max_iter:   Maximum Number of iterations in EM
        :param _toler:      Tolerance for convergence
        :param sink:        (Optional) Sink for debug output
        """

        # Call Super-Class
        super(AnnotISAR, self).__init__(_num_proc, sink)

        # Initialise own-stuff
        self.Omega = omega
        self.__max_iter = _max_iter
        self.__toler = _toler

    def fit_model(self, Y, S, prior, _starts=1):
        """
        Fit the Parameters Pi/Psi to the data, and generate MAP estimates for the latent behaviour:

        :param Y:        Annotator provided labels [N by |K|]
        :param S:        Schema for each sample (one-hot encoded): [N]
        :param prior:    Prior Probabilities for Pi and Psi (|Z|, [|Z| by K by |U|]): Note these must be raw (smoothing)
                            counts (equivalent to [alpha-1] in Dirichlet prior formulation)
        :param _starts:  This can be either:
                            * Integer - Number of random starts to perform
                            * List of Starting points (each starting point should be a tuple/list, containing the
                              starting pi/psi matrices. N.B. These will be modified (so pass a copy if need to preserve)
        :return:    NamedTuple of Type ISACResults
                        * ModelDims: Model Dimensions [|Z|, K, |S|, |X|]
                        * DataDims:  Data Dimensions [N]
                        * Pi:  Latent Probabilities
                        * Psi:  Emission Probabilities
                        * BestRun: Index of the best-run
                        * Converged: Which runs (boolean array) converged
                        * LogLikelihood: the Penalised Observed Data Log-Likelihood at end of the procedure (best one)
                        * Times: Times to fit the model (dictionary)
        """
        # Handle Starts
        if hasattr(_starts, '__len__'):
            workers = len(_starts)
        else:
            workers = _starts
            _starts = None

        # Indicate Start of Fit
        self._print('Fitting ISAR Model on Data with {0} starts, each running for (max) {1} iterations.'
                    .format(workers, self.__max_iter))
        self.start_timer('global')

        # First Generate the M_Xi message
        self._write('Generating Latent-State Message (M_Omega)')
        self._flush()
        self.start_timer('m_omega')
        _M_Omega = self.omega_msg(self.Omega, Y, S)
        self.stop_timer('m_omega')
        self._print('... Done')

        # Now Run EM on the Data
        self._write('Running EM:\n')
        self.start_timer('em')
        results = self.run_workers(workers, self.EMWorker,
                                  _configs=self.EMWorker.ComputeCommon_t(max_iter=self.__max_iter,
                                                                         tolerance=self.__toler, m_omega=_M_Omega,
                                                                         prior_pi=prior[0], prior_psi=prior[1]),
                                  _args=_starts)
        self.stop_timer('em')

        # Stop Main timer
        self.stop_timer('global')

        # Display some statistics
        self._write('ISAR Model was fit in {0:1.5f}s of which:\n'.format(self.elapsed('global')))
        self._write('\t\tGenerating Emission Messages : {0:1.3f}s\n'.format(self.elapsed('m_omega')))
        self._print('\t\tExpectation Maximisation     : {0:1.3f}s ({1:1.5f}s/run)'.format(self.elapsed('em'),
                                                                                          self.elapsed('em')/workers))

        # Build (and return) Information Structure
        return self.AISARResults_t(ModelDims=[len(prior[0]), _M_Omega.shape[1], self.Omega.shape[0], self.Omega.shape[2]],
                                   DataDims=len(Y), Pi=results.Pi, Psi=results.Psi, BestRun=results.Best,
                                   Converged=results.Converged, LogLikelihood=results.LogLikelihood,
                                   Times={'Total': self.elapsed('global'), 'Message': self.elapsed('m_omega'),
                                          'EM': self.elapsed('em')}, LLEvolutions=results.LLEvolutions)

    def aggregate_results(self, results):
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
        _final_pis  = []  # Final Pi for each EM run
        _final_psis = []  # Final Psi for each EM run
        _final_llik = []  # Final Likelihood, for each EM run
        _evol_llik = []   # List of Likelihood Evolutions (for debugging)
        _converged  = []  # Whether the run converged:

        # Iterate over results from each worker:
        for result in results:
            _final_pis.append(result.Pi)
            _final_psis.append(result.Psi)
            _final_llik.append(result.LogLikelihood)
            _evol_llik.append(result.LLEvolutions)
            _converged.append(result.Converged)

        # Convert to Numpy Arrays for Indexing
        _final_llik = np.asarray(_final_llik)
        _final_pis = np.asarray(_final_pis)
        _final_psis = np.asarray(_final_psis)

        # Check whether at least one converged, and warn if not...
        if np.any(_converged):
            _masked_likelihood = np.ma.masked_array(_final_llik, np.logical_not(_converged))
        else:
            warnings.warn('None of the Runs Converged: results may be incomplete')
            _masked_likelihood = _final_llik

        # Find the best one out of those converged, or out of all, if none converged
        _best_index = _masked_likelihood.argmax().squeeze()

        # Return Results in Dictionary:
        return self.WorkerResults_t(Pi=_final_pis[_best_index], Psi=_final_psis[_best_index], Best=_best_index,
                                    LogLikelihood=_final_llik[_best_index], Converged=np.asarray(_converged),
                                    LLEvolutions=_evol_llik)

    @staticmethod
    def omega_msg(omega, Y, S):
        """
        Generate M_Omega Message:

        :param omega:  Schema-Specific Emission Probabilities [|S| by |U| by |Y|]
        :param Y:      Annotator Labels: np.NaN if unlabelled [N by |K|]
        :param S:      The Schema used by the Annotator [N by |K|] or [N] if S is common per annotator.
        :return:       N by K by |U| matrix
        """
        # Get the Dimensionality of the Problem
        sN, sK = Y.shape
        sU = omega.shape[1]
        sch_per_ann = (len(S.shape) == 2)
        # Now iterate over samples
        m_omega = np.empty([sN, sK, sU])
        if sch_per_ann:
            for n in range(sN):
                for k in range(sK):
                    m_omega[n, k, :] = np.ones(sU) if np.isnan(Y[n, k]) else omega[int(S[n, k]), :, int(Y[n, k])]
        else:
            for n in range(sN):
                for k in range(sK):
                    m_omega[n, k, :] = np.ones(sU) if np.isnan(Y[n, k]) else omega[int(S[n]), :, int(Y[n, k])]

        # Return
        return m_omega

    @staticmethod
    def estimate_map(pi, psi, m_omega, label_set):
        """
        Compute Predictions (most probable, based on MAP) for the latent states given the observations

        :param pi:         Latent distribution
        :param psi:        Class Conditional Densities
        :param m_omega:    M_Omega
        :param label_set:  The original (ordered) label set for the latent states (i.e. actual values)
        :return:           Maximum a Posteriori latent state (if label_set is a list/tuple), otherwise (if None),
                            returns the conditional probability distribution
        """
        # Compute Messages and Responsibility
        msg = AnnotISAR.EMWorker._compute_responsibilities(m_omega, pi, psi)

        # Branch on whether to compute MAP or just output probabilities
        if label_set is not None:
            map_raw = np.argmax(msg.gamma, axis=1)
            return npext.value_map(map_raw, label_set, shuffle=True)  # Transform into original indexing (via label_set)
        else:
            return npext.sum_to_one(msg.gamma, axis=1)                # Return Normalised Probabilities

    @staticmethod
    def data_loglikelihood(theta, prior, hot, sch):
        """
        Compute the Observed Data Log-Likelihood

        :param theta:   Parameters (Pi, Psi, Omega)
        :param prior:   Priors over (Pi, Psi) - these would be already discounted counts ([alpha-1])! If not passed,
                            then penalisation is not added.
        :param hot:     1-Hot Encoded Data
        :param sch:     1-Hot Encoded Schema
        :return:
        """
        _m_omega = AnnotISAR.omega_msg(theta[2], hot, sch)
        msg = AnnotISAR.EMWorker._compute_responsibilities(_m_omega, theta[0], theta[1])
        if prior is not None:
            return npext.Dirichlet(prior[0] + 1.0).logpdf(theta[0]) + \
                   np.sum(npext.Dirichlet(prior[1] + 1.0).logpdf(theta[1])) + msg.log_likelihood
        else:
            return msg.log_likelihood