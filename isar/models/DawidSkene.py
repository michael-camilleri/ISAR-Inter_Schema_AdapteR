"""
This is the Dawid-Skene Model, for comparing the ISAR model against.
"""
from mpctools.parallel import IWorker, WorkerHandler
from mpctools.extensions import npext
from collections import namedtuple
from numba import jit, float64
import numpy as np
import time as tm
import warnings
import copy


class DawidSkeneIID(WorkerHandler):
    """
    This Class implements the Noisy Annotator Model, following the Formulation of Dawid-Skene in an IID fashion.
    """
    DSIIDResult_t = namedtuple('DSIIDResult_t', ['Pi', 'Psi', 'Converged', 'Best', 'LLEvolutions'])

    # ========================== Initialisers ========================== #
    def __init__(self, dims, params, max_iter=100, tol=1e-4, n_jobs=-1, random_state=None, sink=None):
        """
        Initialiser

        :param dims:            The Dimensions of the Model, basically Z/U, K (See Paper: We assume Z == U)
        :param params:          If not None, specifies a value for the parameters: otherwise, these are initialised to
                                completely perfect distributions (i.e. perfect annotators)
        :param max_iter:        Maximum number of iterations in EM computation
        :param tol:             Tolerance for convergence in EM Computation
        :param n_jobs:          Number of jobs: see documentation for mpctools.WorkerHandler
        :param random_state:    If set, ensures reproducible results
        :param sink:            For debugging, outputs progress etc..
        """

        # Call Super-Class
        super(DawidSkeneIID, self).__init__(n_jobs, sink)

        # Initialise own-stuff
        self.sZU, self.sK = dims
        self.__rand = random_state if random_state is not None else int(tm.time())
        self.__max_iter = max_iter
        self.__toler = tol

        # Prepare a valid probability for the model
        if params is None:
            self.Pi = np.full(self.sZU, 1/self.sZU)
            self.Psi = np.tile(np.eye(self.sZU)[np.newaxis, :, :], [self.sK, 1, 1])
        else:
            self.Pi, self.Psi = params

    def fit(self, U, priors, starts, return_diagnostics=False):
        """
        Fit the Parameters Pi/Psi to the data, and generate MAP estimates for the latent behaviour:

        :param U:        Observations:  Size N x K. Note that where the annotator does not label a sample, this should
                         be represented by NaN
        :param priors:   Prior Probabilities for Pi and Psi [|Z|, |Z| x K x |U|]
        :param starts:   This must be the list of Starting points (each starting point should be a tuple/list,
                              containing the starting pi/psi matrices). Note that in this case, they may be modified.
        :param return_diagnostics: If true, return a named tuple of type DSIIDResult_t containing diagnostic information
        :return:  Self, (Diagnostics) : Self for chaining, Diagnostics if requested. These contain:
                        * Pi : Fitted Prior over Targets
                        * Psi: Annotator Confusion Matrix
                        * Converged: Which runs in the EM scheme converged
                        * Best: The index of the run with the best log-likelihood
                        * LLEvolutions: The evolution of the log-likelihood for all runs.
        """
        # Handle Starts
        num_work = len(starts)

        # Initialise some stuff...
        self._print('Fitting EM Model on Data with {0} starts, each running for (max) {1} iterations.'
                    .format(num_work, self.__max_iter))
        self.start_timer('global')

        # Perform EM (and format output)
        self._write('Running EM:\n')
        self.start_timer('em')
        results = self.run_workers(num_work, self.EMWorker,
                                   _configs=self.EMWorker.ComputeParams_t(self.__max_iter, self.__toler, U, *priors),
                                   _args=starts)
        self.stop_timer('em')

        # Consolidate Data
        self.Pi = results.Pi
        self.Psi = results.Psi

        # Stop Main timer
        self.stop_timer('global')

        # Display some statistics
        self._write('DS Model was fit in {0:1.5f}s of which:\n'.format(self.elapsed('global')))
        self._print('\t\tExpectation Maximisation   : {0:1.3f}s ({1:1.5f}s/run)'.format(self.elapsed('em'),
                                                                                        self.elapsed('em')/num_work))

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
            return npext.Dirichlet(prior[0]).logsumpdf(self.Pi) + npext.Dirichlet(prior[1]).logsumpdf(self.Psi) + \
                   self.EMWorker._compute_responsibilities(U, self.Pi, self.Psi).log_likelihood
        else:
            return self.EMWorker._compute_responsibilities(U, self.Pi, self.Psi).log_likelihood

    def predict(self, U):
        """
        Predict the latent state Z given the observations

        :param U:  Observations: Size N by K (of domain |U| with missing data as NaN)
        :return:   Predictions over the latent states
        """
        return np.argmax(self.EMWorker._compute_responsibilities(U, self.Pi, self.Psi).gamma, axis=-1)

    def predict_proba(self, U):
        """
        Get the distribution over the latent state Z.

        :param U:  Observations: Size N by K (of domain |U| with missing data as NaN)
        :return:   Distribution over latent state Z
        """
        return self.EMWorker._compute_responsibilities(U, self.Pi, self.Psi).gamma

    def aggregate_results(self, results):
        """
        Maximise parameters over runs: for comparison's sake, it assumes that the pi's and psi's are sorted in a
        consistent order (for comparison).

        :param results: the ApplyResult object
        :return: Dictionary, containing all parameters
        """

        # Initialise Placeholders
        _final_pis  = []  # Final Pi for each EM run
        _final_psis = []  # Final Psi for each EM run
        _final_llik = []  # Final Likelihood, for each EM run
        _converged  = []  # Whether the run converged:
        _evol_llikel = [] # Evolutons of log-likelihoods

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

        # Check whether at least one converged, and warn if not...
        if np.any(_converged):
            _masked_likelihood = np.ma.masked_array(_final_llik, np.logical_not(_converged))
        else:
            warnings.warn('None of the Runs Converged: results may be incomplete')
            _masked_likelihood = _final_llik

        # Find the best one out of those converged, or out of all, if none converged
        _best_index = _masked_likelihood.argmax().squeeze()
        _best_llikel = _final_llik[_best_index]
        _best_pi = _final_pis[_best_index]
        _best_psi = _final_psis[_best_index]

        # Return Results in Dictionary:
        return self.DSIIDResult_t(_best_pi, _best_psi, _converged, _best_index, _evol_llikel)

    # @staticmethod
    # def estimate_map(pi, psi, u, label_set, max_only=False, max_size=None):
    #     """
    #     Compute Predictions (most probable, based on MAP) for the latent states given the observations
    #
    #     :param pi:          Latent distribution
    #     :param psi:         Class Conditional Densities
    #     :param u:           Observations (One-Hot Encoding, size [N by K*|U|])
    #     :param label_set:   The original label set for the latent states (i.e. actual values associated with each index
    #                         in the current ordering.
    #     :param max_only:    If True (default) just return the MAP estimate: otherwise, return the posterior probabilities
    #     :param max_size:    Must be set if max_only is False (otherwise ignored).
    #     :return:            Maximum a Posteriori latent state or the posterior probabilities
    #     """
    #     # Compute Posterior
    #     _posterior = np.multiply(np.prod(np.power(psi.T[np.newaxis, :, :], u[:, :, np.newaxis]), axis=1), pi[np.newaxis, :])
    #
    #     # Branch on whether to compute MAP or just output probabilities
    #     if max_only:
    #         return npext.value_map(np.argmax(_posterior, axis=1), label_set, shuffle=True)   # Map to original Label-Set
    #     else:
    #         _posterior = npext.sum_to_one(_posterior, axis=1)        # Normalised Probabilities
    #         _mapped_post = np.zeros([_posterior.shape[0], max_size]) # Placeholder for mapped posterior (mapped to actual value in label_set)
    #         for _i, _l in enumerate(label_set):
    #             _mapped_post[:, _l] = _posterior[:, _i]
    #         return _mapped_post

    # @staticmethod
    # def optimise_permutations(_map, _raw, _pi, _psi, _labels):
    #     """
    #     Optimise the permutation of the latent states (_map) such that there is least error (confusion matrix, highest
    #     agreement) with the raw data.
    #
    #     Note:
    #         * Method can handle disjoing label-_sets (through _labels) BUT they must be the same for _map and _raw
    #         * Method ignores NaN values in _raw (i.e. these do not contribute towards count)
    #
    #     :param _map:    MAP Estimates of the data [N]
    #     :param _raw:    Raw (floating point Numpy Array) data [N by K]
    #     :param _pi:     Pi Vector (to permute)
    #     :param _psi:    Psi Vector (to permute)
    #     :param _labels: Label-Set (for both)
    #     :return: Tuple containing in order:
    #                     * Updated Pi
    #                     * Updated Psi
    #                     * Updated MAP Predictions
    #     """
    #     # Get # Annotators
    #     _K = _raw.shape[1]
    #
    #     # Prepare Data
    #     _observed = _raw.astype(np.float).T.ravel() # Flatten out all annotators
    #     _valid = np.isfinite(_observed)             # Find out only those which have been labelled.
    #     _observed = _observed[_valid]               # Valid Observed
    #     _predicted = np.tile(_map, _K)[_valid]      # Valid Predicted
    #
    #     # Compute Confusion Matrix
    #     _confusion = confusion_matrix(_observed, _predicted, labels=_labels)
    #
    #     # Now Perform Hungarian Algorithm: we need to use the Transpose, since we want to find the row each map value
    #     #   should be assigned to.
    #     _best_permute, _ = npext.maximise_trace(_confusion.T)
    #
    #     # Finally, update Parameters
    #     _pi[list(_best_permute)] = _pi.copy()
    #     _psi[list(_best_permute)] = _psi.copy()
    #     _best_labels = _labels[_best_permute]
    #     return _pi, _psi, npext.value_map(_map, _best_labels, _labels), _best_permute

    # ========================== Private Nested Implementations ========================== #
    class EMWorker(IWorker):
        """
        (Private) Nested class for running the EM Algorithm

        Note that in this case, the index order is n,z,k,u (i.e. Sample, Latent, Annotator, Label)
        """
        ComputeParams_t = namedtuple('ComputeParams_t', ['max_iter', 'tolerance', 'U', 'prior_pi', 'prior_psi'])
        Responsibilities_t = namedtuple('Responsibilities_t', ['gamma', 'log_likelihood'])
        ComputeResult_t = namedtuple('ComputeResult_t', ['Pi', 'Psi', 'Converged', 'LLEvolutions'])

        def __init__(self, _id, _handler):
            """
            Initialiser

            :param _id:         Identifier - allows different seeds for random initialisation
            :param _handler:    The Worker Handler
            """
            # Initialise Super-Class
            super(DawidSkeneIID.EMWorker, self).__init__(_id, _handler)

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
                            > Psi (optimised): size |Z| by (K * |U|)
                            > Whether the run converged or not within max-iterations
                            > Evolution of (evidence) log-likelihood through iterations.
            """
            # Initialise parameters
            # max_iter, toler, sZ, U, pPi, pPsi, sort = _common

            # Initialise Random Points (start)
            pi, psi = _data
            sZ, sK, sU = psi.shape

            # Initialise Dirichlets
            pi_dir = npext.Dirichlet(_common.prior_pi)
            psi_dir = npext.Dirichlet(_common.prior_psi)

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
                loglikelihood.append(pi_dir.logpdf(pi) + psi_dir.logsumpdf(psi) + msg.log_likelihood)

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

            # # Optionally, sort the parameters in ascending probability for pi
            # if _common.sort:
            #     _sort_order = np.argsort(pi)
            #     pi = pi[_sort_order]
            #     psi = np.stack((psi[j] for j in _sort_order), axis=0)

            # Return Result
            return DawidSkeneIID.EMWorker.ComputeResult_t(pi, psi, converged, loglikelihood)

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
            DawidSkeneIID.EMWorker.__gamma(U, pi, psi, gamma)

            # Normalise to sum to 1, and at the same, through the normaliser, compute observed log-likelihood
            gamma, normaliser = npext.sum_to_one(gamma, axis=-1, norm=True)
            log_likelihood = -np.log(normaliser).sum()

            # Return
            return DawidSkeneIID.EMWorker.Responsibilities_t(gamma, log_likelihood)

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
                return abs((likelihoods[-1] - likelihoods[-2]) / likelihoods[-2]) < tolerance

        @staticmethod
        @jit(signature_or_function=(float64[:, :], float64[:], float64[:, :, :], float64[:, :]), nopython=True)
        def __gamma(U, pi, psi, gamma):
            """
            Convenience Wrapper for JIT compilation of Gamma Computation (unnormalised)

            :param U:       The Data, with Unlabelled/Missing data indicated by np.NaN
            :param pi:      The Prior Probabilities
            :param psi:     The Emission Probabilities
            :param gamma:   The gamma placeholder (modified in place)
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