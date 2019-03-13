"""
This is the ISAR Model, applied to the Crowd-sourcing task where P(U|Z) is a mixture of categorical distributions
according to the Dawid-Skene Model.
"""
from collections import namedtuple
import numpy as np
import warnings

from Tools import npext, WorkerHandler, NullableSink


class AnnotISAR(WorkerHandler):
    """
    This Class implements the Inter-Schema Adapter model, based on a mixture of categorical distributions,
    which are in turn emitted via a deterministic schema translation. It is implemented in a multi-processing framework
    allowing multiple independent restarts to happen in parallel.

    Refer to the Paper, esp Appendix for derivations/definitions of the equations
    """
    # ============================== Data Type Definitions =============================== #
    WorkerResults_t = namedtuple('WorkerResults_t', ['Pi', 'Psi', 'LogLikelihood', 'Converged', 'Best'])
    AISARResults_t = namedtuple('AISARResults_t', ['ModelDims', 'DataDims', 'Pi', 'Psi', 'BestRun', 'Converged',
                                                   'LogLikelihood', 'Times'])

    # ========================== Private Nested Implementations ========================== #
    class EMWorker(WorkerHandler.Worker):
        """
        (Private) Nested class for running the EM Algorithm
        """
        # Definition of Named Data-Types
        ComputeCommon_t = namedtuple('ComputeParams_t', ['max_iter', 'tolerance', 'm_omega', 'prior_pi', 'prior_psi'])
        Messages_t = namedtuple('Messages_t', ['m_omega_star', 'm_omega_star_sum_u', 'm_psi_pi', 'm_psi_pi_sum_z'])
        ComputeResult_t = namedtuple('ComputeResult_t', ['Pi', 'Psi', 'LogLikelihood', 'Converged'])

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
            # Note that we pre-compute the first set of messages, since due to efficiency, we compute the msgs at the
            #   end of the loop, since they are needed for the Likelihood Computation
            msg = self._compute_messages(_common.m_omega, pi, psi)

            while iterations < _common.max_iter and not self._converged(loglikelihood, _common.tolerance):
                # -------------- E-Step: -------------- #
                # First Compute Gamma
                gamma = np.divide(msg.m_psi_pi, msg.m_psi_pi_sum_z)                         # Gamma  [N, |Z|, 1,  1 ]
                # Now Compute Rho
                rho = np.multiply(np.divide(gamma, msg.m_psi_pi), msg.m_omega_star_sum_u)   # Rho    [N, |Z|, K, |U|]

                # -------------- M-Step: -------------- #
                # Now Compute Pi
                pi = (np.sum(gamma, axis=0).squeeze() + _common.prior_pi)/pi_denom
                # Finally Compute Psi
                rho_n = np.sum(rho, axis=0)                                                 # Rho'  [|Z|, K, |U|]
                psi = np.divide(rho_n + _common.prior_psi, np.sum(rho_n, axis=2, keepdims=True) + psi_den_p)

                # -------------- Message Preparation -------------- #
                msg = self._compute_messages(_common.m_omega, pi, psi)

                # ------------ Likelihood Computation ----------- #
                loglikelihood.append(pi_dir.logpdf(pi) + np.sum(psi_dir.logpdf(psi)) + np.sum(np.log(msg.m_psi_pi_sum_z)))

                # -------------- Iteration Control -------------- #
                iterations += 1

                # -------------- Debug Updates -------------- #
                self.update_progress(iterations * 100.0 / _common.max_iter)

            # Clean up
            self.update_progress(100.0)
            converged = self._converged(loglikelihood, _common.tolerance)

            # Return Result
            return self.ComputeResult_t(Pi=pi, Psi=psi, LogLikelihood=loglikelihood[-1], Converged=converged)

        @staticmethod
        def _compute_messages(m_omega: np.ndarray, pi, psi):
            """
            Compute the Messages for this iteration

            :param m_omega: Omega Message
            :param pi:  Latent Probabilities
            :param psi:  Emission Probabilities
            :return:
            """
            m_omega_star = np.multiply(m_omega[:, np.newaxis, :, :], psi[np.newaxis, :, :, :])  # [N, |Z|, K, |U|]
            m_omega_star_sum_u = np.sum(m_omega_star, axis=3, keepdims=True)                    # [N, |Z|, K,  1 ]
            m_psi_pi = np.multiply(np.prod(m_omega_star_sum_u, axis=2, keepdims=True),
                                   pi[np.newaxis, :, np.newaxis, np.newaxis])                   # [N, |Z|, 1,  1 ]
            m_psi_pi_sum_z = np.sum(m_psi_pi, axis=1, keepdims=True)                            # [N,  1,  1,  1 ]
            return AnnotISAR.EMWorker.Messages_t(m_omega_star, m_omega_star_sum_u, m_psi_pi, m_psi_pi_sum_z)

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
        self.__sink = NullableSink(sink)

    def _write(self, *args):
        self.__sink.write(*args)
        self.__sink.flush()

    def fit_model(self, data_hot, schema, prior, _starts=1):
        """
        Fit the Parameters Pi/Psi to the data, and generate MAP estimates for the latent behaviour:

        :param data_hot:        One-Hot Encoded Observations: [N by K * |X|]
        :param schema:          Schema for each sample (one-hot encoded): [N]
        :param prior:           Prior Probabilities for Pi and Psi (|Z|, [|Z| by K by |U|]): Note these must be raw
                                (smoothing) counts (equivalent to [alpha-1] in Dirichlet prior formulation)
        :param _starts:         This can be either:
                                    * Integer - Number of random starts to perform
                                    * List of Starting points (each starting point should be a tuple/list, containing
                                        the starting phi/psi matrices.
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
        self._write('Fitting ISAR Model on Data with {0} starts, each running for (max) {1} iterations.\n'
                    .format(workers, self.__max_iter))
        self.start_timer('global')

        # First Generate the M_Xi message
        self._write('Generating Latent-State Message (M_Omega)')
        self.start_timer('m_omega')
        _M_Omega = self.msg_omega(self.Omega, data_hot, schema)
        self.stop_timer('m_omega')
        self._write('... Done\n')

        # Now Run EM on the Data
        self._write('Running EM:\n')
        self.start_timer('em')
        results = self.RunWorkers(workers, self.EMWorker,
                                  _configs= self.EMWorker.ComputeCommon_t(max_iter=self.__max_iter,
                                                                          tolerance=self.__toler, m_omega=_M_Omega,
                                                                          prior_pi=prior[0], prior_psi=prior[1]),
                                  _args=_starts)
        self.stop_timer('em')

        # Stop Main timer
        self.stop_timer('global')

        # Display some statistics
        self._write('ISAR Model was fit in {0:1.5f}s of which:\n'.format(self.elapsed('global')))
        self._write('\t\tGenerating Emission Messages : {0:1.3f}s\n'.format(self.elapsed('m_omega')))
        self._write('\t\tExpectation Maximisation     : {0:1.3f}s ({1:1.5f}s/run)\n'.format(self.elapsed('em'),
                                                                                            self.elapsed('em')/workers))

        # Build (and return) Information Structure
        return self.AISARResults_t(ModelDims=[len(prior[0]), _M_Omega.shape[1], schema.shape[1], self.Omega.shape[2]],
                                   DataDims=len(data_hot), Pi=results.Pi, Psi=results.Psi, BestRun=results.Best,
                                   Converged=results.Converged, LogLikelihood=results.LogLikelihood,
                                   Times={'Total': self.elapsed('global'), 'Message': self.elapsed('m_omega'),
                                          'EM': self.elapsed('em')})

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
        _converged  = []  # Whether the run converged:

        # Iterate over results from each worker:
        for result in results:
            _final_pis.append(result.Pi)
            _final_psis.append(result.Psi)
            _final_llik.append(result.LogLikelihood)
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
                                    LogLikelihood=_final_llik[_best_index], Converged=np.asarray(_converged))

    @staticmethod
    def msg_omega(omega, _hot, _schema):
        """
        Generate M_omega Message

        This can be done once at the start, and then used throughout
        :param omega:   Schema-Specific Emission Probabilities [|S| by |U| by |X|]
        :param _hot:    Schema-Specific One-Hot Encoded Data [N by |K|*|X|]
        :param _schema: The schema (1-Hot Encoded) [N by |S|]
        :return:        N by K by |U| matrix
        """
        # Reshape:
        _Ns = _hot.shape[0]
        _Xs = omega.shape[-1]
        _Ks = np.int(_hot.shape[1]/_Xs)
        _hot = np.reshape(_hot, (_Ns, _Ks, _Xs))
        _X_pow = np.power(omega[np.newaxis, np.newaxis, :, :, :], _hot[:, :, np.newaxis, np.newaxis, :]).prod(axis=4)
        return np.power(_X_pow, _schema[:, np.newaxis, :, np.newaxis]).prod(axis=2)

    # @staticmethod
    # def msg_xi_update(_xi, _X, _S):
    #     """
    #     Generate M_Xi Message
    #
    #     This can be done once at the start, and then used throughout.
    #
    #     :param _xi:     Schema-Specific Emission Probabilities [|S| by |U| by |X|]
    #     :param _X:      Annotator Labels: np.NaN if unlabelled [N by |K|]
    #     :param _S:      The Schema used by the Annotator [N by |K|]
    #     :return:        N by K by |U| matrix
    #     """
    #     # Get the Dimensionality of the Problem
    #     _sN = _X.shape[0]
    #     _sK = _X.shape[1]
    #     _sU = _xi.shape[1]
    #
    #     # Now iterate over samples
    #     _M_Xi = np.empty([_sN, _sK, _sU])
    #     for n in range(_sN):
    #         for k in range(_sK):
    #             _M_Xi[n, k, :] = np.ones(_sU) if np.isnan(_X[n, k]) else _xi[int(_S[n, k]), :, int(_X[n, k])]
    #
    #     # Return
    #     return _M_Xi

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
        msgs = AnnotISAR.EMWorker._compute_messages(m_omega, pi, psi)
        gamma = np.divide(msgs.m_psi_pi, msgs.m_psi_pi_sum_z).squeeze()

        # Branch on whether to compute MAP or just output probabilities
        if label_set is not None:
            map_raw = np.argmax(gamma, axis=1)
            return npext.value_map(map_raw, label_set, shuffle=True)  # Transform into original indexing (via label_set)
        else:
            return npext.sum_to_one(gamma, axis=1)                    # Return Normalised Probabilities

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
        _m_omega = AnnotISAR.msg_omega(theta[2], hot, sch)
        _msgs = AnnotISAR.EMWorker._compute_messages(_m_omega, theta[0], theta[1])
        if prior is not None:
            return npext.Dirichlet(prior[0] + 1.0).logpdf(theta[0]) + \
                   np.sum(npext.Dirichlet(prior[1] + 1.0).logpdf(theta[1])) + np.sum(np.log(_msgs.m_psi_pi_sum_z))
        else:
            return np.sum(np.log(_msgs.m_psi_pi_sum_z))
