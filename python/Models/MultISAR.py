"""
This is the ISAR Model, applied to a Supervised Learning Scenario: in this case, where X|Z is a multinomial
"""
from sklearn.naive_bayes import MultinomialNB
from collections import namedtuple

import numpy as np
import time as tm
import warnings

from Tools import npext, WorkerHandler, NullableSink


class MultISAR(WorkerHandler):
    """
    This Class implements the Multinomial ISAR Model, in a multi-processing framework (allowing multiple independent
    restarts to happen in parallel)
    """
    # ============================== Data Type Definitions =============================== #
    WorkerResults_t = namedtuple('WorkerResults_t', ['Pi', 'Phi', 'LogLikelihood', 'Converged', 'Best', 'Evolutions'])
    MISARResults_t = namedtuple('MISARResults_t', ['ModelDims', 'DataDims', 'Pi', 'Phi', 'BestRun', 'Converged',
                                                   'LogLikelihood', 'Evolutions', 'Times'])

    # ========================== Private Nested Implementations ========================== #
    class EMWorker(WorkerHandler.Worker):
        """
        (Private) Nested class for running the EM Algorithm
        """
        # Definition of Named Data-Types
        ComputeCommon_t = namedtuple('ComputeParams_t', ['max_iter', 'tolerance', 'update_rate', 'm_omega', 'X',
                                                         'smoothing'])
        ComputeResult_t = namedtuple('ComputeResult_t', ['Pi', 'Phi', 'LogLikelihood', 'Converged', 'Evolutions'])

        def __init__(self, _id, _handler):
            """
            Initialiser

            :param _id:         Identifier - allows different seeds for random initialisation
            :param _handler:    The Worker Handler
            """
            # Initialise Super-Class
            super().__init__(_id, _handler)

        def parallel_compute(self, _common: ComputeCommon_t, _data):
            """
            Implementation of Parallel Computation
            :param _common: NamedTuple of type ComputeCommon
                            > max_iter:     Maximum number of iterations to compute for
                            > tolerance:    Tolerance Parameter for early stopping: if Likelihood does not change more
                                            than this amount between Iterations, then stop.
                            > update_rate:  For storing statistics - the rate at which to store intermediary results. If
                                            0, then do not store intermediate statistics
                            > m_omega:      The m_omega message, of size N by |Z|
                            > X:            The Feature-Values, Integer counts N by |X|
                            > smoothing:    Any smoothing offset over Z and X|Z.
            :param _data:   None or [pi, phi]
                            > pi - Initial value for pi
                            > phi - Initial value for phi
            :return: NamedTuple of type ComputeResult:
                            > Ph (optimised): size |Z|
                            > Phi (optimised): size |Z| by |X|
                            > Likelihood at end: Note that this is Observed Data Log-Likelihood
                            > Whether converged or not within max-iterations
                            > Evolutions over iterations, in a dictionary, containing Likelihood, Phi, Pi. The
                              likelihood is always present, but the others may be empty lists.
            """
            # Initialise parameters (and compute lengths)
            sN, sZ = _common.m_omega.shape
            sX = _common.X.shape[1]

            # Re-Seed with ID and Time
            np.random.seed(int(tm.time()) + self.ID)

            # Initialise Random Points (start)
            pi = _data[0].copy() if _data is not None else np.random.dirichlet(np.ones(sZ))
            phi = _data[1].copy() if _data is not None else np.random.dirichlet(np.ones(sX), size=[sZ])

            # Prepare for Loop
            iterations = 0        # Iteration Count
            loglikelihood = []    # Observed Data Log-Likelihood
            pi_evo = []
            phi_evo = []

            # Optionally Store the first (initialisation)
            if _common.update_rate > 0:
                pi_evo.append(pi)
                phi_evo.append(phi)

            # Start the Loop: we make heavy use of Message Passing (Appendix A)
            # Note that we pre-compute the first set of messages, since due to efficiency, we compute the msgs at the
            #   end of the loop, since they are needed for the Likelihood Computation.
            _unnorm_jll = self.compute_message(pi, phi, _common.smoothing, _common.X, _common.m_omega)
            log_prob_x = npext.masked_logsumexp(_unnorm_jll, axis=1)

            while iterations < _common.max_iter and not self._converged(loglikelihood, _common.tolerance):
                # -------------- E-Step: -------------- #
                # Normalise to get responsibility: I am following the example for MultinomialNB implementation
                _gamma = np.exp(_unnorm_jll - np.atleast_2d(log_prob_x).T)
                _gamma[~np.isfinite(_gamma)] = 0

                # -------------- M-Step: -------------- #
                # Compute Pi:
                pi = npext.sum_to_one(_gamma.sum(axis=0))
                # Learn Phi
                for z in range(sZ):
                    phi[z, :] = np.sum(np.multiply(_gamma[:, z][:, np.newaxis], _common.X), axis=0) + _common.smoothing
                phi = npext.sum_to_one(phi, axis=1)

                # Prepare for next iteration: Update the Multinomial Object with the new probabilities
                _unnorm_jll = self.compute_message(pi, phi, _common.smoothing, _common.X, _common.m_omega)
                log_prob_x = npext.masked_logsumexp(_unnorm_jll, axis=1)

                # ------------ Likelihood Computation ----------- #
                loglikelihood.append(np.nansum(log_prob_x))

                # -------------- Iteration Control -------------- #
                iterations += 1

                # -------------- Debug Updates -------------- #
                self.update_progress(iterations * 100.0 / _common.max_iter)
                if (_common.update_rate > 0) and (iterations % _common.update_rate == 0):
                    pi_evo.append(pi)
                    phi_evo.append(phi)

            # Optionally, store last entry if it is not the same as the last one
            if (_common.update_rate > 0) and (iterations % _common.update_rate != 0):
                pi_evo.append(pi)
                phi_evo.append(phi)

            # Clean up
            self.update_progress(100.0)
            converged = self._converged(loglikelihood, _common.tolerance)

            # Return Result
            return self.ComputeResult_t(Pi=pi, Phi=phi, LogLikelihood=loglikelihood[-1], Converged=converged,
                                        Evolutions={'Likelihood': np.asarray(loglikelihood), 'Pi': pi_evo,
                                                    'Phi': phi_evo})

        @staticmethod
        def compute_message(pi, phi, alpha, X, m_omega=None):
            """
            Note that this returns the log of the message function, with the m_omega used as mask. To this end, it
             assumes that m_omega is always binary: i.e. either 0 or 1!
            """
            _multinomial = MultinomialNB(alpha=alpha)
            _multinomial.classes_ = np.arange(len(pi))
            _multinomial.class_log_prior_ = np.log(pi)
            _multinomial.feature_log_prob_ = np.log(phi)
            if m_omega is not None:
                jll = _multinomial._joint_log_likelihood(X)
                jll[m_omega == 0] = np.NaN
                return jll
            else:
                return _multinomial

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
    def __init__(self, omega, _num_proc, _max_iter, _toler=1e-4, _track_rate=5, sink=None):
        """
        Initialiser

        :param omega:      Fixed Emission Matrix Omega. This must be an NdArray of size |S| by |Z| by |Y|
        :param _num_proc:   Number of processes to run in - see MultiProgramming
        :param _max_iter:   Maximum Number of iterations in EM
        :param _toler:      Tolerance for convergence
        :param _track_rate  Debug Storage rate
        :param sink:        (Optional) Sink for debug output
        """

        # Call Super-Class
        super(MultISAR, self).__init__(_num_proc, sink)

        # Initialise own-stuff
        self.Omega = omega
        self.Pi = None
        self.Phi = None
        self.__max_iter = _max_iter
        self.__toler = _toler
        self.__track_rate = _track_rate
        self.__sink = NullableSink(sink)

    def _write(self, *args):
        self.__sink.write(*args)
        self.__sink.flush()

    def fit_model(self, features, labels, schema, prior, _starts=1):
        """
        Fit the Parameters Pi/Phi to the data, and generate MAP estimates for the latent behaviour:

        :param features:        BOW Counts: [N by |X|]
        :param labels:          Target Labels, in schema (1-Hot Encoded): [N by |Y|]
        :param schema:          Schema for each sample (1-Hot Encoded): [N by |S|]
        :param prior:           Prior Probability smoothing for Pi and Phi (scalar)
        :param _latent_set:     The set of latent values (for mapping to if these are not just contiguous numbers)
        :param _starts:         This can be either:
                                    * Integer - Number of random starts to perform
                                    * List of Starting points (each starting point should be a tuple/list, containing
                                        the starting pi/phi matrices.
        :return:    NamedTuple of Type MISARResults
                        * ModelDims: Model Dimensions [|Z|, |S|, |X|, |Y|]
                        * DataDims:  Data Dimensions [N]
                        * Pi:   Latent Probabilities
                        * Phi:  Feature-Probabilities
                        * BestRun: Index of the best-run
                        * Converged: Which runs (boolean array) converged
                        * LogLikelihood: the Penalised Observed Data Log-Likelihood at end of the procedure (best one)
                        * Evolutions: List of Evolution_t namedtuples, indexed by run
                        * Times: Times to fit the model (dictionary)
        """
        # Handle Starts
        if hasattr(_starts, '__len__'):
            workers = len(_starts)
        else:
            workers = _starts
            _starts = None

        # Indicate Start of Fit
        self._write('Fitting Multinomial ISAR Model on Data with {0} starts, each running for (max) {1} iterations.\n'
                    .format(workers, self.__max_iter))
        self.start_timer('global')

        # First Generate the M_Omega message
        self._write('Generating Latent-State Message (M_Omega)')
        self.start_timer('m_omega')
        _M_Omega = self.msg_omega(self.Omega, labels, schema)
        self.stop_timer('m_omega')
        self._write('... Done\n')

        # Now Run EM on the Data
        self._write('Running EM:\n')
        self.start_timer('em')
        results = self.RunWorkers(workers, self.EMWorker,
                                  _configs= self.EMWorker.ComputeCommon_t(max_iter=self.__max_iter, X=features,
                                                                          tolerance=self.__toler, m_omega=_M_Omega,
                                                                          update_rate=self.__track_rate,
                                                                          smoothing=prior),
                                  _args=_starts)
        self.stop_timer('em')

        # Stop Main timer
        self.stop_timer('global')

        # Display some statistics
        self._write('MISAR Model was fit in {0:1.5f}s of which:\n'.format(self.elapsed('global')))
        self._write('\t\tGenerating Emission Messages : {0:1.3f}s\n'.format(self.elapsed('m_omega')))
        self._write('\t\tExpectation Maximisation     : {0:1.3f}s ({1:1.5f}s/run)\n'.format(self.elapsed('em'),
                                                                                            self.elapsed('em')/workers))
        # Keep track of parameters:
        self.Pi = results.Pi.copy()
        self.Phi = results.Phi.copy()

        # Build (and return) Information Structure
        return self.MISARResults_t(ModelDims=[self.Omega.shape[1], schema.shape[1], features.shape[1], labels.shape[1]],
                                   DataDims=len(schema), Pi=results.Pi, Phi=results.Phi, BestRun=results.Best,
                                   Converged=results.Converged, LogLikelihood=results.LogLikelihood,
                                   Evolutions=results.Evolutions, Times={'Total': self.elapsed('global'),
                                                                         'Message': self.elapsed('m_omega'),
                                                                         'EM': self.elapsed('em')})

    def aggregate_results(self, results):
        """
        Maximise parameters over runs:

        :param results: the ApplyResult object, which is a list of ComputeResult_t
        :return: NamedTuple of type WorkerResults:
                    > Pi:               Best Pi Vector
                    > Phi:              Best Phi Matrix
                    > LogLikelihood:    Best Log-Likelihood
                    > Converged:        Boolean Array of which runs converged
                    > Best:             Index of the Best Run (absolute)
                    > Evolutions:       Evolutions (list of Evolution_t namedtuples indexed by run)
        """

        # Initialise Placeholders
        _final_pis = []  # Final Pi for each EM run
        _final_phis = []  # Final Phi for each EM run
        _final_llik = []  # Final Likelihood, for each EM run
        _converged  = []  # Whether the run converged:
        _evolutions = []  # Evolutions (by Run)

        # Iterate over results from each worker:
        for result in results:
            _final_pis.append(result.Pi)
            _final_phis.append(result.Phi)
            _final_llik.append(result.LogLikelihood)
            _converged.append(result.Converged)
            _evolutions.append(result.Evolutions)

        # Convert to Numpy Arrays for Indexing
        _final_llik = np.asarray(_final_llik)
        _final_pis = np.asarray(_final_pis)
        _final_phis = np.asarray(_final_phis)

        # Check whether at least one converged, and warn if not...
        if np.any(_converged):
            _masked_likelihood = np.ma.masked_array(_final_llik, np.logical_not(_converged))
        else:
            warnings.warn('None of the Runs Converged: results may be incomplete')
            _masked_likelihood = _final_llik

        # Find the best one out of those converged, or out of all, if none converged
        _best_index = _masked_likelihood.argmax().squeeze()

        # Return Results in Dictionary:
        return self.WorkerResults_t(Pi=_final_pis[_best_index], Phi=_final_phis[_best_index], Best=_best_index,
                                    Evolutions=_evolutions, LogLikelihood=_final_llik[_best_index],
                                    Converged=np.asarray(_converged))

    @staticmethod
    def msg_omega(_omega, _labels, _schema):
        """
        Generate M_Omega Message

        This can be done once at the start, and then used throughout
        :param _omega:    Schema-Specific Emission Probabilities [|Z| by |S| by |Y|]
        :param _labels: Schema-Specific Annotator-provided targets [N by |Y|]
        :param _schema: The schema (1-Hot Encoded) [N by |S|]
        :return:        N by |Z| matrix
        """
        # Reshape:
        _Y_pow = np.power(_omega[np.newaxis, :, :, :], _labels[:, np.newaxis, np.newaxis, :]).prod(axis=-1)  # N |Z| |S|
        return np.power(_Y_pow, _schema[:, np.newaxis, :]).prod(axis=-1)                                     # N |Z|

    def predict(self, X):
        return self.EMWorker.compute_message(self.Pi, self.Phi, 0, X).predict(X)

    def predict_proba(self, X):
        return self.EMWorker.compute_message(self.Pi, self.Phi, 0, X).predict_proba(X)
