"""
This is the Dawid-Skene Model, for comparing the ISAR model against.
"""
from mpctools.multiprocessing import IWorker, WorkerHandler
from sklearn.metrics import confusion_matrix
from mpctools.extensions import npext, utils
from scipy.stats import dirichlet
import numpy as np
import warnings


class AnnotDS(WorkerHandler):
    """
    This Class implements the Noisy Annotator Model, following the Formulation of Dawid-Skene

    Some important Notes:
       * For performance reasons, Psi is usually flattened across annotators: i.e. it is 2D of size |Z| \times K|U|
       * With regards to the raw data, these should be categorical data-types which encompass all and only those
         manifestations one is interested in: if the categorical variable has more categories than what is required,
         recategorise before passing the data - otherwise, it will compute for more labels than is necessary, and not
         only will it impact performance, but may also bias results...
    """

    # ========================== Private Nested Implementations ========================== #
    class EMWorker(IWorker):
        """
        (Private) Nested class for running the EM Algorithm

        Note that in this case, the index order is n,z,k,u (i.e. Sample, Latent, Annotator, Label)
        """

        def __init__(self, _id, _handler):
            """
            Initialiser

            :param _id:         Identifier - allows different seeds for random initialisation
            :param _handler:    The Worker Handler
            """
            # Initialise Super-Class
            super(AnnotDS.EMWorker, self).__init__(_id, _handler)

        def parallel_compute(self, _common, _data):
            """
            Implementation of Parallel Computation
            :param _common: (max_iter, tolerance, [dims], one_hot, [prior_counts], order)
                            > max_iter:     Maximum number of iterations to compute for
                            > tolerance:    Tolerance Parameter for early stopping: if Likelihood does not change more
                                            than this amount between Iterations, then stop.
                            > dims:         The dimensionality of the data, in a Tuple, [N, |Z|, K and |U|]
                            > one_hot:      The one-hot encoded Data, of size N by (K * |U|).
                            > prior_counts: Prior Probabilities for the Distributions, Tuple, in order, for Z and U. The
                                            dimensionality must match: i.e. the prior for Z must be of size |Z| and that
                                            for U of size |Z| by (K * |U|). Note that these should be raw counts equal
                                            to [alpha - 1] already.
                            > order:        True/False: if True, sorts the thetas in ascending order for identifiability
            :param _data:   None or [pi, psi]
                            > pi - Initial value for pi
                            > psi - Initial value for psi
            :return: Tuple containing (in order):
                            > Pi (optimised): size |Z|
                            > Psi (optimised): size |Z| by (K * |U|)
                            > Likelihood at each iteration: Note that this is the Expected Completed Penalised log-likelihood
                            > Whether converged or not within max-iterations
            """
            # Initialise parameters
            max_iterations, tolerance, [sN, sZ, sK, sU], U, [pPi, pPsi], order = _common

            # Initialise Random Points (start)
            pi, psi = _data

            # Initialise some precomputations
            pi_denom = np.sum(pPi) + sN

            # Prepare for Loop
            iterations = 0
            loglikelihood_O = []    # Observed Data Log-Likelihood
            loglikelihood_C = []    # Complete Data Log-Likelihood
            gamma = np.zeros((sN, sZ))

            # Start the Loop
            while iterations < max_iterations and not self.converged(loglikelihood_O, tolerance):
                # -------------- E-Step: -------------- #
                # Get the Unnormalised Responsibilities
                for z in range(sZ):
                    gamma[:, z] = pi[z]*np.prod(np.power(psi[z], U), axis=1)  # Just Numerator for now
                # Now Normalise
                gamma = npext.sum_to_one(gamma, axis=1)

                # -------------- M-Step: -------------- #
                # Now Compute Pi
                pi = (np.sum(gamma, axis=0) + pPi)/pi_denom
                # Finally Compute Psi
                psi = np.multiply(gamma[:, :, np.newaxis], U[:, np.newaxis, :]).sum(axis=0) + pPsi
                for k in range(sK): # Normalise
                    psi[:, k*sU:(k+1)*sU] = npext.sum_to_one(psi[:, k*sU:(k+1)*sU], axis=1)

                # -------------- Iteration Control -------------- #
                _likel = self.full_likelihood(pi, psi, pPi, pPsi, gamma, U)
                loglikelihood_O.append(_likel[0])
                loglikelihood_C.append(_likel[1])
                iterations += 1

                # -------------- Debug Updates -------------- #
                self.update_progress(iterations * 100.0 / max_iterations)

            # Clean up
            self.update_progress(100.0)
            converged = self.converged(loglikelihood_O, tolerance)

            # Optionally, sort the parameters in ascending probability for pi
            if order:
                _sort_order = np.argsort(pi)
                pi = pi[_sort_order]
                psi = np.stack((psi[j] for j in _sort_order), axis=0)

            # Return Result
            return pi, psi, np.asarray(loglikelihood_O), np.asarray(loglikelihood_C), converged

        @staticmethod
        def converged(likelihoods, tolerance):
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

        @staticmethod
        def full_likelihood(pi, psi, pPi, pPsi, gamma, data):
            """
            Convenience Function for Computing the Observed (Penalised) Data Likelihood, and the Complete Data Log-
            Likelihood. If Gamma is None, then the complete likelihood is not computed. Also, if pPi or pPsi is None,
            then their contribution to penalisation is not added. Note that this assumes that sU == sZ.
            """
            # Evaluate Dimensions
            sZ = len(pi)
            sU = sZ
            sK = int(data.shape[1]/sU)

            # Compute Observed Data Log-Likelihood
            _emission = np.prod(np.power(psi[np.newaxis, :, :], data[:, np.newaxis, :]), axis=2)  # N by Z (by KU)
            obs_likelihood = np.sum(np.log(np.matmul(_emission, pi)))

            if pPi is not None:
                obs_likelihood += dirichlet.logpdf(pi, pPi + 1.0)
            if pPsi is not None:
                for z in range(sZ):
                    for k in range(sK):
                        obs_likelihood += dirichlet.logpdf(psi[z, k*sU:(k + 1) * sU], pPsi[z, k*sU:(k + 1) * sU] + 1.0)

            # Compute Complete Data Log-Likelihood
            comp_likelihood = None
            if gamma is not None:
                comp_likelihood = np.sum(gamma * (np.log(pi)[np.newaxis, :] + np.matmul(data, np.log(psi.T))))
                if pPi is not None:
                    comp_likelihood += dirichlet.logpdf(pi, pPi + 1)
                if pPsi is not None:
                    for z in range(sZ):
                        for k in range(sK):
                            comp_likelihood += dirichlet.logpdf(psi[z, k * sU:(k + 1) * sU], pPsi[z, k * sU:(k + 1) * sU] + 1)

            # Return Both
            return [obs_likelihood, comp_likelihood]

    # ========================== Initialisers ========================== #
    def __init__(self, _dims, _num_proc, _max_iter, _toler=1e-4, sink=None):
        """
        Initialiser

        :param _dims:       The Dimensions of the Model, in order [N, |Z|, K, |U|]
        :param _num_proc:   Number of processes Control - see MultiProgramming
        :param _max_iter:   Maximum Number of iterations in EM
        :param _toler:      Tolerance for convergence
        :param _track_rate  Debug Storage rate
        :param sink:        (Optional) Sink for debug output
        """

        # Call Super-Class
        super(AnnotDS, self).__init__(_num_proc, sink)

        # Initialise own-stuff
        self.Dims = _dims
        self.__max_iter = _max_iter
        self.__toler = _toler

    def fit_model(self, data_hot, prior, _starts=1):
        """
        Fit the Parameters Pi/Psi to the data, and generate MAP estimates for the latent behaviour:

        :param data_hot:        One-Hot Encoded Observations:  Size N x K|U|
        :param prior:           Prior Probabilities for Pi and Psi [|Z|, |Z| x K|U|]
        :param _starts:         This can be either:
                                    * Integer - Number of random starts to perform
                                    * List of Starting points (each starting point should be a tuple/list, containing
                                        the starting pi/psi matrices.
        :return:    Dictionary, containing the results:
                        * Pi:  Latent Probabilities
                        * Psi:  Emission Probabilities
                        * LogLikelihood: the Penalised Observed Data Log-Likelihood at end of the procedure (best one)
                        * Best: Index of which run gave the best results
                        * Converged: Which runs (boolean array) converged
                        * LogLike_Evol: Evolution of the Log-Likelihood per iteration
                        * Times: Times to fit the model
        """
        # Handle Starts
        if hasattr(_starts, '__len__'):
            workers = len(_starts)
        else:
            workers = _starts
            _starts = None

        # Initialise some stuff...
        self._print('Fitting EM Model on Data with {0} starts, each running for (max) {1} iterations.'
                    .format(workers, self.__max_iter))
        self.start_timer('global')

        # Perform EM (and format output)
        self._write('Running EM:\n')
        self.start_timer('em')
        results = self.run_workers(workers, self.EMWorker,
                                   _configs=(self.__max_iter, self.__toler, self.Dims, data_hot, prior, False),
                                   _args=_starts)
        self.stop_timer('em')

        # Consolidate Data
        pi = results['Pi']
        psi = results['Psi']

        # Stop Main timer
        self.stop_timer('global')

        # Display some statistics
        self._write('DS Model was fit in {0:1.5f}s of which:\n'.format(self.elapsed('global')))
        self._print('\t\tExpectation Maximisation   : {0:1.3f}s ({1:1.5f}s/run)'.format(self.elapsed('em'),
                                                                                        self.elapsed('em')/workers))

        # Build (and return) Information Structure
        return {'Dims': self.Dims, 'Pi': pi, 'Psi': psi, 'Best': results['Best'], 'Converged': results['Converged'],
                'LogLikelihood': results['LogLikelihood'], 'LogLike_Evol': results['Stat_LLikel'],
                'Times': [self.elapsed('global'), self.elapsed('em')]
               }

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

        _evol_likel_O = []  # LogLikelihood Evolution (Observed)
        _evol_likel_C = []  # LogLikelihood Evolution (Complete)

        # Iterate over results from each worker:
        for result in results:
            _final_pis.append(result[0])
            _final_psis.append(result[1])
            _final_llik.append(result[2][-1])
            _converged.append(result[4])

            _evol_likel_O.append(result[2])
            _evol_likel_C.append(result[3])

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
        return {'Pi': _best_pi,
                'Psi': _best_psi,
                'LogLikelihood': _best_llikel,
                'Stat_LLikel': [_evol_likel_O, _evol_likel_C],
                'Converged': _converged,
                'Best': _best_index}

    @staticmethod
    def estimate_map(pi, psi, u, label_set, max_only=False, max_size=None):
        """
        Compute Predictions (most probable, based on MAP) for the latent states given the observations

        :param pi:          Latent distribution
        :param psi:         Class Conditional Densities
        :param u:           Observations (One-Hot Encoding, size [N by K*|U|])
        :param label_set:   The original label set for the latent states (i.e. actual values associated with each index
                            in the current ordering.
        :param max_only:    If True (default) just return the MAP estimate: otherwise, return the posterior probabilities
        :param max_size:    Must be set if max_only is False (otherwise ignored).
        :return:            Maximum a Posteriori latent state or the posterior probabilities
        """
        # Compute Posterior
        _posterior = np.multiply(np.prod(np.power(psi.T[np.newaxis, :, :], u[:, :, np.newaxis]), axis=1), pi[np.newaxis, :])

        # Branch on whether to compute MAP or just output probabilities
        if max_only:
            return npext.value_map(np.argmax(_posterior, axis=1), label_set, shuffle=True)   # Map to original Label-Set
        else:
            _posterior = npext.sum_to_one(_posterior, axis=1)        # Normalised Probabilities
            _mapped_post = np.zeros([_posterior.shape[0], max_size]) # Placeholder for mapped posterior (mapped to actual value in label_set)
            for _i, _l in enumerate(label_set):
                _mapped_post[:, _l] = _posterior[:, _i]
            return _mapped_post

    @staticmethod
    def optimise_permutations(_map, _raw, _pi, _psi, _labels):
        """
        Optimise the permutation of the latent states (_map) such that there is least error (confusion matrix, highest
        agreement) with the raw data.

        Note:
            * Method can handle disjoing label-_sets (through _labels) BUT they must be the same for _map and _raw
            * Method ignores NaN values in _raw (i.e. these do not contribute towards count)

        :param _map:    MAP Estimates of the data [N]
        :param _raw:    Raw (floating point Numpy Array) data [N by K]
        :param _pi:     Pi Vector (to permute)
        :param _psi:    Psi Vector (to permute)
        :param _labels: Label-Set (for both)
        :return: Tuple containing in order:
                        * Updated Pi
                        * Updated Psi
                        * Updated MAP Predictions
        """
        # Get # Annotators
        _K = _raw.shape[1]

        # Prepare Data
        _observed = _raw.astype(np.float).T.ravel() # Flatten out all annotators
        _valid = np.isfinite(_observed)             # Find out only those which have been labelled.
        _observed = _observed[_valid]               # Valid Observed
        _predicted = np.tile(_map, _K)[_valid]      # Valid Predicted

        # Compute Confusion Matrix
        _confusion = confusion_matrix(_observed, _predicted, labels=_labels)

        # Now Perform Hungarian Algorithm: we need to use the Transpose, since we want to find the row each map value
        #   should be assigned to.
        _best_permute, _ = npext.maximise_trace(_confusion.T)

        # Finally, update Parameters
        _pi[list(_best_permute)] = _pi.copy()
        _psi[list(_best_permute)] = _psi.copy()
        _best_labels = _labels[_best_permute]
        return _pi, _psi, npext.value_map(_map, _best_labels, _labels), _best_permute

    @staticmethod
    def data_likelihood(theta, prior, data):
        """
        Wrapper function for computing the Observed Data Log-Likelihood, assuming symmetric latent/visible spaces

        :param theta: (Pi, Psi)
        :param prior: (Pi_prior, Psi_Prior): If None, then do not include penalisation
        :param data:  The (1-Hot Encoded) Data
        :return:      Observed Data Log-Likelihood
        """
        if prior is not None:
            return AnnotDS.EMWorker.full_likelihood(theta[0], theta[1], prior[0], prior[1], None, data)[0]
        else:
            return AnnotDS.EMWorker.full_likelihood(theta[0], theta[1], None, None, None, data)[0]
