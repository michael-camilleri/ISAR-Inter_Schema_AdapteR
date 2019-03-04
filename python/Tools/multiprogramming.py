from pathos.helpers import mp
import numpy as np
import threading
import weakref
import queue
import time
import abc
import sys

from Tools.utilities import dictextend


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100, print_suffix=False, sink=sys.stdout):
    """
    Call in a loop to create terminal progress bar

    Code courtesy of https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

    :param iteration: Current Progress
    :param total:     Maximum number of iterations
    :param prefix:    Prefix Wording
    :param suffix:    Suffix Wording
    :param decimals:  positive number of decimals in percent complete (Int)
    :param bar_length:character length of bar (Int)
    :param print_suffix: If True, always prints the suffix: else prints only when iteration is total
    :param sink: Where to write to
    :return: None
    """
    # Handle against invalid sink
    if sink is None: return

    # Format
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sink.write('\r%s |%s| %s%s' % (prefix, bar, percents, '%')),

    if iteration == total or print_suffix:
        sink.write(' {0}'.format(suffix))

    if iteration == total:
        sink.write('\n')

    sink.flush()


class WorkerHandler(metaclass=abc.ABCMeta):
    """
    This Class Implements an interface for controlling multiple workers. It is itself based on the multiprocessing
    library from pathos.

    Each worker communicates with this through a queue. The queue accepts a message-tuple consisting of:
        a) Flag which is 0 for the Handler class and >0 otherwise
        b) Completion Index (count out of 100)

    The class also provides a simple timing mechanism based on naming a timer as required.
    """

    HANDLER_ID = 0

    # ======================================== Internal Interfaces ======================================== #
    class Worker(metaclass=abc.ABCMeta):
        """
        This is the Worker Interface
        """

        @property
        def Statistics(self):
            return self.__stats

        def __init__(self, _id, handler):
            """
            Initialiser

            Note that the implementation Constructor should not take additional arguments: these may be passed to the
            parallel_compute method

            :param _id:      A Unique Identifier: This must be greater than 0!
            :param handler: An instance of a Worker Handler for getting the queue from
            :param params:  Additional parameters (in a tuple) to be used by the child implementation
            """
            self.ID = _id
            self.__queue = weakref.ref(handler.Queue)   # Keep Weak-Reference to Handler Class' Queue
            self.__stats = {}                           # Empty Dictionary of Statistics
            handler._register_worker(_id)

        @abc.abstractmethod
        def parallel_compute(self, _common, _data):
            """
            Parallel Computation to be performed

            Must be implemented by respective classes.
            :param _common: Any common data, across all workers
            :param _data:   The Data to utilise for each individual computation
            :return:        Any results (tupled)
            """
            raise NotImplementedError()

        def update_progress(self, progress, stats=None):
            """
            Must be called to indicate progress updates

            Note, that due to some bug in the MultiProcessing Library, which seems to sometimes reset the connection,
             I had to wrap this in a try-catch block

            :param progress: A number between 0 and 100 indicating the *CURRENT* progress as a percentage
            :param stats:    Dictionary of Key-Value pairs (statistics)
            :return:
            """
            try:
                self.__queue().put((self.ID, progress), block=False)
            except TypeError:
                sys.stderr.write('Warn: TypeError Encountered')
                pass
            if stats:
                dictextend(self.__stats, stats)


    # ========================================= Abstract Interface ========================================= #

    @abc.abstractmethod
    def aggregate_results(self, results):
        """
        Must be implemented to aggregate results from the Worker Pool

        :param results: List of Results (one from each worker)
        :return:       Aggregated Result Data
        """
        raise NotImplementedError()

    # =========================================== Implementation =========================================== #

    @property
    def Queue(self):
        return self._queue

    def __init__(self, num_proc, sink=sys.stdout):
        """
        Initialiser

        :param num_proc:    Number of processes to employ (default to the number of cores less 2) If 0 or less, then
                            defaults to Multi-Threading instead
        :param sink:        Sink where to write progress to (may be None)
        """
        # Parameters
        self.Sink = sink                        # Sink where to write progress to
        self.NumProc = num_proc                 # Number of Processes to employ

        # Management
        self._queue = mp.Manager().Queue() if num_proc > 0 else queue.Queue()   # Queue
        self.__timers = {}                                                      # Timers
        self.__thread = None                                                    # Progress Thread Handler
        self.__worker_set = None                                                # Dictionary of Workers and progress
        self.__done = 0                         # How many are finished
        self.__tasks_done = 0                   # How many (workers) are finished
        self.__complexity = 0                   # How many Tasks (work-points) we have

    def _reset(self, num_workers):
        self.__thread = threading.Thread(target=self.__handler_thread)
        self.__worker_set = np.zeros(num_workers)  # Set of Workers and associated progress
        self.__tasks_done = 0                      # How many (workers) are finished
        self.__complexity = 100*num_workers        # How many Tasks (work-points) we have

    def start_timer(self, name):
        """
        Starts a timer running on name: if exists, resets

        :param name: Name of the timer to assign
        :return:     None
        """
        self.__timers[name] = [time.time(), None]

    def stop_timer(self, name):
        """
        Stops a timer: If name does not exist, raises an exception

        :param name: Timer to stop
        :return:     Elapsed time
        """
        self.__timers[name][1] = time.time()
        return self.elapsed(name)

    def elapsed(self, name):
        """
        Returns the elapsed time (after having stopped)

        :param name: Name of timer: must exist
        :return: Elapsed time
        """
        return self.__timers[name][1] - self.__timers[name][0]

    def RunWorkers(self, _num_work, _type, _configs, _args):
        """
        Starts the Pool of Workers and executes.

        The method blocks until all workers have completed

        :param _num_work:   Number of workers to initialise
        :param _type:       The worker type to run
        :param _configs:    These are common across all workers: may be None
        :param _args:       These are arguments per-worker. Must be a list equal in length to _workers
        :return:            Result of the Aggregator
        """
        # Reset Everything
        self._reset(_num_work)
        _args = _args if _args is not None else [None for _ in range(_num_work)]

        # Create List of Worker Objects, and initialise thread
        _workers = [_type(_i+1, self) for _i in range(_num_work)]
        self.__thread.start()

        # Start Pool and aggregate results
        if self.NumProc > 0:
            with mp.Pool(processes=self.NumProc) as pool:
                processes = [pool.apply_async(self.__computer, args=(_workers[_i], (_configs, _args[_i]))) for _i in range(_num_work)]
                aggregated = self.aggregate_results([result.get() for result in processes])
        else:
            r_q = queue.Queue()
            threads = [threading.Thread(target=self.__threader, args=(_workers[_i], (_configs, _args[_i]), r_q)) for _i in range(_num_work)]
            for thr in threads: thr.start(); thr.join()
            results = []
            while not r_q.empty():
                results.append(r_q.get())
                r_q.task_done()
            aggregated = self.aggregate_results(results)

        # Inform and join thread
        self.Queue.put([0, -1])
        self.__thread.join()

        # Return the aggregated information
        return aggregated

    def __computer(self, worker, arguments):
        return worker.parallel_compute(arguments[0], arguments[1])

    def __threader(self, worker, arguments, _queue):
        _queue.put(worker.parallel_compute(arguments[0], arguments[1]))

    def _register_worker(self, wrk_id):
        """
        Used by the Worker Class to register its' ID
        :param wrk_id:
        :return:
        """
        assert wrk_id > 0, 'Worker ID must be greater than 0'
        self.__worker_set[wrk_id-1] = 0.0

    def __handler_thread(self):
        # Flag for Stopping the function
        _continue = True

        # Print First one
        print_progress(0, self.__complexity, sink=self.Sink, prefix=('Working [{0} Proc]'.format(self.NumProc) if self.NumProc > 0 else 'Working [Multi-Thread]'),
                       suffix='Completed {0}/{1} Tasks'.format(self.__done, len(self.__worker_set)), print_suffix=True)

        # This provides an infinite loop until signalled to stop
        while _continue:
            # Handle Message
            _msg = self.Queue.get()
            if _msg[0] > self.HANDLER_ID:
                # Update Counts (and done count if this is the first time)
                if _msg[1] == 100.0 and self.__worker_set[_msg[0]-1] < 100: self.__done += 1
                self.__worker_set[_msg[0]-1] = _msg[1]
                # Print Progress
                print_progress(self.__worker_set.sum(), self.__complexity, prefix='Working [{0} Proc]'.format(self.NumProc),
                               suffix='Completed {0}/{1} Tasks'.format(self.__done, len(self.__worker_set)),
                               print_suffix=True, sink=self.Sink)
            else:
                _continue = False
                if self.Sink is not None:
                    self.Sink.write('Done: Completed All Tasks\n' if self.__done == len(self.__worker_set) else
                                    '\nStopped after {0}/{1} Tasks\n'.format(self.__done, len(self.__worker_set)))

            # Indicate Task Done
            self.Queue.task_done()
