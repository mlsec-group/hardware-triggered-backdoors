import collections
import logging
import threading
import time
from enum import Enum
from typing import Any, Dict

from jobscheduler.client import Client


class WorkerState(Enum):
    IDLE = 0
    BUSY = 1
    DEAD = 2


class Worker:
    def __init__(self, name: str, client: Client, *, job_type="", max_jobs=1):
        self.name = name
        self.client = client
        self.job_type = job_type

        self.max_jobs = max_jobs

        now = time.time()
        self.timestamp_added = now

        ## LOCK
        self.state_lock = threading.RLock()
        self.worker_state = WorkerState.IDLE
        self.assigned_jobs = 0

        # How long we spend without being assigned to a job
        self.idle_time = 0
        self.timestamp_idle_start = time.time()

        # How long we spend being assigned to at least one job
        self.busy_time = 0
        self.timestamp_busy_start = None

        self.initialized = False
        ##

        ## LOCK
        self.step_lock = threading.RLock()

        # How long we spend waiting for the worker to be available
        self.wait_times = collections.deque(maxlen=50)
        # How long we spend doing a step
        self.step_times = collections.deque(maxlen=50)
        # How long we took between steps
        self.between_step_times = collections.deque(maxlen=50)
        self.last_step_end = None

        self.n_steps = 0
        ##

        self.logger = logging.getLogger(f"{self.job_type}:Worker-{self.name}")

    def get_number_of_jobs(self):
        return self.assigned_jobs

    def get_current_capacity(self):
        return self.max_jobs - self.assigned_jobs

    def get_max_jobs(self):
        return self.max_jobs

    def set_max_jobs(self, max_jobs: int):
        self.max_jobs = max_jobs

    def get_state(self):
        return self.worker_state

    def set_worker_state(self, state: WorkerState):
        with self.state_lock:
            if self.worker_state == WorkerState.DEAD:
                return

            self.worker_state = state

            if self.worker_state == WorkerState.IDLE:
                self.last_step_end = None

    def try_reserve_for_job(self, *client_args):
        with self.state_lock:
            if self.assigned_jobs >= self.max_jobs:
                return False

            # If this worker is already assigned, we can only assign another job
            # to it, if the command and arguments match
            if self.assigned_jobs > 0:
                if not self.client.client_arguments_match(*client_args):
                    return False

            # If the worker is currently idling, we need to initialize it, if it
            # has been used for another command and client args
            if self.assigned_jobs == 0:
                if not self.client.client_arguments_match(*client_args):
                    self.initialized = False

                self.timestamp_busy_start = time.time()
                self.set_worker_state(WorkerState.BUSY)

            self.assigned_jobs += 1
            return True

    def release_from_job(self):
        with self.state_lock:
            self.assigned_jobs -= 1

            if self.assigned_jobs == 0:
                self.timestamp_idle_start = time.time()
                self.set_worker_state(WorkerState.IDLE)

    def worker_init(self, *client_args):
        with self.step_lock, self.state_lock:
            if self.initialized:
                return

            try:
                self.client.client_init(*client_args)
            except Exception as e:  # pylint: disable=broad-except
                self.logger.warning("Worker died during cmd init %s", str(e))
                self.set_worker_state(WorkerState.DEAD)
                return False

            self.initialized = True

    def worker_heartbeat(self):
        with self.step_lock:
            if self.worker_state == WorkerState.DEAD:
                return False

            # We do not acquire the state lock here, because if the state
            # of the worker changes between the check and the heartbeat, we
            # have no way of detecting it anyway (since changing the state in
            # another thread would require the lock, which we would currently
            # hold here)

            try:
                self.client.client_heartbeat()
            except Exception as e:  # pylint: disable=broad-except
                self.logger.warning("Worker died during heartbeat %s", str(e))
                self.set_worker_state(WorkerState.DEAD)
                return False

        return True

    def worker_step(self, *args, **kwargs):
        wait_start = time.time()
        with self.step_lock:
            self.wait_times.append(time.time() - wait_start)

            if self.worker_state == WorkerState.DEAD:
                raise WorkerDiedException(self)

            # We do not acquire the state lock here, because if the state
            # of the worker changes between the check and the step, we
            # have no way of detecting it anyway (since changing the state in
            # another thread would require the lock, which we would currently
            # hold here)

            with self.state_lock:
                if self.last_step_end is not None:
                    self.between_step_times.append(time.time() - self.last_step_end)

            step_start = time.time()
            try:
                step_output = self.client.client_step(*args, **kwargs)
            except AssertionError as e:
                self.logger.warning("AssertionError during step %s", str(e))
                self.set_worker_state(WorkerState.DEAD)
                raise WorkerException(self) from e
            except Exception as e:
                self.logger.warning("Caught exception during step %s", str(e))
                self.set_worker_state(WorkerState.DEAD)
                raise WorkerException(self) from e

            with self.state_lock:
                self.last_step_end = time.time()
            self.step_times.append(time.time() - step_start)
            self.n_steps += 1
            return step_output

    def get_name(self):
        return self.name

    def get_client(self):
        return self.client

    def is_dead(self):
        return self.worker_state == WorkerState.DEAD

    def set_dead(self):
        self.set_worker_state(WorkerState.DEAD)

    def update_time(self):
        with self.state_lock:
            now = time.time()

            if self.worker_state == WorkerState.IDLE:
                self.idle_time += now - self.timestamp_idle_start
                self.timestamp_idle_start = now

            elif self.worker_state == WorkerState.BUSY:
                assert self.timestamp_busy_start is not None
                self.busy_time += now - self.timestamp_busy_start
                self.timestamp_busy_start = now

    def get_idle_time(self):
        return self.idle_time

    def get_busy_time(self):
        return self.busy_time

    def get_wait_times(self):
        return [t for t in self.wait_times]

    def get_step_times(self):
        return [t for t in self.step_times]

    def get_between_step_times(self):
        return [t for t in self.between_step_times]

    def get_number_of_steps(self):
        return self.n_steps


class WorkerException(Exception):
    def __init__(self, worker: Worker):
        self.worker = worker

    def get_worker(self):
        return self.worker


class WorkerDiedException(Exception):
    def __init__(self, worker: Worker):
        self.worker = worker

    def get_worker(self):
        return self.worker


def worker_group_summary(worker_group: Dict[str, Worker]):
    return [
        {
            "worker_id": w.get_name(),
            "client_name": w.get_client().get_name(),
            "client_identifier": w.get_client().get_client_identifier(),
        }
        for w in worker_group.values()
    ]
