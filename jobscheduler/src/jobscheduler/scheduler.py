import logging
import threading
import time
from typing import Callable, Dict, List, Tuple, Union

from jobscheduler.client import Client
from jobscheduler.job import Job
from jobscheduler.jobthread import JobThread
from jobscheduler.progresstracker import ProgressTracker
from jobscheduler.worker import worker_group_summary
from jobscheduler.workermanager import WorkerHeartbeatThread, WorkerManager


class Scheduler(threading.Thread):
    def __init__(
        self,
        name=None,
        maxsize=None,
        max_jobs_per_worker: Union[None, int, Dict[str, int]] = None,
        global_tracker: Union[ProgressTracker, None] = None,
        run_path=None,
    ):
        super().__init__()

        self.name = name or "Scheduler"
        self.global_tracker = global_tracker

        self.worker_manager = WorkerManager(max_jobs_per_worker=max_jobs_per_worker)
        self.worker_heartbeat_thread = WorkerHeartbeatThread(self.worker_manager)

        self.maxsize = maxsize

        ## LOCK
        self.lock = threading.Lock()

        self.queue: List[Tuple[Job, Callable]] = []
        self.redo_queue: List[Tuple[Job, Callable]] = []
        self.job_threads: List[JobThread] = []
        self.job_thread_id = 0
        ##

        self.stop_event = threading.Event()
        self.run_path = run_path

        self.logger = logging.getLogger(self.name)

        self.misc_message = ""

    def is_empty(self):
        with self.lock:
            return (
                len(self.queue) == 0
                and len(self.redo_queue) == 0
                and len(self.job_threads) == 0
            )

    def try_add_job(self, job: Job, poll=True, poll_freq=1, callback=None):
        first = True

        while True:
            if not first:
                time.sleep(poll_freq)
            else:
                first = False

            with self.lock:
                if self.maxsize is not None:
                    if len(self.queue) >= self.maxsize:
                        if poll:
                            continue
                        else:
                            return False

                self.queue.append((job, callback))
                self.logger.debug("Job %s added to the main queue", job.get_name())
                return True

    def wait_for_queue(self, *, size, poll_freq=1):
        while True:
            with self.lock:
                if len(self.queue) < size:
                    return
            time.sleep(poll_freq)

    def add_client(self, client: Client):
        self.worker_manager.add_client(client)

    def try_pop_clients_to_delete(self):
        return self.worker_manager.try_pop_clients_to_delete()

    def update_workers(self):
        self.worker_manager.update_workers()
        self.worker_manager.mark_workers_for_deletion()

    def get_workers_lock(self):
        return self.worker_manager.get_workers_lock()

    def get_workers(self):
        return self.worker_manager.get_workers()

    def handle_finished_jobs(self):
        with self.lock:
            unfinished_job_threads = []

            for job_thread in self.job_threads:
                if not job_thread.has_finished():
                    unfinished_job_threads.append(job_thread)
                    continue

                for worker in job_thread.worker_group.values():
                    self.worker_manager.deassign_worker(worker)

                if job_thread.has_error():
                    self.redo_queue.append((job_thread.job, job_thread.callback))
                    self.logger.warning(
                        "JobThread-%s with Job %s failed; rescheduling",
                        job_thread.get_name(),
                        job_thread.job.get_name(),
                    )
                else:
                    self.logger.info(
                        "JobThread-%s with Job %s finished successfully",
                        job_thread.get_name(),
                        job_thread.job.get_name(),
                    )

            self.job_threads = unfinished_job_threads

    def assign_job_from_queue(
        self,
        queue: List[Tuple[Job, Callable]],
        *,
        weights: Dict[str, int] = None,
    ):
        if weights is None:
            weights = {
                "cuda-12.4:gpu": 100,
                "accelerate": 1,
                "accelerate:mps": 2,
            }

        def sort_by_client_identifier(x: Tuple[int, Tuple[Job, Callable]]):
            return sum(
                [
                    weights.get(c.client_identifier, 0)
                    for c in x[1][0].get_required_clients()
                ]
            )

        for index, (job, callback) in sorted(
            enumerate(queue), key=sort_by_client_identifier, reverse=True
        ):
            worker_group = self.worker_manager.try_assign_job(job)
            if worker_group is not None:
                queue.pop(index)
                return job, callback, worker_group

        return None, None, None

    def try_schedule_job(self) -> JobThread:
        with self.lock:
            job, callback, worker_group = self.assign_job_from_queue(self.redo_queue)

            if worker_group is None:
                job, callback, worker_group = self.assign_job_from_queue(self.queue)

            if worker_group is None:
                return

            summary = worker_group_summary(worker_group)
            self.logger.info(
                "Schedule Job %s with worker group %s", job.get_name(), summary
            )

            job_thread = JobThread(
                self.job_thread_id,
                job,
                worker_group,
                self.worker_manager.get_workers_lock(),
                callback,
                job_type=self.name,
                global_tracker=self.global_tracker,
            )
            job_thread.start()

            self.job_thread_id += 1

            return job_thread

    def run(self):
        self.logger.info("Scheduler thread %s started", self.name)

        iteration = 0

        with self.worker_heartbeat_thread:
            while not self.stop_event.is_set():
                iteration += 1

                if self.global_tracker:
                    tracker_iteration, tracker_total_steps = (
                        self.global_tracker.get_progress()
                    )
                else:
                    tracker_iteration, tracker_total_steps = 0, 0

                if tracker_total_steps is None:
                    self.logger.info(
                        "Scheduler loop iteration %d - Queue: %d - Progress: %d - JobThreads: %d - Workers: %d%s",
                        iteration,
                        len(self.queue),
                        tracker_iteration,
                        len(self.job_threads),
                        len(self.get_workers()),
                        self.misc_message,
                    )
                else:
                    self.logger.info(
                        "Scheduler loop iteration %d - Queue: %d - Progress: %d / %d - JobThreads: %d - Workers: %d%s",
                        iteration,
                        len(self.queue),
                        tracker_iteration,
                        tracker_total_steps,
                        len(self.job_threads),
                        len(self.get_workers()),
                        self.misc_message,
                    )

                self.update_workers()
                self.handle_finished_jobs()

                scheduled_something = False
                while (job_thread := self.try_schedule_job()) is not None:
                    scheduled_something = True

                    with self.lock:
                        self.logger.info(
                            "Starting Job %s on JobThread-%s",
                            job_thread.job.get_name(),
                            job_thread.get_name(),
                        )
                        self.job_threads.append(job_thread)

                if not scheduled_something:
                    time.sleep(1)

        self.logger.info("Scheduler thread %s finished", self.name)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        self.stop_event.set()
        self.logger.info("Stop event set for scheduler thread %s", self.name)
