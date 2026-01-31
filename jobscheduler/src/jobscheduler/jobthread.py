import logging
import threading
import time
from typing import Dict, Union

from jobscheduler.job import Job
from jobscheduler.progresstracker import ProgressTracker
from jobscheduler.worker import (
    Worker,
    WorkerDiedException,
    WorkerException,
    worker_group_summary,
)


class JobThread(threading.Thread):
    def __init__(
        self,
        name: str,
        job: Job,
        worker_group: Dict[str, Worker],
        workers_lock: threading.Lock,
        callback,
        *,
        job_type="",
        global_tracker: Union[ProgressTracker, None] = None,
    ):
        super().__init__()

        self.name = name
        self.job = job
        self.worker_group = worker_group
        self.workers_lock = workers_lock
        self.job_type = job_type
        self.global_tracker = global_tracker

        self.callback = callback

        self.start_time = None
        self.error = False
        self.finished = False

        # Setting up logger for this JobThread using the job ID for context
        self.logger = logging.getLogger(f"{self.job_type}:JobThread-{self.name}")

    def get_name(self):
        return self.name

    def has_finished(self) -> bool:
        return self.finished

    def has_error(self):
        return self.error

    def get_runtime(self):
        if self.start_time is None:
            return float("nan")
        else:
            return time.time() - self.start_time

    def cleanup_error(self):
        self.error = True
        iteration, _ = self.job.get_progress()
        if self.global_tracker:
            self.global_tracker.update(-iteration)

    def handle_worker_died_exception(self, e: WorkerDiedException):
        self.cleanup_error()
        self.logger.error(
            "Job %s failed because worker died earlier (Id: %s, Backend: %s",
            self.job.get_name(),
            e.get_worker().get_name(),
            e.get_worker().get_client().get_client_identifier(),
        )

    def handle_worker_exception(self, e: WorkerException):
        self.logger.error(
            "Job %s failed because of worker (Id: %s, Backend: %s:\n%s",
            self.job.get_name(),
            e.get_worker().get_name(),
            e.get_worker().get_client().get_client_identifier(),
            str(e),
            exc_info=True,
        )
        self.cleanup_error()
        raise e

    def handle_generic_exception(self, e):
        summary = worker_group_summary(self.worker_group)
        self.logger.error(
            f"Job %s failed with workers {summary}: {e}",
            self.job.get_name(),
            exc_info=True,
        )
        self.cleanup_error()
        self.mark_all_workers_as_dead()
        raise e

    def run(self):
        self.job.init(self.worker_group)

        self.start_time = time.time()
        self.logger.info("Starting Job %s", self.job.get_name())
        try:
            result = self.job.run(self.worker_group)
            self.logger.info("Job %s completed successfully", self.job.get_name())
            if self.callback is not None:
                self.callback(result)
        except AssertionError as e:
            summary = worker_group_summary(self.worker_group)
            self.logger.info(
                "AssertionError during job run %s in %s", str(e), summary, exc_info=True
            )
            self.mark_all_workers_as_dead()
            raise WorkerException(self) from e
        except WorkerDiedException as e:
            self.handle_worker_died_exception(e)
        except WorkerException as e:
            self.handle_worker_exception(e)
        except Exception as e:  # pylint: disable=broad-except
            self.handle_generic_exception(e)
        finally:
            self.finished = True
            # self.logger.info(f"Job {self.job.get_name()} marked as finished")

    def mark_all_workers_as_dead(self):
        with self.workers_lock:
            summary = worker_group_summary(self.worker_group)
            self.logger.debug("Mark workers as dead: %s", summary)

            for worker in self.worker_group.values():
                worker.set_dead()
