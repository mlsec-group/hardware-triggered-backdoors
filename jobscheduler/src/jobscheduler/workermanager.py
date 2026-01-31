import datetime
import logging
import threading
import time
from typing import Any, Dict, List, Union

from jobscheduler.client import Client
from jobscheduler.job import Job
from jobscheduler.worker import Worker, worker_group_summary


class WorkerManager:
    def __init__(self, *, max_jobs_per_worker: Union[None, int, Dict[str, int]] = None):
        self.name = "WorkerManager"
        self.max_jobs_per_worker = max_jobs_per_worker
        self.default_max_jobs_per_worker = 1

        ## LOCK
        self.lock = threading.RLock()

        self.workers: Dict[int, Worker] = {}
        self.client_number = 0
        self.clients_to_delete: List[Client] = []
        ##

        self.logger = logging.getLogger(self.name)

    def add_client(self, client: Client):
        with self.lock:
            worker_id = self.client_number

            max_jobs = None
            if isinstance(self.max_jobs_per_worker, int):
                max_jobs = self.max_jobs_per_worker
            elif isinstance(self.max_jobs_per_worker, dict):
                max_jobs = self.max_jobs_per_worker.get(
                    client.get_client_identifier(), None
                )

            if max_jobs is None:
                max_jobs = self.default_max_jobs_per_worker

            worker = Worker(
                worker_id,
                client,
                job_type=self.name,
                max_jobs=max_jobs,
            )
            self.workers[worker_id] = worker
            self.client_number += 1

            self.logger.info(
                "Create new worker '%s' from client ['%s']",
                worker_id,
                client.get_name(),
            )

    def mark_workers_for_deletion(self):
        with self.lock:
            worker_ids = list(self.workers.keys())

            for worker_id in worker_ids:
                if not self.workers[worker_id].is_dead():
                    continue

                worker = self.workers[worker_id]
                del self.workers[worker_id]

                self.clients_to_delete.append(worker.get_client())
                self.logger.info(
                    "Mark dead worker with ID %s for deletion (handles client %s)",
                    worker_id,
                    worker.get_client().get_name(),
                )

    def try_pop_clients_to_delete(self) -> Union[Client, None]:
        with self.lock:
            try:
                client = self.clients_to_delete.pop(0)
            except IndexError:
                return None

            self.logger.info("Popped client %s for deletion", client.get_name())
            return client

    def get_least_exhausted_worker(
        self,
        client_identifier: str,
        *client_args,
    ):
        with self.lock:
            available_workers = [
                worker
                for worker in self.workers.values()
                if not worker.is_dead()
                and worker.get_client().get_client_identifier() == client_identifier
            ]
            if len(available_workers) == 0:
                raise IndexError()

            highest_capacity_first = sorted(
                available_workers, key=lambda x: x.get_current_capacity(), reverse=True
            )
            for worker in highest_capacity_first:
                if worker.try_reserve_for_job(*client_args):
                    return worker

            raise IndexError()

    def try_assign_job(self, job: Job):
        with self.lock:
            worker_group: Dict[str, Worker] = {}

            try:
                for client_config in job.get_required_clients():
                    worker = self.get_least_exhausted_worker(
                        client_config.client_identifier, *job.get_client_args()
                    )
                    worker_group[client_config.client_identifier] = worker
            except IndexError:
                for worker in worker_group.values():
                    worker.release_from_job()
                return None
            except Exception:  # pylint: disable=broad-except
                for worker in worker_group.values():
                    worker.release_from_job()
                return None

            self.logger.debug(
                "Assign workers: %s", str(worker_group_summary(worker_group))
            )
            return worker_group

    def deassign_worker(self, worker: Worker):
        with self.lock:
            worker.release_from_job()
            self.logger.debug(
                "Deassining worker %s with client_identifier %s",
                worker.get_name(),
                worker.get_client().get_client_identifier(),
            )

    def update_workers(self):
        with self.lock:
            for worker in self.workers.values():
                worker.update_time()

    def get_workers_lock(self):
        return self.lock

    def get_workers(self):
        return self.workers


class WorkerHeartbeatThread(threading.Thread):
    def __init__(self, manager: WorkerManager):
        super(WorkerHeartbeatThread, self).__init__()

        self.manager = manager
        self.heartbeat_delay = datetime.timedelta(minutes=2)
        self.sleep_delay = 1

        self.stop_event = threading.Event()

        self.name = "WorkerHeartbeatThread"
        self.logger = logging.getLogger("WorkerHeartbeatThread")
        self.last_heartbeat = None

    def run(self):
        while not self.stop_event.is_set():
            now = datetime.datetime.now()
            if self.last_heartbeat is not None:
                if now - self.last_heartbeat < self.heartbeat_delay:
                    time.sleep(self.sleep_delay)
                    continue

            with self.manager.get_workers_lock():
                workers = list(self.manager.get_workers().values())

            cardiogram = {
                w.get_client().get_name(): w.worker_heartbeat() for w in workers
            }
            self.last_heartbeat = now
            self.logger.info("Heartbeat results: %s", str(cardiogram))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        self.stop_event.set()
        self.logger.info("Stop event set for scheduler thread %s", self.name)
