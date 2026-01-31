import logging
import math
import os
import threading
from typing import Callable, Dict, List, Tuple

from flask import Flask, jsonify, render_template_string, request, send_from_directory
from jobscheduler.client import Client
from jobscheduler.job import Job
from jobscheduler.jobthread import JobThread
from jobscheduler.progresstracker import ProgressTracker
from jobscheduler.scheduler import Scheduler
from jobscheduler.worker import Worker


class MonitorServer:
    def __init__(
        self,
        scheduler: Scheduler,
        global_tracker: ProgressTracker,
        host_hostname: str,
        host_port: int,
        port: int,
        base_directory=os.path.join(os.getcwd(), "www"),
        run_path=None,
    ):
        self.scheduler = scheduler
        self.global_tracker = global_tracker

        self.host_hostname = host_hostname
        self.host_port = host_port
        self.port = port

        self.run_path = run_path

        self.base_directory = base_directory
        self.app = Flask(
            __name__,
            static_url_path="/static",
            static_folder=os.path.join(base_directory, "static"),
            template_folder=os.path.join(base_directory, "templates"),
        )
        self.server_thread = None
        logging.getLogger("werkzeug").disabled = True

        self.max_queue = 50

        self.setup_routes()

    def setup_routes(self):
        @self.app.route("/", methods=["GET"])
        @self.app.route("/index.html", methods=["GET"])
        def index():
            if self.run_path:
                try:
                    success_rate_path = os.path.join(self.run_path, "success-rate.txt")
                    with open(success_rate_path, encoding="utf-8") as f:
                        success_rate = f.readlines()
                except FileNotFoundError:
                    success_rate = []
            else:
                success_rate = []

            index_path = os.path.join(self.base_directory, "templates", "index.html")
            with open(index_path, encoding="utf-8") as f:
                return render_template_string(
                    f.read(),
                    hostname=self.host_hostname,
                    port=self.host_port,
                    success_rate=success_rate,
                )

        def serialize_worker(w: Worker):
            return {
                "id": w.get_name(),
                "state": w.get_state().name,
                "client": serialize_client(w.get_client()),
                "idle_time": int(w.get_idle_time()),
                "busy_time": int(w.get_busy_time()),
                "step_times": [int(t) for t in w.get_step_times()],
                "wait_times": [int(t) for t in w.get_wait_times()],
                "between_step_times": [int(t) for t in w.get_between_step_times()],
                "n_steps": w.get_number_of_steps(),
                "n_assigned_jobs": w.get_number_of_jobs(),
                "current_capacity": w.get_current_capacity(),
                "max_jobs": w.get_max_jobs(),
            }

        def serialize_worker_group(worker_group: Dict[str, Worker]):
            return [serialize_worker(w) for w in worker_group.values()]

        def serialize_job(job: Job):
            iteration, n_steps = job.get_progress()

            return {
                "id": job.get_name(),
                "iteration": iteration,
                "n_steps": n_steps,
                "required_clients": [
                    c.client_identifier for c in job.get_required_clients()
                ],
            }

        def serialize_job_thread(job_thread: JobThread):
            runtime = job_thread.get_runtime()

            if not math.isnan(runtime):
                runtime = int(runtime)
            else:
                runtime = 0

            return {
                "worker_group": serialize_worker_group(job_thread.worker_group),
                "job": serialize_job(job_thread.job),
                "runtime": runtime,
            }

        def serialize_client(client: Client):
            return {
                "id": client.get_name(),
                "client_identifier": client.get_client_identifier(),
            }

        def summarize_queue(queue: List[Tuple[Job, Callable]]):
            single_summary = {}
            tuple_summary = {}
            for job, _ in queue:
                client_identifiers = [
                    c.client_identifier for c in job.get_required_clients()
                ]
                for c in client_identifiers:
                    if c not in single_summary:
                        single_summary[c] = 0
                    single_summary[c] += 1

                combination = ",".join(
                    [c.client_identifier for c in job.get_required_clients()]
                )
                if combination not in tuple_summary:
                    tuple_summary[combination] = 0
                tuple_summary[combination] += 1

            return {"client_identifiers": single_summary, "combinations": tuple_summary}

        @self.app.route("/api/data.json", methods=["GET"])
        def api_data():
            # We do not get the scheduler lock here, because we only need an
            # approximation of the current state.
            n_queued_jobs = len(self.scheduler.queue)
            queued_jobs = [
                serialize_job(job)
                for job, future in self.scheduler.queue[: self.max_queue]
            ]
            queue_summary = summarize_queue(self.scheduler.queue)
            n_redo_queued_jobs = len(self.scheduler.redo_queue)
            redo_queued_jobs = [
                serialize_job(job)
                for job, future in self.scheduler.redo_queue[: self.max_queue]
            ]
            redo_queue_summary = summarize_queue(self.scheduler.redo_queue)

            job_threads = [
                serialize_job_thread(job_thread)
                for job_thread in self.scheduler.job_threads
            ]

            workers = {
                worker_id: serialize_worker(worker)
                for worker_id, worker in self.scheduler.get_workers().items()
            }

            iteration, total_steps = self.global_tracker.get_progress()

            response = {
                "elapsed_time": self.global_tracker.get_elapsed_time(),
                "progress": {
                    "iteration": iteration,
                    "total_steps": total_steps,
                    # "log": self.global_tracker.get_log(),
                },
                "job_threads": job_threads,
                "queued_jobs": queued_jobs,
                "n_queued_jobs": n_queued_jobs,
                "redo_queued_jobs": redo_queued_jobs,
                "n_redo_queued_jobs": n_redo_queued_jobs,
                "workers": workers,
                "max_queue": self.max_queue,
                "redo_queue_summary": redo_queue_summary,
                "queue_summary": queue_summary,
            }

            return jsonify(response)

        @self.app.route("/api/<int:worker_id>/max_jobs", methods=["POST"])
        def set_max_jobs(worker_id):
            # Parse the new max_jobs value from the request body
            data = request.get_json()

            # Ensure 'value' is provided and is an integer
            if not data or "value" not in data or not isinstance(data["value"], int):
                return (
                    jsonify({"error": "Invalid input. 'value' must be an integer."}),
                    400,
                )

            workers = self.scheduler.get_workers()
            if worker_id not in workers:
                return jsonify({"error": f"Worker with ID {worker_id} not found."}), 404

            try:
                workers[worker_id].set_max_jobs(data["value"])
            except Exception as e:  # pylint: disable=broad-except
                return jsonify({"error": f"Failed to set max_jobs: {str(e)}"}), 500

            return (
                jsonify(
                    {
                        "message": f"max_jobs updated successfully for worker {worker_id}."
                    }
                ),
                200,
            )

        @self.app.route("/api/max_queue", methods=["POST"])
        def max_queue():
            # Parse the new max_queue value from the request body
            data = request.get_json()

            # Ensure 'value' is provided and is an integer
            if not data or "value" not in data or not isinstance(data["value"], int):
                return (
                    jsonify({"error": "Invalid input. 'value' must be an integer."}),
                    400,
                )

            self.max_queue = data["value"]
            return (
                jsonify({"message": "max_queue updated successfully."}),
                200,
            )

        # Serve static files from the specified directory
        @self.app.route("/static/<path:filename>", methods=["GET"])
        def serve_static(filename):
            return send_from_directory(self.app.static_folder, filename)

    def run(self):
        """Run the Flask app in a separate thread."""
        self.app.run(host="0.0.0.0", port=self.port, threaded=True)

    def __enter__(self):
        # Start the server in a background thread
        self.server_thread = threading.Thread(target=self.run)
        self.server_thread.daemon = True  # Thread will exit when the main program does
        self.server_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flask does not provide a direct way to shut down the server gracefully
        # To shut down the Flask development server, we'll raise an exception
        # to stop it. Alternatively, we could implement a more sophisticated
        # mechanism, but that is out of scope for this example.
        if self.server_thread.is_alive():
            # Use the Flask built-in function for terminating the server
            # This may not be graceful; consider using other server setups in production
            # Here we just simulate shutdown by forcing an exception (not recommended
            # for production)
            self.server_thread.join(timeout=1)
