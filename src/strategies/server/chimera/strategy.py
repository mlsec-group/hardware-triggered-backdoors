from argparse import Namespace
import json
import os
import time
from typing import List, Union

import torch

from common.interface import DirConfig
from datasets.cifar10 import load_cifar10_batch
from jobscheduler.client import ClientConfig
from jobscheduler.progresstracker import ProgressTracker
from jobscheduler.scheduler import Scheduler
from strategies.chimera_config import ChimeraSearchConfig
from strategies.server.chimera.job import ChimeraJob
from strategies.server_cli.chimera import ChimeraCLI
from strategies.server_strategy import ServerStrategy


class ChimeraServer(ChimeraCLI, ServerStrategy):
    def __init__(
        self,
        run_path: str,
        dir_config: DirConfig,
        seed: int,
        client_configs: List[ClientConfig],
        args: Namespace,
    ):
        super().__init__(run_path, dir_config, seed, client_configs)
        self.args = args
        self.search_config = ChimeraSearchConfig()
        self.job_outputs = []
        self.log_path = os.path.join(self.run_path, "submission.log")
        self.summary_path = os.path.join(self.run_path, "summary.json")

        self.generator_backend = args.generator_backend
        self.blis_backend = args.blis_backend
        self.openblas_backend = args.openblas_backend
        self.required_backends = [
            self.generator_backend,
            self.blis_backend,
            self.openblas_backend,
        ]

        configured = {cfg.client_identifier for cfg in client_configs}
        missing = [backend for backend in self.required_backends if backend not in configured]
        if missing:
            raise ValueError(
                "Chimera requires --backends entries for all roles. Missing: "
                + ", ".join(missing)
            )

        images, labels = load_cifar10_batch(args.cifar_batch)
        start = int(args.sample_index)
        end = start + int(args.n_samples)
        self.samples = [
            (idx, torch.as_tensor(images[idx], dtype=torch.float32), int(labels[idx]))
            for idx in range(start, min(end, len(images)))
        ]
        self._write_log(
            "Starting chimera run: "
            f"samples={len(self.samples)} sample_index={start} "
            f"requested_n_samples={int(args.n_samples)} "
            f"generator={self.generator_backend} blis={self.blis_backend} "
            f"openblas={self.openblas_backend} model={args.model_path} "
            f"walk_rounds={int(self.search_config.walk_rounds)} "
            f"probe_batch_size={int(self.search_config.probe_batch_size)} "
            f"sweep_coords_per_round={self.search_config.sweep_coords_per_round} "
            f"gd_steps={int(self.search_config.gd_steps)} "
            f"generator_device={args.generator_device}"
        )

    def _write_log(self, message: str) -> None:
        os.makedirs(self.run_path, exist_ok=True)
        timestamp = time.strftime("%H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}]>  {message}\n")

    def _write_summary(self) -> None:
        processed = len(self.job_outputs)
        chimera = sum(1 for output in self.job_outputs if output.get("success"))
        attempted = sum(
            1
            for output in self.job_outputs
            if output.get("status", {}).get("candidate_saved", False)
        )
        errors = sum(1 for output in self.job_outputs if output.get("error"))
        summary = {
            "processed": processed,
            "attempted": attempted,
            "chimera": chimera,
            "errors": errors,
            "requested_samples": int(self.args.n_samples),
            "actual_samples": len(self.samples),
            "success_rate": None if processed == 0 else chimera / processed,
            "jobs": sorted(self.job_outputs, key=lambda item: item["sample_index"]),
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

    def _client_config(self, backend: str) -> ClientConfig:
        for config in self.client_configs:
            if config.client_identifier == backend:
                return config
        raise KeyError(backend)

    def get_number_of_steps_per_job(self) -> int:
        return int(self.search_config.walk_rounds)

    def get_number_of_jobs(self) -> int:
        return len(self.samples)

    def start_campaign(
        self,
        scheduler: Scheduler,
        *,
        global_tracker: Union[ProgressTracker, None] = None,
    ):
        client_cli_args = {
            "--generator_backend": self.generator_backend,
            "--blis_backend": self.blis_backend,
            "--openblas_backend": self.openblas_backend,
            "--model_path": self.args.model_path,
            "--generator_device": self.args.generator_device,
        }

        required_clients = [
            self._client_config(self.generator_backend),
            self._client_config(self.blis_backend),
            self._client_config(self.openblas_backend),
        ]

        for job_id, (sample_index, image, label) in enumerate(self.samples):
            scheduler.try_add_job(
                ChimeraJob(
                    f"ChimeraJob-{job_id}-sample-{sample_index}",
                    required_clients,
                    client_cli_args,
                    image=image,
                    label=label,
                    sample_index=sample_index,
                    seed=self.seed + job_id,
                    run_path=self.run_path,
                    log_path=self.log_path,
                    generator_backend=self.generator_backend,
                    blis_backend=self.blis_backend,
                    openblas_backend=self.openblas_backend,
                    max_rounds=int(self.search_config.walk_rounds),
                ),
                callback=self.job_finished,
            )

    def job_finished(self, output):
        self.job_outputs.append(output)
        status = output.get("status", {})
        if output.get("success"):
            self._write_log(
                f"[data_batch:{output['sample_index']}] Chimera found! "
                f"(total {sum(1 for item in self.job_outputs if item.get('success'))})"
            )
        else:
            self._write_log(
                f"[data_batch:{output['sample_index']}] Chimera NOT found! "
                f"best_abs_margin={status.get('best_abs_margin')} "
                f"round={status.get('round')}"
            )
        self._write_summary()
        print("Chimera job finished:", output)

    def final(self) -> None:
        self._write_summary()
        totals = {
            "processed": len(self.job_outputs),
            "attempted": sum(
                1
                for output in self.job_outputs
                if output.get("status", {}).get("candidate_saved", False)
            ),
            "chimera": sum(1 for output in self.job_outputs if output.get("success")),
            "errors": sum(1 for output in self.job_outputs if output.get("error")),
        }
        self._write_log(
            "\nOverall:"
            f"\n  processed: {totals['processed']}"
            f"\n  attempted: {totals['attempted']}"
            f"\n  chimeras:  {totals['chimera']}"
            f"\n  errors:    {totals['errors']}"
        )
        print("All chimera jobs finished!", totals)
