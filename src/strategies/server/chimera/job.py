import json
import os
import time
import uuid
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from jobscheduler.client import ClientConfig
from jobscheduler.job import Job
from jobscheduler.worker import Worker

EMPTY_HASH = bytes(32)


class ChimeraJob(Job):
    def __init__(
        self,
        name: str,
        client_configs: List[ClientConfig],
        client_cli_args,
        *,
        image: torch.Tensor,
        label: int,
        sample_index: int,
        dataset_index: int,
        seed: int,
        run_path: str,
        log_path: Optional[str],
        generator_backend: str,
        blis_backend: str,
        openblas_backend: str,
        max_rounds: int,
        save_preview_images: bool,
    ):
        self.name = name
        self.client_configs = client_configs
        self.client_cli_args = client_cli_args
        self.image = image
        self.label = int(label)
        self.sample_index = int(sample_index)
        self.dataset_index = int(dataset_index)
        self.seed = int(seed)
        self.run_path = run_path
        self.log_path = log_path
        self.generator_backend = generator_backend
        self.blis_backend = blis_backend
        self.openblas_backend = openblas_backend
        self.max_rounds = int(max_rounds)
        self.save_preview_images = bool(save_preview_images)
        self.iteration = 0
        self.probe_trace = []
        self.search_trace = []

    def get_name(self):
        return self.name

    def get_required_clients(self):
        return self.client_configs

    def get_client_args(self):
        return "chimera", self.client_cli_args

    def get_progress(self):
        return self.iteration, self.max_rounds

    def init(self, worker_group: Dict[str, Worker]):
        for worker in worker_group.values():
            worker.worker_init(*self.get_client_args())

    def _workers_by_identifier(self, worker_group: Dict[str, Worker]):
        by_identifier = {}
        for worker in worker_group.values():
            identifier = worker.get_client().get_client_identifier()
            by_identifier[identifier] = worker
        return by_identifier

    def _jsonable_status(self, status):
        out = {}
        for key, value in status.items():
            if key == "trace_events":
                continue
            if isinstance(value, torch.Tensor):
                continue
            out[key] = value
        return out

    def _collect_search_trace(self, status):
        events = status.get("trace_events")
        if events:
            self.search_trace.extend(events)

    def _append_log(self, message: str):
        if self.log_path is None:
            return
        timestamp = time.strftime("%H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}]>  {message}\n")

    def _sample_dir(self):
        return os.path.join(self.run_path, f"sample-{self.sample_index}")

    def _tensor_to_json(self, tensor: torch.Tensor):
        return tensor.detach().cpu().tolist()

    def _save_tensor_png(self, tensor: torch.Tensor, path: str) -> None:
        image = tensor.detach().cpu().to(torch.float32)
        if image.ndim == 4:
            image = image[0]
        image = image.clamp(0.0, 1.0)
        if image.ndim == 3:
            if image.shape[0] in (1, 3):
                image = image.permute(1, 2, 0)
            if image.shape[-1] == 1:
                image = image[..., 0]
        array = (image.numpy() * 255.0 + 0.5).astype(np.uint8)
        pil_image = Image.fromarray(array)
        max_dim = max(pil_image.size)
        if max_dim < 128:
            scale = max(1, 256 // max_dim)
            pil_image = pil_image.resize(
                (pil_image.width * scale, pil_image.height * scale),
                resample=Image.Resampling.NEAREST,
            )
        pil_image.save(path)

    def _save_result(self, status, blis_output=None, openblas_output=None):
        sample_dir = self._sample_dir()
        os.makedirs(sample_dir, exist_ok=True)

        torch.save(self.image, os.path.join(sample_dir, "original.pt"))

        candidate_saved = False
        preview_saved = False
        if "candidate" in status and isinstance(status["candidate"], torch.Tensor):
            torch.save(status["candidate"], os.path.join(sample_dir, "candidate.pt"))
            candidate_saved = True
            if self.save_preview_images and status.get("success"):
                self._save_tensor_png(self.image, os.path.join(sample_dir, "original.png"))
                self._save_tensor_png(
                    status["candidate"], os.path.join(sample_dir, "chimera.png")
                )
                preview_saved = True

        if blis_output is not None:
            torch.save(blis_output, os.path.join(sample_dir, "blis-output.pt"))
        if openblas_output is not None:
            torch.save(openblas_output, os.path.join(sample_dir, "openblas-output.pt"))

        jsonable_status = self._jsonable_status(status)
        jsonable_status["candidate_saved"] = candidate_saved
        jsonable_status["preview_saved"] = preview_saved
        with open(os.path.join(sample_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job": self.name,
                    "sample_index": self.sample_index,
                    "dataset_index": self.dataset_index,
                    "label": self.label,
                    "status": jsonable_status,
                },
                f,
                indent=4,
            )
        with open(
            os.path.join(sample_dir, "probe-log.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.probe_trace, f, indent=4)
        with open(
            os.path.join(sample_dir, "search-log.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.search_trace, f, indent=4)

    def run(self, worker_group: Dict[str, Worker]):
        start_time = time.time()
        run_id = str(uuid.uuid4())
        workers = self._workers_by_identifier(worker_group)
        generator = workers[self.generator_backend]
        blis = workers[self.blis_backend]
        openblas = workers[self.openblas_backend]

        _, start_status = generator.worker_step(
            EMPTY_HASH,
            {
                "run_id": run_id,
                "generator_start": True,
                "image": self.image,
                "label": self.label,
                "seed": self.seed,
            },
        )
        self._collect_search_trace(start_status)
        self._append_log(
            f"[sample:{self.sample_index} dataset_index:{self.dataset_index}] "
            f"start label={self.label} "
            f"original_class={start_status.get('original_class')} "
            f"initial_abs_margin={start_status.get('initial_abs_margin')}"
        )

        final_status = start_status
        final_blis_output = None
        final_openblas_output = None

        for round_idx in range(self.max_rounds):
            self.iteration = round_idx
            _, gen_output = generator.worker_step(
                EMPTY_HASH,
                {"run_id": run_id, "generator_next": True},
            )

            if gen_output.get("done"):
                final_status = gen_output
                self._collect_search_trace(gen_output)
                break

            self._collect_search_trace(gen_output)
            candidates = gen_output["candidates"]
            candidate_abs_margins = gen_output.get("candidate_abs_margins")
            self._append_log(
                f"[sample:{self.sample_index} dataset_index:{self.dataset_index} "
                f"ROUND:{round_idx}] "
                f"probing batch_size={int(candidates.shape[0])} "
                f"best_abs_margin={gen_output.get('best_abs_margin')} "
                f"candidate_abs_margins="
                f"{None if candidate_abs_margins is None else self._tensor_to_json(candidate_abs_margins)}"
            )

            _, blis_output = blis.worker_step(
                EMPTY_HASH,
                {
                    "run_id": run_id,
                    "oracle_infer": True,
                    "candidates": candidates,
                },
            )
            _, openblas_output = openblas.worker_step(
                EMPTY_HASH,
                {
                    "run_id": run_id,
                    "oracle_infer": True,
                    "candidates": candidates,
                },
            )

            blis_predictions = blis_output["predictions"]
            openblas_predictions = openblas_output["predictions"]
            disagreement = blis_predictions != openblas_predictions
            round_trace = {
                "round": round_idx,
                "candidate_shape": list(candidates.shape),
                "candidate_abs_margins": (
                    None
                    if candidate_abs_margins is None
                    else self._tensor_to_json(candidate_abs_margins)
                ),
                "best_abs_margin": gen_output.get("best_abs_margin"),
                "blis_predictions": self._tensor_to_json(blis_predictions),
                "openblas_predictions": self._tensor_to_json(openblas_predictions),
                "disagreement": self._tensor_to_json(disagreement),
            }
            self.probe_trace.append(round_trace)
            self._append_log(
                f"[sample:{self.sample_index} dataset_index:{self.dataset_index} "
                f"PROBE:R{round_idx}] "
                f"blis_pred={round_trace['blis_predictions']} "
                f"openblas_pred={round_trace['openblas_predictions']} "
                f"disagreement={round_trace['disagreement']}"
            )

            _, update_status = generator.worker_step(
                EMPTY_HASH,
                {
                    "run_id": run_id,
                    "generator_update": True,
                    "blis_predictions": blis_predictions,
                    "openblas_predictions": openblas_predictions,
                    "blis_logits": blis_output["logits"],
                    "openblas_logits": openblas_output["logits"],
                },
            )
            final_status = update_status
            final_blis_output = blis_output
            final_openblas_output = openblas_output
            self._collect_search_trace(update_status)
            self.probe_trace[-1]["update_status"] = self._jsonable_status(update_status)

            if update_status.get("done"):
                break

        elapsed_seconds = time.time() - start_time
        candidates_probed = sum(
            int(item.get("candidate_shape", [0])[0]) for item in self.probe_trace
        )
        candidates_to_chimera = None
        chimera_round = None
        chimera_batch_index = None
        chimera_candidate_abs_margin = None
        candidates_before_round = 0
        for item in self.probe_trace:
            disagreement = item.get("disagreement") or []
            if any(disagreement):
                update_status = item.get("update_status", {})
                chimera_batch_index = int(
                    update_status.get("chimera_index", disagreement.index(True))
                )
                chimera_round = int(item["round"])
                candidates_to_chimera = candidates_before_round + chimera_batch_index + 1
                candidate_abs_margins = item.get("candidate_abs_margins") or []
                if chimera_batch_index < len(candidate_abs_margins):
                    chimera_candidate_abs_margin = candidate_abs_margins[
                        chimera_batch_index
                    ]
                break
            candidates_before_round += int(item.get("candidate_shape", [0])[0])

        final_status["elapsed_seconds"] = elapsed_seconds
        final_status["probe_rounds"] = len(self.probe_trace)
        final_status["candidates_probed"] = candidates_probed
        final_status["candidates_to_chimera"] = candidates_to_chimera
        final_status["chimera_round"] = chimera_round
        final_status["chimera_batch_index"] = chimera_batch_index
        final_status["chimera_candidate_abs_margin"] = chimera_candidate_abs_margin
        final_status["chimera_candidate_margin"] = final_status.get("chimera_margin")
        final_status["chimera_candidate_competitor"] = final_status.get(
            "chimera_competitor"
        )
        final_status["chimera_candidate_l0_distance"] = final_status.get(
            "chimera_l0_distance"
        )
        final_status["chimera_candidate_l1_distance"] = final_status.get(
            "chimera_l1_distance"
        )
        final_status["chimera_candidate_linf_distance"] = final_status.get(
            "chimera_linf_distance"
        )
        self._save_result(final_status, final_blis_output, final_openblas_output)
        candidate_saved = os.path.exists(
            os.path.join(self._sample_dir(), "candidate.pt")
        )
        jsonable_status = self._jsonable_status(final_status)
        jsonable_status["candidate_saved"] = candidate_saved
        return {
            "job": self.name,
            "sample_index": self.sample_index,
            "dataset_index": self.dataset_index,
            "success": bool(final_status.get("success", False)),
            "status": jsonable_status,
        }
