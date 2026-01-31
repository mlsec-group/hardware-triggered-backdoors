import copy
import io
import os
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

import torch
from common.util import hash_tensor
from jobscheduler.client import ClientConfig
from jobscheduler.job import Job
from jobscheduler.progresstracker import ProgressTracker
from jobscheduler.worker import Worker
from torchvision.models.vision_transformer import VisionTransformer

from strategies.server.backdoor.after_grads_transformations.bit_flips import (
    BitFlipStrategy,
)
from strategies.server.backdoor.after_grads_transformations.permutation import (
    PermutationStrategy,
)
from strategies.server.backdoor.after_grads_transformations.util import (
    all_different,
    one_vs_all,
)
from strategies.server.backdoor.logging import GradientLogger
from strategies.server.backdoor.util import (
    EMPTY_HASH,
    get_trainer_worker,
    predict_on_workers,
)


@dataclass
class IterationTimes:
    start: datetime
    end: datetime
    sub_measurements: List[Tuple[str, datetime, datetime]]


@dataclass
class JobOutput:
    name: str
    job_start: datetime
    job_end: datetime
    logger_base_dir: str
    iteration_times: List[IterationTimes]
    message: str


class TrainerHandle:
    def __init__(
        self,
        run_id: str,
        trainer_worker: Worker,
        x_fool: torch.Tensor,
        y_fool: torch.Tensor,
        heuristic: str,
        heuristic_parameters: Dict[str, Any],
        use_full_model: bool,
        use_deterministic: bool,
    ):
        self.run_id = run_id
        self.trainer_worker = trainer_worker

        self.x_fool = x_fool
        self.y_fool = y_fool
        self.heuristic = heuristic
        self.heuristic_parameters = heuristic_parameters
        self.use_full_model = use_full_model
        self.use_deterministic = use_deterministic

    def __enter__(self):
        self.trainer_worker.worker_step(
            EMPTY_HASH,
            {
                "run_id": self.run_id,
                "reset": True,
                "x_fool": self.x_fool,
                "y_fool": self.y_fool,
                "heuristic": self.heuristic,
                "heuristic_parameters": self.heuristic_parameters,
                "use_full_model": self.use_full_model,
                "use_deterministic": self.use_deterministic,
            },
        )
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.trainer_worker.worker_step(
            EMPTY_HASH,
            {
                "run_id": self.run_id,
                "delete": True,
            },
        )


class TesterHandle:
    def __init__(
        self,
        run_id: str,
        model_path: str,
        model_hash: bytes,
        test_workers: Dict[str, Worker],
    ):
        self.run_id = run_id
        self.model_path = model_path
        self.model_hash = model_hash
        self.test_workers = test_workers

    def __enter__(self):
        for worker in self.test_workers.values():
            worker.worker_step(
                EMPTY_HASH,
                {
                    "run_id": self.run_id,
                    "set_model": True,
                    "model_path": self.model_path,
                    "model_hash": self.model_hash,
                },
            )

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for worker in self.test_workers.values():
            worker.worker_step(
                EMPTY_HASH,
                {
                    "run_id": self.run_id,
                    "del_model": True,
                },
            )


class SuccessEvaluator:
    def __init__(
        self,
        x_fool: torch.Tensor,
        trainer_worker: Worker,
        test_workers: Dict[str, Worker],
        logger: GradientLogger,
        *,
        accuracy_boundary: float,
        initial_accuracy: float,
        do_one_vs_all: bool,
    ):
        self.x_fool = x_fool
        self.trainer_worker = trainer_worker
        self.test_workers = test_workers
        self.logger = logger

        self.accuracy_boundary = accuracy_boundary
        self.initial_accuracy = initial_accuracy
        self.do_one_vs_all = do_one_vs_all

    def is_success(
        self,
        predictions: Dict[str, torch.Tensor],
        output_trainer,
    ):
        assert len(predictions) == len(self.test_workers)

        # [n_workers, n_samples]
        prediction_matrix = torch.empty(
            len(self.test_workers), self.x_fool.shape[0], dtype=torch.int32
        )

        for i, worker_name in enumerate(self.test_workers):
            prediction_matrix[i] = torch.argmax(predictions[worker_name], dim=-1)

        if self.do_one_vs_all:
            if not one_vs_all(prediction_matrix, 0):
                return False
        else:
            if not all_different(prediction_matrix):
                return False

        accuracy = output_trainer["accuracy"]
        if accuracy < self.accuracy_boundary * self.initial_accuracy:
            return False

        return True

    def is_prediction_close_enough(
        self, predictions: Dict[str, torch.Tensor], after_grad_threshold=0.1
    ):
        # (n_samples, n_classes)
        for prediction in predictions.values():
            top2 = torch.topk(prediction, k=2)
            abs_val = torch.abs(top2.values[:, 0] - top2.values[:, 1])

            if (abs_val > after_grad_threshold).any():
                return False

        return True


class BackdoorJob(Job):
    def __init__(
        self,
        name: str,
        client_configs: List[ClientConfig],
        model_path: str,
        x_index: int,
        x_fool: torch.Tensor,
        y_fool: torch.Tensor,
        client_cli_args,
        *,
        n_iterations: int,
        heuristic: str,
        heuristic_parameters: Dict[str, Any],
        run_path: str,
        global_tracker: ProgressTracker,
        permute_after_gradient: bool,
        flip_bits_after_gradient: bool,
        n_bits_flipped: int,
        use_full_model: bool,
        use_deterministic: bool,
        skip_is_prediction_close_check: bool,
        do_one_vs_all: bool = False,
    ):
        self.name = name
        self.client_configs = client_configs

        self.model_path = model_path
        self.x_index = x_index
        self.x_fool = x_fool
        self.y_fool = y_fool

        self.client_cli_args = client_cli_args

        self.n_iterations = n_iterations
        self.heuristic = heuristic
        self.heuristic_parameters = heuristic_parameters

        self.run_path = run_path

        # The new model must be above 95% accuracy of the original model
        self.accuracy_boundary = 0.95

        # Threshold for how close samples need to be, before we do the
        # permutation / bit flips transformations
        self.after_grad_threshold = 0.05

        self.iteration = 0
        self.global_tracker = global_tracker

        self.permute_after_gradient = permute_after_gradient
        self.flip_bits_after_gradient = flip_bits_after_gradient
        self.n_bits_flipped = n_bits_flipped

        self.use_full_model = use_full_model
        self.use_deterministic = use_deterministic
        self.do_one_vs_all = do_one_vs_all

        self.skip_is_prediction_close_check = skip_is_prediction_close_check

    def get_name(self):
        return self.name

    def get_required_clients(self):
        return self.client_configs

    def get_client_args(self):
        return "backdoor", self.client_cli_args

    def get_progress(self):
        return self.iteration, self.n_iterations

    def init(self, worker_group: Dict[str, Worker]):
        for worker in worker_group.values():
            worker.worker_init(*self.get_client_args())

    def retrieve_initial_prediction(
        self,
        run_id: str,
        trainer_worker: Worker,
        use_full_model: bool,
        model_hash: bytes,
    ):
        _, initial_trainer_output = trainer_worker.worker_step(
            EMPTY_HASH,
            {
                "run_id": run_id,
                "test": True,
                "x_fool": self.x_fool,
                "state_dict_update_compressed": None,
                "use_full_model": use_full_model,
                "model_hash": model_hash,
                "is_trainer": True,
            },
        )
        return initial_trainer_output

    def retrieve_initial_accuracy(self, run_id: str, trainer_worker: Worker):
        return trainer_worker.worker_step(
            EMPTY_HASH,
            {
                "run_id": run_id,
                "accuracy": True,
            },
        )[1]["accuracy"]

    def build_log_dir(self):
        pair_subdir = "-".join(
            [
                (
                    c.client_identifier[:-4]
                    if c.client_identifier.endswith(":gpu")
                    else c.client_identifier
                )
                for c in self.client_configs
            ]
        )
        return os.path.join(self.run_path, pair_subdir)

    def perform_training_step(
        self, run_id: str, trainer_worker: Worker, initial_predictions
    ):
        _, output_trainer = trainer_worker.worker_step(
            EMPTY_HASH,
            {
                "run_id": run_id,
                "training": True,
                "predictions": initial_predictions,
            },
        )
        state_dict_update_compressed = output_trainer["state_dict_update_compressed"]
        return output_trainer, state_dict_update_compressed

    def run(self, worker_group: Dict[str, Worker]):
        assert all(
            self.client_configs[i].client_identifier in worker_name
            for i, worker_name in enumerate(worker_group)
        ), f"Workers are not in the same order as the client configs"

        job_start = datetime.now()

        run_id = str(uuid.uuid4())
        trainer_worker = get_trainer_worker(worker_group)
        test_workers = worker_group

        worker_names = list(test_workers.keys())

        # TrainerHandle initializes and de-initializes the trainer worker's client
        # side. This is important for memory management reasons, because the
        # client has some state that needs to be deleted at the end of the run.
        with TrainerHandle(
            run_id,
            trainer_worker,
            self.x_fool,
            self.y_fool,
            self.heuristic,
            self.heuristic_parameters,
            self.use_full_model,
            self.use_deterministic,
        ):
            model = torch.load(
                self.model_path,
                map_location=torch.device("cpu"),
                weights_only=True,
            )
            initial_model_hash = hash_tensor(*model.parameters())
            initial_model_hash_hex = initial_model_hash.hex()

            initial_trainer_output = self.retrieve_initial_prediction(
                run_id, trainer_worker, self.use_full_model, initial_model_hash
            )

            if (
                initial_model_hash_hex
                == "acfa13bc0d21cbc91057eb66eae5fff6454feed00323e5f134017a8208af9997"
            ):
                initial_accuracy = 0.75908
            elif (
                initial_model_hash_hex
                == "c177ef855cd691eacd8caec92fcec9ef1c116cad755e12b4fb20468236a72a9f"
            ):
                initial_accuracy = 0.81322
            elif (
                initial_model_hash_hex
                == "9d05cb7cc110598f7475560b8d7d4ccf8a8efa57ebd5ef20c0da09aa8850ea09"
            ):
                initial_accuracy = 0.6976
            else:
                initial_accuracy = self.retrieve_initial_accuracy(
                    run_id, trainer_worker
                )

            logger_base_dir = self.build_log_dir()

            with GradientLogger(
                logger_base_dir,
                self.x_index,
                self.x_fool,
                self.y_fool,
                worker_names,
                **self.heuristic_parameters,
            ) as logger, ThreadPoolExecutor() as executor:
                with TesterHandle(
                    run_id, self.model_path, initial_model_hash, test_workers
                ):
                    #
                    # Get initial predictions of workers
                    #
                    initial_worker_outputs, initial_predictions, _ = predict_on_workers(
                        run_id,
                        self.x_fool,
                        test_workers,
                        executor,
                        state_dict_update_compressed=None,
                        model_hash=initial_model_hash,
                        use_full_model=self.use_full_model,
                    )

                    logger.log_initial_state(
                        model,
                        initial_accuracy,
                        initial_worker_outputs,
                        initial_trainer_output,
                    )

                    success_evaluator = SuccessEvaluator(
                        self.x_fool,
                        trainer_worker,
                        test_workers,
                        logger,
                        accuracy_boundary=self.accuracy_boundary,
                        initial_accuracy=initial_accuracy,
                        do_one_vs_all=self.do_one_vs_all,
                    )

                    bit_flip_strategy = BitFlipStrategy(
                        run_id,
                        self.x_fool,
                        trainer_worker,
                        test_workers,
                        executor,
                        logger,
                        n_bits_flipped=self.n_bits_flipped,
                        initial_accuracy=initial_accuracy,
                        accuracy_boundary=self.accuracy_boundary,
                        do_one_vs_all=self.do_one_vs_all,
                        use_full_model=self.use_full_model,
                    )

                    permutation_strategy = PermutationStrategy(
                        run_id,
                        self.x_fool,
                        test_workers,
                        executor,
                        logger,
                        initial_accuracy=initial_accuracy,
                        accuracy_boundary=self.accuracy_boundary,
                        do_one_vs_all=self.do_one_vs_all,
                    )

                    iteration_times: List[IterationTimes] = []

                    for iteration in range(0, self.n_iterations):
                        sub_measurements: List[Tuple[str, datetime, datetime]] = []
                        iteration_start = datetime.now()

                        #
                        # Do one training step on the train worker to update the model
                        #

                        train_start = datetime.now()
                        output_trainer, state_dict_update_compressed = (
                            self.perform_training_step(
                                run_id, trainer_worker, initial_predictions
                            )
                        )

                        state_dict_update = torch.load(
                            io.BytesIO(zlib.decompress(state_dict_update_compressed)),
                            map_location=torch.device("cpu"),
                            weights_only=True,
                        )

                        if self.use_full_model:
                            model.load_state_dict(state_dict_update)
                        elif isinstance(model, VisionTransformer):
                            model.conv_proj.load_state_dict(state_dict_update)
                        else:
                            assert False

                        fingerprint = output_trainer["model_hash"]
                        assert fingerprint == hash_tensor(*model.parameters())

                        train_end = datetime.now()
                        sub_measurements.append(("Training", train_start, train_end))

                        #
                        # Evaluate the new model on the test_workers (parallelized)
                        #
                        predict_start = datetime.now()
                        _, predictions_after_grad, client_model_hash = (
                            predict_on_workers(
                                run_id,
                                self.x_fool,
                                test_workers,
                                executor,
                                self.use_full_model,
                                fingerprint,
                                state_dict_update_compressed,
                            )
                        )
                        assert client_model_hash == fingerprint

                        predict_end = datetime.now()
                        sub_measurements.append(
                            ("Prediction", predict_start, predict_end)
                        )

                        logger.log_iteration(
                            iteration,
                            output_trainer,
                            predictions_after_grad,
                            client_model_hash,
                        )

                        #
                        # Check for success after we used gradient descent
                        #

                        if success_evaluator.is_success(
                            predictions_after_grad, output_trainer
                        ):
                            self.handle_success(
                                iteration,
                                model,
                                output_trainer,
                                predictions_after_grad,
                                logger,
                            )
                            logger.write("SUCCESS AFTER GRAD")
                            logger.create_success_file("grad")
                            iteration_end = datetime.now()
                            iteration_times.append(
                                IterationTimes(
                                    iteration_start, iteration_end, sub_measurements
                                )
                            )
                            return JobOutput(
                                self.name,
                                job_start,
                                datetime.now(),
                                logger.log_dir,
                                iteration_times,
                                "Success (after grad)",
                            )

                        #
                        # Check whether we can do "smaller" transformations:
                        # - permutation
                        # - bit flips
                        #

                        if (
                            success_evaluator.is_prediction_close_enough(
                                predictions_after_grad,
                                after_grad_threshold=self.after_grad_threshold,
                            )
                            or self.skip_is_prediction_close_check
                        ):
                            logger.save_intermediate_model(
                                iteration,
                                "prediction-close-enough-models",
                                model=None,
                                info={
                                    "predictions_after_grad": {
                                        platform: torch.topk(pred, 2, dim=-1)
                                        for platform, pred in predictions_after_grad.items()
                                    },
                                    "accuracy": output_trainer["accuracy"],
                                },
                            )

                            if self.permute_after_gradient:
                                permute_start = datetime.now()
                                permute_success = permutation_strategy.run(
                                    iteration, copy.deepcopy(model), output_trainer
                                )
                                permute_end = datetime.now()
                                sub_measurements.append(
                                    ("Permutation", permute_start, permute_end)
                                )

                                if permute_success:
                                    self.global_tracker.update(
                                        self.n_iterations - iteration
                                    )
                                    self.iteration = self.n_iterations
                                    iteration_end = datetime.now()
                                    iteration_times.append(
                                        IterationTimes(
                                            iteration_start,
                                            iteration_end,
                                            sub_measurements,
                                        )
                                    )
                                    return JobOutput(
                                        self.name,
                                        job_start,
                                        datetime.now(),
                                        logger.log_dir,
                                        iteration_times,
                                        "Success (after permute)",
                                    )

                            if self.flip_bits_after_gradient:
                                bitflip_start = datetime.now()
                                bit_flip_success = bit_flip_strategy.run(
                                    iteration, copy.deepcopy(model)
                                )
                                bitflip_end = datetime.now()
                                sub_measurements.append(
                                    ("Bit-Flip", bitflip_start, bitflip_end)
                                )

                                if bit_flip_success:
                                    self.global_tracker.update(
                                        self.n_iterations - iteration
                                    )
                                    self.iteration = self.n_iterations
                                    iteration_end = datetime.now()
                                    iteration_times.append(
                                        IterationTimes(
                                            iteration_start,
                                            iteration_end,
                                            sub_measurements,
                                        )
                                    )
                                    return JobOutput(
                                        self.name,
                                        job_start,
                                        datetime.now(),
                                        logger.log_dir,
                                        iteration_times,
                                        "Success (after bit_flip)",
                                    )

                        self.iteration = iteration
                        self.global_tracker.update()
                        iteration_end = datetime.now()
                        iteration_times.append(
                            IterationTimes(
                                iteration_start, iteration_end, sub_measurements
                            )
                        )

                return JobOutput(
                    self.name,
                    job_start,
                    datetime.now(),
                    logger.log_dir,
                    iteration_times,
                    "Fail",
                )

    def handle_success(
        self,
        iteration: int,
        model: torch.nn.Module,
        output_trainer,
        predictions: Dict[str, torch.Tensor],
        logger: GradientLogger,
    ):
        logger.log_success(
            iteration,
            model,
            output_trainer,
            predictions,
            output_trainer["accuracy"],
        )

        self.global_tracker.update(self.n_iterations - iteration)
        self.iteration = self.n_iterations
