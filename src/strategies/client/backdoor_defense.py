import gc
import io
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator

import torch
import tqdm
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torch.autograd.grad_mode import inference_mode

from common.util import hash_tensor
from datasets.imagenet import ImageNetAccuracyEvaluator
from datasets.loader import get_dataset_loader
from strategies.client.backdoor_heuristics.util.compression import compress, decompress
from strategies.client_strategy import ClientStrategy

from torch.utils.data import DataLoader

N_STEPS_FOR_IMAGENET = 195



@dataclass
class TrainerRunState:
    model: torch.nn.Module
    train_loader: Iterator
    optimizer: torch.optim.Optimizer
    epoch_iteration: int
    batch_iteration: int


@dataclass
class TesterRunState:
    model: torch.nn.Module
    x_fool: torch.Tensor
    x_fool_normalized: torch.Tensor


class BackdoorDefenseClient(ClientStrategy):
    def __init__(self, backend: str, args: Any):
        super().__init__(backend)

        self.dataset = args.dataset
        self.config_dataset_dir = args.config_dataset_dir

        assert torch.cuda.is_available()

        self.model_device = torch.device("cuda")

        self.trainer_run_states: Dict[str, TrainerRunState] = {}
        self.tester_run_states: Dict[str, TesterRunState] = {}

        self.loader = get_dataset_loader(self.dataset, self.config_dataset_dir)

    @classmethod
    def get_cmd_name(cls) -> str:
        return "backdoor-defense"

    @classmethod
    def install_argparser(cls, subparsers) -> None:
        parser = super().install_argparser(subparsers)
        parser.add_argument("--config_dataset_dir", required=True, type=str)
        parser.add_argument(
            "--dataset", required=True, choices=["mnist", "fmnist", "cifar", "imagenet"]
        )

    @inference_mode()
    def get_fast_accuracy(self, model):
        model.eval()

        evaluator = ImageNetAccuracyEvaluator(
            os.path.join(self.config_dataset_dir, self.dataset, "val_set")
        )
        tqdm_bar = tqdm.tqdm(
            desc="Accuracy",
            total=N_STEPS_FOR_IMAGENET,
        )
        it = enumerate(evaluator.get_iterator())

        correct = 0
        total = 0

        while True:
            try:
                i, batch = next(it)
                if i == N_STEPS_FOR_IMAGENET:
                    break
            except StopIteration:
                break

            data = batch[0]["data"]
            outputs = model(data).cpu()
            preds = outputs.argmax(dim=1)

            labels = batch[0]["label"].squeeze().long().cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            approx_acc = correct / total

            # ChatGPT suggestion for data leaks. I am not sure if this is
            # needed...
            del data, outputs, preds

            tqdm_bar.update()
            tqdm_bar.set_description(f"Accuracy ({approx_acc*100:.2f}%)")

        print(torch.cuda.memory_summary(device=self.model_device, abbreviated=True))

        torch.cuda.empty_cache()
        gc.collect()

        print(torch.cuda.memory_summary(device=self.model_device, abbreviated=True))

        return correct / total

    def do_setup_trainer(self, run_id: str, *, model_path: str):
        assert run_id not in self.trainer_run_states

        model = torch.load(
            model_path, weights_only=True, map_location=self.model_device
        )

        train_dataset = self.loader.load_train()
        train_loader = iter(DataLoader(train_dataset, batch_size=256, shuffle=True))

        self.trainer_run_states[run_id] = TrainerRunState(
            model,
            train_loader,
            optimizer=torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9),
            epoch_iteration=0,
            batch_iteration=0,
        )

        return hash_tensor(), {"model_hash": hash_tensor(*model.parameters())}

    def do_setup_tester(self, run_id: str, *, model_path: str, x_fool_path: str):
        assert run_id not in self.tester_run_states

        model = torch.load(
            model_path, weights_only=True, map_location=self.model_device
        )
        x_fool = torch.load(
            x_fool_path, weights_only=True, map_location=self.model_device
        )
        x_fool_normalized = self.loader.normalize(x_fool)
        self.tester_run_states[run_id] = TesterRunState(
            model, x_fool, x_fool_normalized
        )

        return hash_tensor(), {"model_hash": hash_tensor(*model.parameters())}

    def do_gradient_step_trainer(self, run_id: str):
        run_state = self.trainer_run_states[run_id]

        train_loader = run_state.train_loader
        optimizer = run_state.optimizer

        model: torch.nn.Module = run_state.model
        model.train()

        criterion = torch.nn.CrossEntropyLoss()

        x, y = next(train_loader)
        x, y = x.to(self.model_device), y.to(self.model_device)
        x = self.loader.normalize(x)

        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()
        run_state.batch_iteration += 1

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        state_dict_bytes = buffer.read()
        state_dict_bytes_compressed = compress(state_dict_bytes)

        return hash_tensor(), {
            "state_dict_update_compressed": state_dict_bytes_compressed,
            "model_hash": hash_tensor(*model.parameters()),
        }

    @inference_mode()
    def do_update_tester(self, run_id: str, *, state_dict_update_compressed: bytes):
        model = self.tester_run_states[run_id].model

        state_dict_update = torch.load(
            io.BytesIO(decompress(state_dict_update_compressed)),
            map_location=self.model_device,
            weights_only=True,
        )

        model.load_state_dict(state_dict_update)

        return hash_tensor(), {"model_hash": hash_tensor(*model.parameters())}

    @inference_mode()
    def do_evaluate_tester(self, run_id: str):
        model = self.tester_run_states[run_id].model
        model.eval()

        x_fool_normalized = self.tester_run_states[run_id].x_fool_normalized
        y = model(x_fool_normalized)

        return hash_tensor(), {"model_hash": hash_tensor(*model.parameters()), "y": y}

    @inference_mode()
    def do_get_accuracy(self, run_id: str, *, is_trainer: bool):
        if is_trainer:
            model = self.trainer_run_states[run_id].model
        else:
            model = self.tester_run_states[run_id].model
        model.eval()

        return hash_tensor(), {
            "accuracy": self.get_fast_accuracy(model),
            "model_hash": hash_tensor(*model.parameters()),
        }

    def do_delete_trainer(self, run_id: str):
        del self.trainer_run_states[run_id]
        return hash_tensor(), {}

    def do_delete_tester(self, run_id: str):
        del self.tester_run_states[run_id]
        return hash_tensor(), {}

    def step(
        self,
        server_hash,
        run_id: str,
        *,
        setup_trainer=False,
        setup_tester=False,
        gradient_step=False,
        update=False,
        evaluate=False,
        get_accuracy=False,
        delete_trainer=False,
        delete_tester=False,
        **kwargs,
    ):

        assert (
            sum(
                [
                    setup_trainer,
                    setup_tester,
                    gradient_step,
                    update,
                    evaluate,
                    get_accuracy,
                    delete_trainer,
                    delete_tester,
                ]
            )
            == 1
        )

        if setup_trainer:
            print(f"[{run_id}] SETUP TRAINER", flush=True)
            out = self.do_setup_trainer(run_id, **kwargs)
            print(f"[{run_id}] SETUP TRAINER (out)", flush=True)
            return out

        if setup_tester:
            print(f"[{run_id}] SETUP TESTER", flush=True)
            out = self.do_setup_tester(run_id, **kwargs)
            print(f"[{run_id}] SETUP TESTER (out)", flush=True)
            return out

        if gradient_step:
            print(f"[{run_id}] GRADIENT STEP", flush=True)
            out = self.do_gradient_step_trainer(run_id, **kwargs)
            print(f"[{run_id}] GRADIENT STEP (out)", flush=True)
            return out

        if update:
            print(f"[{run_id}] UPDATE", flush=True)
            out = self.do_update_tester(run_id, **kwargs)
            print(f"[{run_id}] UPDATE (out)", flush=True)
            return out

        if evaluate:
            print(f"[{run_id}] EVALUATE", flush=True)
            out = self.do_evaluate_tester(run_id, **kwargs)
            print(f"[{run_id}] EVALUATE (out)", flush=True)
            return out

        if get_accuracy:
            print(f"[{run_id}] GET ACCURACY", flush=True)
            out = self.do_get_accuracy(run_id, **kwargs)
            print(f"[{run_id}] GET ACCURACY (out)", flush=True)
            return out

        if delete_trainer:
            print(f"[{run_id}] DELETE TRAINER", flush=True)
            out = self.do_delete_trainer(run_id, **kwargs)
            print(f"[{run_id}] DELETE TRAINER (out)", flush=True)
            return out

        if delete_tester:
            print(f"[{run_id}] DELETE TESTER", flush=True)
            out = self.do_delete_tester(run_id, **kwargs)
            print(f"[{run_id}] DELETE TESTER (out)", flush=True)
            return out

        assert False
