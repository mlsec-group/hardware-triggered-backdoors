from pathlib import Path
from typing import Any, Dict

import torch

from strategies.client.chimera_model import load_cifar_vgg
from strategies.chimera_config import ChimeraSearchConfig
from strategies.client.chimera_search import ChimeraSearch
from strategies.client_strategy import ClientStrategy

EMPTY_HASH = bytes(32)


class ChimeraClient(ClientStrategy):
    def __init__(self, backend: str, args: Any):
        super().__init__(backend)
        self.args = args
        self.role = self._determine_role(backend, args)
        self.searches: Dict[str, ChimeraSearch] = {}
        self.device = (
            self._select_device(args.generator_device)
            if self.role == "generator"
            else torch.device("cpu")
        )

        self.model = load_cifar_vgg(Path(args.model_path), self.device)
        self.model.to(self.device)
        self.model.eval()

        print(
            f"Starting chimera client backend={backend!r} role={self.role!r} "
            f"device={self.device} model={args.model_path}"
        )

    @classmethod
    def get_cmd_name(cls) -> str:
        return "chimera"

    @classmethod
    def install_argparser(cls, subparsers) -> None:
        parser = super().install_argparser(subparsers)
        parser.add_argument("--generator_backend", required=True, type=str)
        parser.add_argument("--blis_backend", required=True, type=str)
        parser.add_argument("--openblas_backend", required=True, type=str)
        parser.add_argument(
            "--model_path", default="models/cifar10/final.pt", type=str
        )
        parser.add_argument(
            "--generator_device",
            default="cpu",
            choices=["cpu", "cuda", "auto"],
            type=str,
        )

    def _determine_role(self, backend: str, args: Any) -> str:
        if backend == args.generator_backend:
            return "generator"
        if backend in (args.blis_backend, args.openblas_backend):
            return "oracle"
        raise ValueError(
            f"Backend {backend!r} is not configured as generator, BLIS, or OpenBLAS"
        )

    def _select_device(self, requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--generator_device cuda requested but CUDA is unavailable")
            return torch.device("cuda")
        return torch.device("cpu")

    def _config(self) -> ChimeraSearchConfig:
        return ChimeraSearchConfig()

    def _require_generator(self):
        if self.role != "generator":
            raise RuntimeError(f"Action requires generator role, got {self.role}")

    def _require_oracle(self):
        if self.role != "oracle":
            raise RuntimeError(f"Action requires oracle role, got {self.role}")

    def generator_start(self, run_id: str, *, image: torch.Tensor, label: int, seed: int):
        self._require_generator()
        search = ChimeraSearch(
            self.model,
            image,
            int(label),
            device=self.device,
            seed=int(seed),
            config=self._config(),
        )
        self.searches[run_id] = search
        return EMPTY_HASH, search.status()

    def generator_next(self, run_id: str):
        self._require_generator()
        search = self.searches[run_id]
        return EMPTY_HASH, search.next_probe_batch()

    def generator_update(
        self,
        run_id: str,
        *,
        blis_predictions: torch.Tensor,
        openblas_predictions: torch.Tensor,
        blis_logits: torch.Tensor,
        openblas_logits: torch.Tensor,
    ):
        self._require_generator()
        search = self.searches[run_id]
        return EMPTY_HASH, search.update_with_probe_results(
            blis_predictions=blis_predictions,
            openblas_predictions=openblas_predictions,
            blis_logits=blis_logits,
            openblas_logits=openblas_logits,
        )

    def oracle_infer(self, *, candidates: torch.Tensor):
        self._require_oracle()
        with torch.inference_mode():
            batch = candidates.detach().to(torch.device("cpu"), dtype=torch.float32)
            if batch.ndim == 3:
                batch = batch.unsqueeze(0)
            logits = self.model(batch).detach().cpu()
            predictions = torch.argmax(logits, dim=1).to(torch.long)
        return EMPTY_HASH, {"logits": logits, "predictions": predictions}

    def step(
        self,
        server_hash,
        run_id: str = "",
        *,
        generator_start=False,
        generator_next=False,
        generator_update=False,
        oracle_infer=False,
        **kwargs,
    ):
        assert (
            sum([generator_start, generator_next, generator_update, oracle_infer]) == 1
        )

        if generator_start:
            return self.generator_start(run_id, **kwargs)
        if generator_next:
            return self.generator_next(run_id)
        if generator_update:
            return self.generator_update(run_id, **kwargs)
        if oracle_infer:
            return self.oracle_infer(**kwargs)

        raise AssertionError("unreachable")
