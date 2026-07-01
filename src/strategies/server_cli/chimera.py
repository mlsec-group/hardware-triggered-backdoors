from argparse import SUPPRESS

from strategies.server_cli.command_line_interface import StrategyCLI


class ChimeraCLI(StrategyCLI):
    @classmethod
    def get_cmd_name(cls) -> str:
        return "chimera"

    @classmethod
    def install_argparser(cls, subparsers):
        parser = super().install_argparser(subparsers)
        parser.add_argument("--generator_backend", required=True, type=str)
        parser.add_argument("--blis_backend", required=True, type=str)
        parser.add_argument("--openblas_backend", required=True, type=str)
        parser.add_argument("--dataset_path", default="data/cifar10", type=str)
        parser.add_argument("--cifar_batch", default=None, type=str, help=SUPPRESS)
        parser.add_argument("--sample_index", default=0, type=int)
        parser.add_argument("--n_samples", default=100, type=int)
        parser.add_argument("--model_path", default="models/cifar10/final.pt", type=str)
        parser.add_argument(
            "--generator_device",
            default="cpu",
            choices=["cpu", "cuda", "auto"],
            type=str,
        )
