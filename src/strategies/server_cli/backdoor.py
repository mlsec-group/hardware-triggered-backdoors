from datasets.loader import get_datasets
from strategies.server_cli.command_line_interface import StrategyCLI


class BackdoorCLI(StrategyCLI):
    @classmethod
    def get_cmd_name(cls) -> str:
        return "backdoor"

    @classmethod
    def install_argparser(cls, subparsers):
        parser = super().install_argparser(subparsers)

        parser.add_argument("--model_type", required=True, choices=["dense", "cnn"])
        parser.add_argument(
            "--model_dtype", required=True, choices=["float32", "float16", "bfloat16"]
        )
        parser.add_argument("--model_path", required=True, type=str)
        parser.add_argument("--dataset", required=True, choices=get_datasets())
        parser.add_argument("--n_poison_samples", required=True, type=int)
        parser.add_argument("--n_iterations", required=True, type=int)
        parser.add_argument("--n_samples", required=True, type=int)
        parser.add_argument("--heuristic", required=True, type=str)
        parser.add_argument(
            "--permute_after_gradient",
            required=True,
            type=str,
            choices=["true", "false"],
        )
        parser.add_argument(
            "--flip_bits_after_gradient",
            required=True,
            type=str,
            choices=["true", "false"],
        )
        parser.add_argument(
            "--n_bits_flipped",
            required=True,
            type=int,
        )
        parser.add_argument(
            "--do_crossover",
            default="true",
            type=str,
            choices=["true", "false"],
        )
        parser.add_argument(
            "--do_one_vs_all",
            default="false",
            type=str,
            choices=["true", "false"],
        )
        parser.add_argument(
            "--use_full_model",
            default="true",
            type=str,
            choices=["true", "false"],
        )
        parser.add_argument(
            "--use_deterministic",
            default="false",
            type=str,
            choices=["true", "false"],
        )
        parser.add_argument(
            "--skip_is_prediction_close_check",
            default="false",
            type=str,
            choices=["true", "false"],
        )
