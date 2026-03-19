from datasets.loader import get_datasets
from strategies.server_cli.command_line_interface import StrategyCLI


class ExampleCLI(StrategyCLI):
    @classmethod
    def get_cmd_name(cls) -> str:
        return "example"

    @classmethod
    def install_argparser(cls, subparsers):
        parser = super().install_argparser(subparsers)

        parser.add_argument("--example_server_arg", required=True, type=str)
        parser.add_argument("--example_client_arg", required=True, type=str)
