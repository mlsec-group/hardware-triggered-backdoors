from strategies.server_cli.command_line_interface import StrategyCLI


class BackdoorDefenseCLI(StrategyCLI):
    @classmethod
    def get_cmd_name(cls) -> str:
        return "backdoor-defense"

    @classmethod
    def install_argparser(cls, subparsers):
        parser = super().install_argparser(subparsers)

        parser.add_argument(
            "--backdoor_filelist",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--max_iterations",
            required=True,
            type=int,
        )
