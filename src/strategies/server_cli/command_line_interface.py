import abc


class StrategyCLI(abc.ABC):
    @classmethod
    def install_argparser(cls, subparsers, help_text="") -> None:
        """
        This command lets you install parameters for the server call. When
        starting the server (e.g., through host.py) we need to provide the
        parameters to the process.
        """
        cmd_name = cls.get_cmd_name()
        strategy_parser = subparsers.add_parser(cmd_name, help=help_text)
        return strategy_parser
