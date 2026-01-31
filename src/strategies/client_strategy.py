import abc
import argparse


class ClientStrategy(abc.ABC):
    def __init__(self, backend):
        self.backend = backend

    @classmethod
    @abc.abstractmethod
    def get_cmd_name(cls) -> str:
        """
        Return the name of the command. The command name must be unique across
        all commands and must be same on the client and server strategy.
        """

    @classmethod
    def install_argparser(cls, subparsers) -> None:
        """
        Each strategy installs a separate command in the client and takes --backend
        as a parameter. If you want to add further parameters to the client, you
        need to create a custom `install_argparser` function and add the parameters
        to strategy_parser:

        @classmethod
        def install_argparser(cls, subparsers) -> None:
            parser = super().install_argparser(subparsers)
            parser.add_argument(
                "--my-parameter", type=int, help="Specify the my-parameter parameter"
            )
        """

        cmd_name = cls.get_cmd_name()
        strategy_parser = subparsers.add_parser(
            cmd_name, help="Add a file to the directory"
        )
        return strategy_parser

    @abc.abstractmethod
    def step(self, server_hash) -> bytes: ...
