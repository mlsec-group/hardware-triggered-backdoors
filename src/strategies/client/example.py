from typing import Any

from strategies.client_strategy import ClientStrategy

EMPTY_HASH = bytes(32)


class ExampleClient(ClientStrategy):
    def __init__(self, backend: str, args: Any):
        super().__init__(backend)

        print(f"Starting client on backend '{backend}' with args", args)

    @classmethod
    def get_cmd_name(cls) -> str:
        return "example"

    @classmethod
    def install_argparser(cls, subparsers) -> None:
        parser = super().install_argparser(subparsers)
        parser.add_argument("--example_arg", required=True, type=str)

    def do_example_action(self, run_id, *, iteration):
        print("Doing example action with param: ", iteration)

        return EMPTY_HASH, {"y": 2**iteration}

    def step(
        self,
        server_hash,
        run_id: str,
        *,
        example_action=False,
        **kwargs,
    ):
        assert sum([example_action]) == 1

        if example_action:
            return self.do_example_action(run_id, **kwargs)
