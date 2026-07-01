from typing import Dict

from strategies.client.chimera import ChimeraClient
from strategies.client_strategy import ClientStrategy


def get_client_commands() -> Dict[str, ClientStrategy]:
    commands = [ChimeraClient]

    try:
        from strategies.client.example import ExampleClient

        commands.append(ExampleClient)
    except ImportError as exc:
        print(f"Skipping example client command: {exc}")

    try:
        from strategies.client.backdoor import BackdoorClient

        commands.append(BackdoorClient)
    except ImportError as exc:
        print(f"Skipping backdoor client command: {exc}")

    try:
        from strategies.client.backdoor_defense import BackdoorDefenseClient

        commands.append(BackdoorDefenseClient)
    except ImportError as exc:
        print(f"Skipping backdoor-defense client command: {exc}")

    return {c.get_cmd_name(): c for c in commands}
