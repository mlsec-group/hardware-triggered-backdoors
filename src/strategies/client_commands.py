from typing import Dict

from strategies.client.backdoor import BackdoorClient
from strategies.client.backdoor_defense import BackdoorDefenseClient
from strategies.client_strategy import ClientStrategy


def get_client_commands() -> Dict[str, ClientStrategy]:
    return {
        c.get_cmd_name(): c
        for c in [
            BackdoorClient,
            BackdoorDefenseClient,
        ]
    }
