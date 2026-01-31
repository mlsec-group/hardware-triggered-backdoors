import io

from scheduler.client_config_meta import ClientConfigMeta
from jobscheduler.client import Client, ClientConfig

from common.interface import SHA256_SIZE
from common.network import (
    BEAT_HEADER,
    BEAT_RESPONSE,
    HEADER_SIZE,
    NEW_STRATEGY_HEADER,
    NEW_STRATEGY_RESPONSE,
    STEP_HEADER,
    DebugCon,
    pack_dict,
    pack_string,
    receive_dict,
    recv_exactly_n,
)


class NetworkClient(Client):
    def __init__(
        self, name: str, client_config: ClientConfig[ClientConfigMeta], con: DebugCon
    ):
        self.name = name
        self.client_config = client_config
        self.con = con

        self.cmd_name = None
        self.client_args = None

    def get_client_identifier(self) -> str:
        return self.client_config.client_identifier

    def get_name(self):
        return f"{self.name}-{self.client_config.client_identifier}-{self.client_config.meta.hostname}"

    def get_config(self):
        return self.client_config

    def client_arguments_match(self, cmd_name, client_args) -> bool:
        return self.cmd_name == cmd_name and self.client_args == client_args

    def client_init(self, cmd_name, client_args):
        buffer = io.BytesIO()
        buffer.write(NEW_STRATEGY_HEADER)
        buffer.write(pack_string(cmd_name))
        buffer.write(pack_dict(client_args))
        buffer.seek(0)

        data = buffer.read()
        self.con.sendall(data, tag="heart beat")

        response = recv_exactly_n(self.con, HEADER_SIZE, tag="init response")
        assert NEW_STRATEGY_RESPONSE == response

        self.cmd_name = cmd_name
        self.client_args = client_args

    def client_step(self, server_hash, client_input) -> bytes:
        buffer = io.BytesIO()
        buffer.write(STEP_HEADER)
        buffer.write(server_hash)
        buffer.write(pack_dict(client_input))
        buffer.seek(0)

        data = buffer.read()
        self.con.sendall(data)

        client_hash = recv_exactly_n(self.con, SHA256_SIZE)
        client_output = receive_dict(self.con)

        return client_hash, client_output

    def client_heartbeat(self):
        buffer = io.BytesIO()
        buffer.write(BEAT_HEADER)
        buffer.seek(0)

        data = buffer.read()
        self.con.sendall(data, tag="heart beat")

        response = recv_exactly_n(self.con, HEADER_SIZE, tag="heart beat response")
        assert BEAT_RESPONSE == response

    def close(self):
        self.con.close()
