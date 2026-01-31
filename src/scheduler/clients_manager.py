import json
import logging
import os
import queue
import socket
import threading
import uuid
from typing import Union

from common.network import (
    HEADER_SIZE,
    READY_MESSAGE,
    DebugCon,
    pack_string,
    receive_string,
    recv_exactly_n,
)
from scheduler.client_config_meta import ClientConfigMeta
from scheduler.network_client import NetworkClient
from jobscheduler.client import Client, ClientConfig


class ClientsManager:
    def __init__(
        self,
        log_path: str = None,
        host="0.0.0.0",
        port=32334,
    ):
        self.log_path = log_path

        self.host = host
        self.port = port

        self.clients = queue.Queue()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)
        self.server_socket.settimeout(1.0)
        self.stop_event = threading.Event()
        self._lock = threading.Lock()  # Lock to manage thread-safe client operations
        self.server_thread = None

        self.client_number = 0

        self.logger = logging.getLogger("ClientsManager")
        self.logger.disabled = True

    def _accept_clients(self):
        while not self.stop_event.is_set():
            try:
                con, _ = self.server_socket.accept()
                con = DebugCon(con, "Server", logging=False)
                try:
                    self._client_handshake(con)
                except Exception as e:  # pylint: disable=broad-except
                    self.logger.warning(
                        "Client handshake failed %s", str(e), exc_info=True
                    )
                    con.close()
            except Exception as e:  # pylint: disable=broad-except
                self.logger.warning(
                    "Accepting connection failed %s", str(e), exc_info=True
                )

    def _client_handshake(self, con: DebugCon):
        handshake_id = uuid.uuid4()

        # [S] handshake
        con.sendall(bytes(1), tag="one byte handshake")

        # [R] backend name
        backend = receive_string(con, tag="backend")

        # [R] git commit
        commit = receive_string(con, tag="git commit")

        # [R] uname info
        uname_info = receive_string(con, tag="uname info")
        hostname = uname_info.split(" ")[1]

        # [R] cpu info
        cpu_info = json.dumps(json.loads(receive_string(con, tag="cpu info")), indent=4)

        # [R] torch info
        torch_info = json.dumps(
            json.loads(receive_string(con, tag="torch info")), indent=4
        )

        # [R] torch config
        torch_config = receive_string(con, tag="torch config")

        # [S] backend
        con.sendall(pack_string(backend), tag="backend")

        # [R] client ready byte
        assert READY_MESSAGE == recv_exactly_n(
            con, HEADER_SIZE, tag="client ready byte"
        )

        with self._lock:
            meta = ClientConfigMeta(backend=backend, commit=commit, hostname=hostname)
            client = NetworkClient(
                self.client_number,
                ClientConfig(client_identifier=backend, meta=meta),
                con,
            )

            client_log_path = os.path.join(self.log_path, client.get_name())
            os.makedirs(client_log_path)

            with open(
                os.path.join(client_log_path, "backend.txt"), "w", encoding="utf-8"
            ) as f:
                print(backend, file=f)

            with open(
                os.path.join(client_log_path, "commit.txt"), "w", encoding="utf-8"
            ) as f:
                print(commit, file=f)

            with open(
                os.path.join(client_log_path, "cpu_info.json"), "w", encoding="utf-8"
            ) as f:
                print(cpu_info, file=f)

            with open(
                os.path.join(client_log_path, "uname_info.json"), "w", encoding="utf-8"
            ) as f:
                print(uname_info, file=f)

            with open(
                os.path.join(client_log_path, "torch_info.json"), "w", encoding="utf-8"
            ) as f:
                print(torch_info, file=f)

            with open(
                os.path.join(client_log_path, "torch_config.txt"), "w", encoding="utf-8"
            ) as f:
                print(torch_config, file=f)

            self.logger.info(
                "Create new client %s (backend: %s, commit: %s) from connection %s",
                client.get_name(),
                client.get_client_identifier(),
                client.client_config.meta.commit,
                str(handshake_id),
            )
            self.clients.put(client)
            self.client_number += 1

    def try_pop_client(self) -> Union[Client, None]:
        with self._lock:
            try:
                client: Client = self.clients.get_nowait()
                self.logger.info(
                    "Pop client %s (%s)",
                    client.get_name(),
                    client.get_client_identifier(),
                )
                return client
            except queue.Empty:
                return None

    def close_all_remaining_clients(self):
        with self._lock:
            while True:
                try:
                    client: Client = self.clients.get_nowait()
                except queue.Empty:
                    return

                try:
                    client.close()
                except socket.error:
                    pass

    def stop(self):
        self.stop_event.set()
        self.server_socket.close()
        if self.server_thread:
            self.server_thread.join()
        self.close_all_remaining_clients()

    def __enter__(self):
        self.server_thread = threading.Thread(target=self._accept_clients)
        self.server_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
