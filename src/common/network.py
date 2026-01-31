import base64
import io
import json
import socket
import struct
import sys
from typing import Dict

import torch

from common.interface import UINT_SIZE

HEADER_SIZE = 4

STEP_HEADER = b"STEP"

BEAT_HEADER = b"BEAT"
BEAT_RESPONSE = b"YES!"

NEW_STRATEGY_HEADER = b"NWST"
NEW_STRATEGY_RESPONSE = b"AYE!"

READY_MESSAGE = b"GO!?"

assert (
    HEADER_SIZE
    == len(STEP_HEADER)
    == len(BEAT_HEADER)
    == len(BEAT_RESPONSE)
    == len(READY_MESSAGE)
    == len(NEW_STRATEGY_HEADER)
    == len(NEW_STRATEGY_RESPONSE)
)


class NotEnoughData(RuntimeError):
    def __init__(self, n_exp, n_act):
        super(NotEnoughData, self).__init__(
            f"Expected {n_exp} byte{'' if n_exp == 0 else 's'} but only read {n_act}"
        )


class DebugCon:
    def __init__(self, con, name, logging=True):
        self.con = con
        self.name = name
        self.logging = logging

    def sendall(self, data, *, tag=None):
        if self.logging:
            print(f">> [{self.name}]: {tag} {len(data)}", file=sys.stderr, flush=True)
        self.con.sendall(data)
        if self.logging:
            print(f"<< [{self.name}]: {tag} {len(data)}", file=sys.stderr, flush=True)

    def recv(self, size, *, tag=None):
        if self.logging:
            print(f">> [{self.name}]: {tag} {size}", file=sys.stderr, flush=True)
        data = self.con.recv(size)
        if self.logging:
            print(f"<< [{self.name}]: {tag} {size}", file=sys.stderr, flush=True)
        return data

    def close(self):
        self.con.close()


def read_size(client: DebugCon, *, tag=None):
    return struct.unpack("!I", recv_exactly_n(client, UINT_SIZE, tag=tag))[0]


def recv_exactly_n(sock: DebugCon, n, *, tag=None):
    buffer = b""
    while (n_remaining := n - len(buffer)) > 0:
        data = sock.recv(n_remaining, tag=tag)

        if not data:
            break

        buffer += data

    if n_remaining > 0:
        raise NotEnoughData(n, len(buffer))

    return buffer


def receive_torch_object(client: DebugCon, *, weights_only=True):
    object_size = read_size(client)
    object_data = io.BytesIO(recv_exactly_n(client, object_size))
    object_data.seek(0)

    return torch.load(
        object_data, weights_only=weights_only, map_location=torch.device("cpu")
    )


def pack_torch_object(object):
    data = io.BytesIO()

    object_data = io.BytesIO()
    torch.save(object, object_data)
    object_data.seek(0)

    object_bytes = object_data.read()
    data.write(struct.pack("!I", len(object_bytes)))
    data.write(object_bytes)

    data.seek(0)
    return data.read()


def pack_string(s: str, encoding="utf-8"):
    encoded_string = s.encode(encoding)

    data = io.BytesIO()
    data.write(struct.pack("!I", len(encoded_string)))
    data.write(encoded_string)
    data.seek(0)
    return data.read()


def receive_string(client: DebugCon, *, tag=None):
    size = read_size(client, tag=f"size: {tag}")
    return recv_exactly_n(client, size, tag=tag).decode()


def pack_dict(o: Dict[str, str]):
    return pack_string(json.dumps(o, cls=TorchJSONEncoder))


def receive_dict(con: DebugCon):
    size = read_size(con)
    data = recv_exactly_n(con, size)
    return json.loads(data, object_hook=torch_json_decoder)


def torch_json_decoder(dct):
    if "__bytes__" in dct:
        b64_data: str = dct["b64_data"]
        return base64.b64decode(b64_data)
    elif "__torch_tensor__" in dct:
        b64_data: str = dct["b64_data"]
        return torch.load(
            io.BytesIO(base64.b64decode(b64_data)),
            weights_only=True,
            map_location=torch.device("cpu"),
        )

    return dct


class TorchJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            b64_data = base64.b64encode(obj)
            return {"__bytes__": True, "b64_data": b64_data.decode("utf-8")}
        elif isinstance(obj, torch.Tensor):
            data = io.BytesIO()
            torch.save(obj, data)
            data.seek(0)

            b64_data = base64.b64encode(data.read())
            return {"__torch_tensor__": True, "b64_data": b64_data.decode("utf-8")}
        return super().default(obj)
