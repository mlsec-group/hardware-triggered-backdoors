import argparse
import datetime
import io
import json
import logging
import os
import re
import socket
import struct
import subprocess
import time
from functools import partial, reduce

import torch
import torch.nn as nn
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torchvision.models.efficientnet import EfficientNet, FusedMBConv, MBConv
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision.models.vision_transformer import (
    Encoder,
    EncoderBlock,
    MLPBlock,
    VisionTransformer,
)
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth

from common.interface import SHA256_SIZE, UINT_SIZE
from common.network import (
    BEAT_HEADER,
    BEAT_RESPONSE,
    HEADER_SIZE,
    NEW_STRATEGY_HEADER,
    NEW_STRATEGY_RESPONSE,
    READY_MESSAGE,
    STEP_HEADER,
    DebugCon,
    NotEnoughData,
    pack_dict,
    pack_string,
    receive_dict,
    receive_string,
    recv_exactly_n,
)
from strategies.client_commands import get_client_commands
from strategies.client_strategy import ClientStrategy


def get_git_commit():
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
        return p.stdout.decode().strip()
    except Exception:  # pylint: disable=broad-except
        return "<unavailable>"


def get_uname_info():
    try:
        p = subprocess.run(["uname", "-a"], stdout=subprocess.PIPE)
        return p.stdout.decode().strip()
    except Exception:  # pylint: disable=broad-except
        return "<unavailable>"


def get_cpuinfo():
    cpus = []
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f.readlines():
                if "model name" in line:
                    cpus.append(line.strip())
    except FileNotFoundError:
        p = subprocess.run(["sysctl", "-a"], stdout=subprocess.PIPE)
        cpus.append(p.stdout.decode())
    return json.dumps(cpus)


def get_torch_info():
    obj = {
        "git_version": getattr(torch.version, "git_version", None),
        "cuda": getattr(torch.version, "cuda", None),
        "debug": getattr(torch.version, "debug", None),
        "cuda.is_available": torch.cuda.is_available(),
        "cuda.is_bf16_supported": torch.cuda.is_bf16_supported(),
    }
    return json.dumps(obj)



def get_client_args(con: DebugCon):
    size = struct.unpack("!I", recv_exactly_n(con, UINT_SIZE, tag="client args size"))[
        0
    ]
    client_args = recv_exactly_n(con, size, tag="client args").decode()
    return json.loads(client_args)


def hack_to_inject_bash_parsing_for_args(client_args):
    try:
        str_args = " ".join(client_args)
    except TypeError as e:
        print(client_args)
        raise e

    arg = (
        'python3 -c "import sys; import json;print(json.dumps(sys.argv[1:]))" '
        + str_args
    )

    p = subprocess.run(["/bin/bash", "-c", arg], stdout=subprocess.PIPE)
    return json.loads(p.stdout.decode())


def do_step(
    con: DebugCon,
    client_strategy: ClientStrategy,
    iteration: int,
    frame_size: int,
    wait_durations: torch.Tensor,
    step_durations: torch.Tensor,
):
    start = time.time()
    server_hash = recv_exactly_n(con, SHA256_SIZE, tag="server hash")
    end = time.time()
    wait_durations[iteration % frame_size] = end - start

    if not server_hash:
        raise StopIteration()

    input_tensors = receive_dict(con)

    start = time.time()
    client_hash, output_tensors = client_strategy.step(server_hash, **input_tensors)
    end = time.time()
    step_durations[iteration % frame_size] = end - start

    buffer = io.BytesIO()
    buffer.write(client_hash)
    buffer.write(pack_dict(output_tensors))
    buffer.seek(0)

    con.sendall(buffer.read(), tag="step")


def do_heartbeat(con: DebugCon):
    con.sendall(BEAT_RESPONSE, tag="heartbeat")


def loop(backend: str, con: DebugCon):
    con.sendall(READY_MESSAGE, tag="client ready byte")

    client_strategy = None

    frame_size = 1000
    wait_durations = torch.full((frame_size,), torch.nan)
    step_durations = torch.full((frame_size,), torch.nan)

    log_frequency = 60
    last_log = time.time()

    logger = logging.getLogger(f"[{backend}]")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    iteration = 0
    while True:
        iteration += 1

        now = time.time()
        if abs(last_log - now) > log_frequency or iteration % frame_size == 0:
            logger.info(
                "Average runtimes: Wait (%.*f s), Step (%.*f s)",
                2,
                torch.nanmean(wait_durations).item(),
                2,
                torch.nanmean(step_durations).item(),
            )
            last_log = now

        header = recv_exactly_n(con, HEADER_SIZE, tag="header")

        if header == STEP_HEADER:
            logger.info("Got STEP_HEADER")
            assert client_strategy is not None
            do_step(
                con,
                client_strategy,
                iteration,
                frame_size,
                wait_durations,
                step_durations,
            )
        elif header == BEAT_HEADER:
            logger.info("Got BEAT_HEADER")
            do_heartbeat(con)
        elif header == NEW_STRATEGY_HEADER:
            # [R] cmd name
            cmd_name = receive_string(con, tag="cmd name")
            # [R] client args
            client_args = get_client_args(con)
            logger.info("Got NEW_STRATEGY_HEADER (%s)", cmd_name)
            client_strategy = create_client_strategy(backend, cmd_name, client_args)
            con.sendall(NEW_STRATEGY_RESPONSE, tag="new-strategy")
        else:
            assert False, f"Received unknown header: {header}"


def init(con: DebugCon, *, backend_name=None):
    # [R] handshake
    recv_exactly_n(con, 1, tag="one byte handshake")

    # [S] backend name
    con.sendall(pack_string(backend_name), tag="backend")

    # [S] git commit
    con.sendall(pack_string(get_git_commit()), tag="git commit")

    # [S] uname info
    con.sendall(pack_string(get_uname_info()), tag="uname info")

    # [S] cpu info
    con.sendall(pack_string(get_cpuinfo()), tag="cpu info")

    # [S] torch info
    con.sendall(pack_string(get_torch_info()), tag="torch info")

    # [S] torch config
    con.sendall(pack_string(torch.__config__.show()), tag="torch config")

    # [R] backend
    backend = receive_string(con, tag="backend")
    assert backend == backend_name, f"{backend} == {backend_name}"

    return backend


def try_print_cpu_info():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f.readlines():
                if "model name" in line:
                    # print(line)
                    pass
    except Exception:  # pylint: disable=broad-except
        pass


def create_client_strategy(backend, cmd_name, client_args):
    commands = get_client_commands()

    parser = argparse.ArgumentParser(description="Argument Parser Example")
    subparsers = parser.add_subparsers(title="Commands", dest="command")

    strategy_cls = commands[cmd_name]
    strategy_cls.install_argparser(subparsers)

    if len(client_args) > 0:
        parser_args = [cmd_name] + list(reduce(lambda x, y: x + y, client_args.items()))
    else:
        parser_args = [cmd_name]
    parser_args = hack_to_inject_bash_parsing_for_args(parser_args)

    args = parser.parse_args(parser_args)
    return strategy_cls(backend, args)


def start_client(con: DebugCon, backend_name):
    try_print_cpu_info()

    backend = init(con, backend_name=backend_name)

    start = datetime.datetime.now()
    try:
        loop(backend, con)
    except (socket.error, NotEnoughData) as e:
        if datetime.datetime.now() - start < datetime.timedelta(seconds=10):
            raise e


def main():
    # dirty_evil_hack_to_download_imagenet("models/imagenet", "src/datasets/imagenet.py")
    torch.serialization.add_safe_globals(
        [
            EfficientNet,
            set,
            Conv2dNormActivation,
            FusedMBConv,
            MBConv,
            StochasticDepth,
            SqueezeExcitation,
            BasicBlock,
            nn.AvgPool2d,
            nn.AdaptiveAvgPool2d,
            nn.BatchNorm2d,
            nn.Conv2d,
            nn.Dropout,
            nn.GELU,
            nn.LayerNorm,
            nn.Linear,
            nn.MaxPool2d,
            nn.MultiheadAttention,
            nn.ReLU,
            nn.Sequential,
            nn.SiLU,
            nn.Sigmoid,
            ResNet,
            NonDynamicallyQuantizableLinear,
            VisionTransformer,
            Encoder,
            EncoderBlock,
            MLPBlock,
            partial,
        ]
    )

    parser = argparse.ArgumentParser(description="Argument Parser Example")
    parser.add_argument("--hostname", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--backend")
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as con:
        con.connect((args.hostname, args.port))
        con = DebugCon(con, "Client", logging=False)
        start_client(con, backend_name=args.backend)


if __name__ == "__main__":
    main()
