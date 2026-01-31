#!/usr/bin/env python3

import argparse
import json
import os
import random
import subprocess
import sys

from strategies.server_cli.backdoor import BackdoorCLI
from strategies.server_cli.backdoor_defense import BackdoorDefenseCLI


def get_git_commit():
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
        return p.stdout.decode().strip()
    except Exception as e:
        return "__unavailable__"


def get_and_del(args, prop):
    val = getattr(args, prop)
    delattr(args, prop)
    return val


def build_server_args(args):
    return " ".join(
        "--" + name + " " + str(value)
        for name, value in vars(args).items()
        if value is not None
    )


def build_readonly_mount(readonly_dir, project_dir):
    mounts = []
    for d in readonly_dir:
        mounts.append("--mount")
        mounts.append(f"type=bind,source={project_dir}/{d},target=/app/{d}")
    return mounts


def build_sh_command(
    tmp_share_dir,
    uuid,
    seed,
    hostname,
    hostport,
    cmd,
    server_args,
    backends,
):
    sh_command = []
    sh_command.append("python3")
    sh_command.append("/app/src/server.py")
    sh_command.append(f"--uuid {uuid}")
    sh_command.append(f"--seed {seed}")
    sh_command.append(f"--hostname {hostname}")
    sh_command.append(f"--hostport {hostport}")
    sh_command.append(f"--commit {get_git_commit()}")

    for backend in backends:
        sh_command.append("--backends")
        sh_command.append(backend)

    sh_command.append(tmp_share_dir)
    sh_command.append(cmd)
    sh_command.append(server_args)

    return " ".join(sh_command)


def build_port_flag(port, http_port):
    if port:
        return [
            "--publish",
            f"{port}:32334",
            "--publish",
            f"{http_port}:32336",
        ]
    else:
        return []


def start(
    uid,
    project_dir,
    tmp_share_dir,
    cmd,
    share_dir,
    readonly_dir,
    seed,
    hostname,
    no_tty,
    uuid,
    port,
    http_port,
    backends,
    args,
):
    print(
        f"Run 'python3 src/client_remote.py --hostname {hostname} --port {port}' to connect"
    )

    datasets_dir = "data"
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(share_dir, exist_ok=True)

    dir_config = {
        "project_dir": project_dir,
        "share_dir": share_dir,
        "readonly_dir": readonly_dir,
        "datasets_dir": datasets_dir,
        "tmp_share_dir": tmp_share_dir,
    }

    with open(
        os.path.join(tmp_share_dir, "dir_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(dir_config, f)

    server_args = build_server_args(args)

    readonly_mount = build_readonly_mount(readonly_dir, project_dir)
    sh_command = build_sh_command(
        tmp_share_dir,
        uuid,
        seed,
        hostname,
        port,
        cmd,
        server_args,
        backends,
    )
    port_flag = build_port_flag(port, http_port)

    proc_args = [
        "docker",
        "run",
        "--rm",
        "-i" if no_tty else "-it",
        "--ipc=host",
        "--privileged",
        *port_flag,
        f"-v/run/user/{uid}/docker.sock:/var/run/docker.sock",
        f"-v{project_dir}/src/__init__.py:/app/src/__init__.py:ro",
        f"-v{project_dir}/jobscheduler/src/jobscheduler:/app/src/jobscheduler:ro",
        f"-v{project_dir}/src/models.py:/app/src/models.py:ro",
        f"-v{project_dir}/src/server.py:/app/src/server.py:ro",
        f"-v{project_dir}/src/common:/app/src/common:ro",
        f"-v{project_dir}/src/scheduler:/app/src/scheduler:ro",
        f"-v{project_dir}/src/datasets:/app/src/datasets:ro",
        f"-v{project_dir}/src/strategies/__init__.py:/app/src/strategies/__init__.py:ro",
        f"-v{project_dir}/src/strategies/server_strategy.py:/app/src/strategies/server_strategy.py:ro",
        f"-v{project_dir}/src/strategies/server:/app/src/strategies/server:ro",
        f"-v{project_dir}/src/strategies/server_cli:/app/src/strategies/server_cli:ro",
        f"-v{tmp_share_dir}:{tmp_share_dir}",
        f"-v{project_dir}/{datasets_dir}:/app/{datasets_dir}",
        f"-v{project_dir}/www:/var/www",
        "--mount",
        f"type=bind,source={project_dir}/{share_dir},target=/app/{share_dir}",
        *readonly_mount,
        "diffmath-server",
        "/bin/bash",
        "-c",
        sh_command,
    ]

    print("Starting: ", "diffmath-server")
    p = subprocess.run(proc_args)
    if p.returncode != 0:
        print("Server crashed", file=sys.stderr)
        sys.exit(1)


def main():
    commands = {
        c.get_cmd_name(): c
        for c in [
            BackdoorCLI,
            BackdoorDefenseCLI,
        ]
    }

    parser = argparse.ArgumentParser(
        description="Process chunk ID and number of chunks"
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="(required) Provide a global seed for the command",
    )

    parser.add_argument(
        "--share_dir",
        required=True,
        type=str,
        help=(
            "(required) This directory (provided relative to current working dir) "
            "is shared across host, server and client. It is used to permanently "
            "save information (e.g., results from experiments)"
        ),
    )

    parser.add_argument(
        "--uuid",
        default="-".join(str(random.randint(1000, 9999)) for i in range(5)),
        type=str,
        help="A global unique identifier to identifier a group of runs (e.g., the same experiment with different parameters)",
    )

    parser.add_argument(
        "--readonly_dir",
        type=str,
        action="append",
        default=[],
        help="Same as shared_dir but readonly (e.g., share trained models)",
    )

    parser.add_argument(
        "--no-tty",
        action="store_true",
        help="Do not pass the -t option to docker containers (--no-tty needs to be set when using gnu parallel)",
    )

    parser.add_argument(
        "--port",
        type=int,
        help="Port which external clients can connect to",
    )

    parser.add_argument(
        "--http_port",
        type=int,
        help="Port for serving http monitor",
    )

    parser.add_argument("--backends", action="append", default=[])

    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    for command in commands.values():
        command.install_argparser(subparsers)

    args = parser.parse_args()

    start(
        uid=int(os.environ["HOST_UID"]),
        project_dir=os.environ["HOST_PROJECT_DIR"],
        tmp_share_dir=os.environ["HOST_TMP_SHARE_DIR"],
        hostname=os.environ["HOST_HOSTNAME"],
        cmd=get_and_del(args, "command"),
        share_dir=get_and_del(args, "share_dir"),
        readonly_dir=get_and_del(args, "readonly_dir"),
        seed=get_and_del(args, "seed"),
        no_tty=get_and_del(args, "no_tty"),
        uuid=get_and_del(args, "uuid"),
        port=get_and_del(args, "port"),
        http_port=get_and_del(args, "http_port"),
        backends=get_and_del(args, "backends"),
        args=args,
    )


if __name__ == "__main__":
    main()
