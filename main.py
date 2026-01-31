#!/usr/bin/env python3

import argparse
import os
import socket
import subprocess
import sys
import tempfile

# This is a bit complicated: Because we can not assume any non-default python
# packages on our host system (especially numpy and torch), we first launch a
# separate "bootstrapping" container that only handles the command-line
# arguments (--> host.py).
#
# This again launches a server container (--> server.py) which initializes and
# handles the communication with the clients (--> client.py). In theory,
# host.py and server.py could be merged, but I like the separation
# between CLI and client-server communication.


def main():
    uid = os.getuid()
    project_dir = os.getcwd()
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser(
        description="Process chunk ID and number of chunks", add_help=False
    )

    parser.add_argument("--no-tty", action="store_true")
    parser.add_argument("--help", "-h", action="store_true")
    args, _ = parser.parse_known_args()

    if args.help:
        cmd = "python3 src/host.py -h"
    else:
        cmd = f"python3 src/host.py {' '.join(sys.argv[1:])}"

    with tempfile.TemporaryDirectory() as tmp_share_dir:
        proc_args = [
            "docker",
            "run",
            "--ipc=host",
            "--privileged",
            *["--env", f"HOST_UID={uid}"],
            *["--env", f"HOST_PROJECT_DIR={project_dir}"],
            *["--env", f"HOST_TMP_SHARE_DIR={tmp_share_dir}"],
            *["--env", f"HOST_HOSTNAME={hostname}"],
            f"-v/run/user/{uid}/docker.sock:/var/run/docker.sock",
            f"-v{tmp_share_dir}:{tmp_share_dir}",
            "--rm",
            "-i" if args.no_tty else "-it",
            f"-v{project_dir}:/app",
            "diffmath-server",
            "/bin/bash",
            "-c",
            cmd,
        ]
        subprocess.run(proc_args)


if __name__ == "__main__":
    main()
