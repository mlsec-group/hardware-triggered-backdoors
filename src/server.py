#!/usr/bin/env python3

import argparse
import datetime
import json
import logging
import os
import socket
import threading
import time
from functools import partial
from typing import Dict

import torch
import torch.nn as nn
from jobscheduler.client import Client, ClientConfig
from jobscheduler.monitor import MonitorServer
from jobscheduler.progresstracker import ProgressTracker
from jobscheduler.scheduler import Scheduler
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

from common.random_names import generate_run_name
from common.util import load_dir_config
from scheduler.clients_manager import ClientsManager
from strategies.server.backdoor.strategy import BackdoorServer
from strategies.server.backdoor_defense.strategy import BackdoorDefenseServer
from strategies.server_strategy import ServerStrategy


class CleanUpManager(threading.Thread):
    def __init__(
        self,
        *,
        scheduler: Scheduler,
        global_tracker: ProgressTracker,
        clients_manager: ClientsManager,
    ):
        super().__init__()

        self.name = "CleanupManager"

        self.scheduler = scheduler
        self.global_tracker = global_tracker
        self.clients_manager = clients_manager

        self.stop_event = threading.Event()

        self.logger = logging.getLogger("CleanUpManager")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def run(self):
        while not self.stop_event.is_set():
            self.remove_dead_clients()
            self.add_new_clients()
            time.sleep(1)

    def add_new_clients(self):
        while (client := self.clients_manager.try_pop_client()) is not None:
            self.scheduler.add_client(client)

    def remove_dead_clients(self):
        while (client := self.scheduler.try_pop_clients_to_delete()) is not None:
            client: Client = client
            try:
                client.close()
                self.logger.info("Closed and removed dead client %s", client.get_name())
            except socket.error:
                self.logger.info(
                    "Encountered socket error when trying to close client %s",
                    client.get_name(),
                )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        self.stop_event.set()
        self.logger.info("Stop event set for scheduler thread %s", self.name)


def main():
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

    commands: Dict[str, ServerStrategy] = {
        c.get_cmd_name(): c
        for c in [
            BackdoorServer,
            BackdoorDefenseServer,
        ]
    }

    parser = argparse.ArgumentParser(description="Argument Parser Example")

    # Required argument
    parser.add_argument("tmp_share_dir", type=str)
    parser.add_argument("--uuid", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--commit", required=True, type=str)
    parser.add_argument("--hostname", required=True, type=str)
    parser.add_argument("--hostport", required=True, type=str)

    parser.add_argument("--server_gpu", action="store_true")

    parser.add_argument("--backends", action="append", default=[])

    subparsers = parser.add_subparsers(title="Commands", dest="command")

    for command in commands.values():
        command.install_argparser(subparsers)

    args = parser.parse_args()

    if args.server_gpu:
        assert (
            torch.cuda.is_available()
        ), "--server_gpu is set, but cuda is not available"

    dir_config = load_dir_config(args.tmp_share_dir)

    required_client_configs = [
        ClientConfig(client_identifier=backend) for backend in args.backends
    ]

    run_name = generate_run_name()
    run_path = os.path.join(dir_config.share_dir, run_name)

    os.makedirs(run_path, exist_ok=False)
    with open(os.path.join(run_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "time": datetime.datetime.now().isoformat(),
                "dir_config": dir_config.__dict__,
                "client_configs": [c.__dict__ for c in required_client_configs],
                "args": vars(args),
            },
            f,
            indent=4,
        )

    server_strategy: ServerStrategy = commands[args.command](
        run_path=run_path,
        dir_config=dir_config,
        client_configs=required_client_configs,
        seed=args.seed,
        args=args,
    )
    global_tracker = ProgressTracker(
        server_strategy.get_number_of_jobs()
        * server_strategy.get_number_of_steps_per_job(),
        log_detailed=True,
    )

    max_jobs_per_worker = {}

    scheduler = Scheduler(
        global_tracker=global_tracker,
        max_jobs_per_worker=max_jobs_per_worker,
        maxsize=10000,
        run_path=run_path,
    )

    with scheduler:
        clients_manager = ClientsManager(
            log_path=os.path.join(run_path, "client-info", "cpu"),
            host=socket.gethostname(),
            port=32334,
        )

        monitor_server = MonitorServer(
            scheduler,
            global_tracker,
            host_hostname=args.hostname,
            host_port=args.hostport,
            port=32336,
            base_directory=os.path.join("/var", "www"),
            run_path=run_path,
        )

        cleanup_manager = CleanUpManager(
            scheduler=scheduler,
            global_tracker=global_tracker,
            clients_manager=clients_manager,
        )

        with clients_manager, cleanup_manager, monitor_server:
            server_strategy.start_campaign(scheduler, global_tracker=global_tracker)

            while not scheduler.is_empty():
                time.sleep(10)

        server_strategy.final()


if __name__ == "__main__":
    main()
