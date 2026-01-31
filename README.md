# Hardware-Triggered Backdoors

Our project consists of a server component to which clients (the GPUs) connect to. The server needs to have `docker` installed and the clients need to have `apptainer` installed.

# Setup

In the following, we assume that you want to run the experiments on an A40 and an A100. We assume clients to be able to connect to the server via the hostname "server-hostname". You need to set up the project on the server and for the clients on all respective computers.

- On the server: Run `make docker-build-server` to build the docker container for the server (the container is named `diffmath-server`)
- On the clients: Run `make apptainer-build-client` to build the apptainer container for the clients (the container will be saved at `apptainer/gpu.sif`). Set this up on both the A40 and the A100.
- Download the ImageNet dataset and set it up in `data/imagenet`. The directory structure needs to contain `data/imagenet/train_set` and `data/imagenet/val_set` on both the server and the client.
- Download the torch models (e.g., using `src/models.py` - you need torch to run this script). The paths to the models can be set in `./start_server.sh`.

# Run

## Main experiment (Section 4.1)

On the server run:

```bash
[server] $ ./start_server.sh a40 a100
```

On the clients run:

```bash
[a40-client] $ ./start_client server-hostname a40
```

```bash
[a100-client] $ ./start_client server-hostname a100
```

The outputs will be saved on the server in `output/backdoor`. To see results for the data types `float16` or `bfloat16`, change the value of `--model_dtype float32` before starting the server. You do not need to change anything when starting the clients. The clients will receive the correct experiment configuration from the server upon start.

## Multiple Target Inputs (Section 4.2)

Before starting the experiment, change the variable `--n_poison_samples 1` in `./start_server` to the desired value.

## One-vs-Rest Trigger (Section 4.3)

Before starting the experiment, add more than two clients (e.g., A40, A100, H100) by adding another `--backends <backend_name>` to `./start_server`and then set `--do_one_vs_all true`.

## Ablation Study (Section 4.4)

Before starting the experiment, set the variables `--permute_after_gradient` and `--flip_bits_after_gradient` to control whether these steps are included in the evaluation. Note that `permute_after_gradient` can only be used on ViT.

