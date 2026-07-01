# Chimera experiment

The chimera experiment searches for CIFAR-10 inputs whose prediction depends on the numerical backend:

```text
same image + same CIFAR-10 model + different backend = different top-1 prediction
```

A candidate is accepted only if the server observes:

```text
predict_blis(candidate) != predict_openblas(candidate)
```

## Architecture

The experiment uses one server and three clients:

```text
server
  orchestrates the experiment and decides whether a candidate is a chimera

generator client
  receives one original CIFAR image per job
  runs the attack/search with PyTorch backpropagation
  proposes candidate batches derived from that image
  can run on CPU, CUDA, or auto-selected device

BLIS client
  forward-only oracle using PyTorch built against BLIS

OpenBLAS client
  forward-only oracle using PyTorch built against OpenBLAS
```

One job processes one original CIFAR image. Inside that job, the generator may perform many search rounds. In each round:

```text
generator -> candidate batch
server -> same candidate batch to BLIS and OpenBLAS
BLIS/OpenBLAS -> logits + predictions
server -> compares predictions and sends results back to generator
```

So all three clients participate in the same image/job, and the backend clients always evaluate the same candidate images.

## Relevant files

```text
src/datasets/cifar10.py
src/strategies/client/chimera.py
src/strategies/client/chimera_model.py
src/strategies/client/chimera_search.py
src/strategies/server/chimera/
src/strategies/server_cli/chimera.py
apptainer/chimera-generator.def
apptainer/chimera-blis.def
apptainer/chimera-openblas.def
```

## Setup

Build the server and dedicated Chimera client images:

```bash
make docker-build-server
make -j3 apptainer-build-chimera-client
```

The Chimera client setup uses three separate images:

```text
apptainer/chimera-generator.sif  generic PyTorch for candidate generation
apptainer/chimera-blis.sif       BLIS-backed PyTorch oracle
apptainer/chimera-openblas.sif   OpenBLAS-backed PyTorch oracle
```

This keeps each client to one Python environment. The BLIS and OpenBLAS builds can take a long time because PyTorch is built from source, but the images can build in parallel with `make -j3`.

If the image recipe or backend install script changed, rebuild before running:

```bash
make -j3 apptainer-build-chimera-client
```

Download CIFAR-10:

```bash
python3 scripts/download_cifar10.py
```

Expected dataset path:

```text
data/cifar10/data_batch_1
```

Place the CIFAR-10 model checkpoint here:

```text
models/cifar10/final.pt
```

## Run

For a local run on one machine, start the full server plus all three clients with:

```bash
./chimera_run.sh
```

That uses backend names `generator`, `blis`, and `openblas`, and the default CIFAR batch at `data/cifar10/data_batch_1`.
The runner checks that the BLIS image imports a BLIS-backed PyTorch and the OpenBLAS image imports an OpenBLAS-backed PyTorch before starting the campaign.

To pass names or a different batch explicitly:

```bash
./chimera_run.sh generator blis openblas data/cifar10/data_batch_1
```

For a short smoke test:

```bash
CHIMERA_N_SAMPLES=1 ./chimera_run.sh
```

For clients on different machines, start the server:

```bash
bash chimera_server.sh generator blis openblas data/cifar10/data_batch_1
```

Then start the three clients:

```bash
CHIMERA_CLIENT_IMAGE=apptainer/chimera-generator.sif bash chimera_client.sh $HOSTNAME generator
CHIMERA_CLIENT_IMAGE=apptainer/chimera-blis.sif bash chimera_client.sh $HOSTNAME blis
CHIMERA_CLIENT_IMAGE=apptainer/chimera-openblas.sif bash chimera_client.sh $HOSTNAME openblas
```

If the clients run on different machines, replace `$HOSTNAME` with the hostname or IP address of the server.

Outputs are written under:

```text
output/chimera/
```

Each run directory contains:

```text
submission.log        human-readable run log, similar to the original runner
summary.json          processed/attempted/chimera/error totals plus per-job status
sample-*/result.json  final status for one CIFAR sample
sample-*/probe-log.json
sample-*/search-log.json
sample-*/original.pt
sample-*/candidate.pt
sample-*/blis-output.pt
sample-*/openblas-output.pt
```

## Common knobs

```text
--sample_index
--n_samples
--generator_device cpu|cuda|auto
```

The Chimera search defaults live in one place: `ChimeraSearchConfig` in `src/strategies/chimera_config.py`. Change `walk_rounds`, `probe_batch_size`, `sweep_coords_per_round`, or `gd_steps` there and rerun the pipeline.

With the current defaults, one job can probe up to `walk_rounds * probe_batch_size` candidates. Most samples stop earlier once a chimera is found.

`chimera_server.sh` runs 100 CIFAR samples by default. For a short smoke test, override the sample count without editing the script:

```bash
CHIMERA_N_SAMPLES=1 bash chimera_server.sh generator blis openblas data/cifar10/data_batch_1
```
