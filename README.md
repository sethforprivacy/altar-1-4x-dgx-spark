# Altar-1 on four DGX Sparks

A pinned recipe for serving [`AikidoSec/altar-1`](https://huggingface.co/AikidoSec/altar-1) with SparkRing on four 128 GB NVIDIA DGX Sparks. Each host stores a full model copy. The runtime uses one GPU per host, TP4/DCP4, FP8 KV cache, and a switchless RoCE ring.

The example configuration sets a 327,680-token total request limit, four concurrent requests, an 8,192-token prefill batch, and a 17 GiB KV pool per host. Adjust these limits only after checking memory headroom on your own hardware.

## Pinned inputs

| Input | Revision |
| --- | --- |
| Model | `AikidoSec/altar-1@5d591cd98958e4e7e429517887e80ca3bb7ce2c4` |
| SparkRing source | `f16b5f43208c188c018c43820ccf52955006d1a5` |
| Base image | `ghcr.io/fujitsupolycom/sparkring@sha256:2375f876bc9ea065e85ae10cebad7a8db8a2ec0e6862b4441c269c5bf56365c6` |

The checkpoint is approximately 328 GB on each host. Allow additional space for the image and compilation cache. The image build verifies upstream source hashes before applying the included patches.

## Prepare the hosts

Install Docker and NVIDIA Container Toolkit on each Spark. Configure the four-host RoCE ring using [NETWORK.md](NETWORK.md), and verify all four ranks can reach their neighbors. Keep management SSH separate from the model fabric.

Download the pinned model revision on every host:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-download.txt
mkdir -p results
python download_model.py /srv/models/altar-1/5d591cd98958e4e7e429517887e80ca3bb7ce2c4 \
  --receipt results/model-verification.json
```

The downloader resumes interrupted transfers. `verify_model.py MODEL_DIR --receipt results/model-verification.json` checks downloaded files against the model revision. Keep Hugging Face credentials outside the repository.

Build the serving image on each ARM64 host:

```sh
docker build --network=none --build-arg SOURCE_DATE_EPOCH=1790035200 \
  -t local/altar-1-sparkring:recipe .
docker image inspect --format '{{.Id}}' local/altar-1-sparkring:recipe
```

Use the resulting image ID in each host's configuration. Compare IDs across hosts before launch. The pinned base image must be available locally when building without network access.

## Configure and launch

Copy `rank.env.example` to a private file on each host. Set the host's management address, rank-zero address, connected HCA ports, model and cache paths, and image ID. Put a dedicated API key in a mode-0600 file outside this repository and point `API_KEY_FILE` at it. Check the chosen HCA names and GID index on every host; the values in the example are hardware-specific.

Copy `inventory.example.json` to a private `site.json` on the control host. Set four SSH destinations in physical ring order, with matching recipe, environment-file, and cache paths. Verify host keys and noninteractive SSH access to Docker on each host. Then launch the controller:

```sh
python3 run-ring.py --inventory /absolute/path/site.json --log /private/path/ring-memory.jsonl
```

The controller starts memory guards, launches all four ranks, and stops the ring if a rank fails. Run it from a persistent session or adapt `systemd/altar-ring.service.example`. The API binds to loopback on rank zero by default; access it through SSH forwarding or an authenticated private proxy.

For manual diagnosis, use the matching rank number and private environment file on each host:

```sh
bash run-rank.sh 0 /absolute/path/rank.env
```

The launcher refuses to replace an existing container. It creates a fresh FlashInfer tuning directory on every launch while retaining the compilation cache. Stop the controller and inspect all four containers before changing the model, image, or configuration.

## Validate operation

Run the ring collective probe before loading the full model. On each host, use a separate rank configuration with a distinct `CONTAINER_PREFIX`, then run `bash run-rank.sh RANK /absolute/path/rank.env ring-probe` for ranks 0–3. Require exit code zero on all four hosts.

After the model starts, run the API and prefix-cache checks against rank zero:

```sh
mkdir -p results
python3 tests/qualify_api.py --url http://127.0.0.1:8025 \
  --key-file /private/path/altar.key --output results/api.json
python3 tests/prefix_cache.py --url http://127.0.0.1:8025 \
  --key-file /private/path/altar.key --output results/prefix-cache.json
```

Keep generated results, logs, host inventories, credentials, and rank configurations outside version control. The local watchdog stops a rank after two consecutive one-second samples below 6 GiB available host memory, or immediately below 2 GiB. Do not lower these thresholds to make a launch succeed.

Run CPU-only checks before changing the launcher or guards:

```sh
python3 tests/test_memory_guard.py
python3 tests/test_controller_shutdown.py
python3 tests/test_ckv_launcher.py
```

## Licenses

Recipe code and documentation are MIT licensed. Model weights, base image, and derived upstream patches retain their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
