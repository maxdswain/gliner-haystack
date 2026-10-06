# gliner-haystack

## Serve GLiNER2.5 with vLLM Factory

Requires Docker Compose and a GPU runtime supported by the selected vLLM image. The image uses the CUDA-based `vllm/vllm-openai:v0.20.0` base and requires a supported NVIDIA GPU and NVIDIA Container Toolkit (with bfloat16 support for the default settings). The published image includes the [vLLM Factory fork](https://github.com/fastino-ai/vllm-factory-fstn) and GLiNER2 pinned to the upstream GLiNER2.5 plugin's tested PR commit.

```sh
git clone https://github.com/maxdswain/gliner-haystack.git
cd gliner-haystack
docker compose up
```

Compose pulls `ghcr.io/maxdswain/vllm-gliner:latest`, built on pushes to `main` by `.github/workflows/publish-image.yml`. The package is linked to this repository by its OCI source label and the workflow's `GITHUB_TOKEN` publishing permissions, so it appears under the repository's Packages section after the first successful publish. If the package is private, authenticate with `docker login ghcr.io` using a GitHub token with `read:packages`; the package owner can make it public in GitHub's package settings. The vLLM base image is large and may exceed the disk available on a standard GitHub-hosted runner; if the build fails for disk space, use a larger or self-hosted runner.

To build locally instead, initialize the submodule and run `docker build -t ghcr.io/maxdswain/vllm-gliner:latest .` before `docker compose up`.

The first launch downloads `fastino/gliner2.5-multi-v1` into a persistent Hugging Face cache, prepares it in the container, and serves `POST /pooling` on port 8000. Example:

```sh
curl http://localhost:8000/pooling \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/current","data":{"text":"Alice works at Acme in Paris.","schema":{"entities":{"person":"","organization":"","location":""}},"runtime":"classic","include_spans":true}}'
```

To fall back to the upstream-documented GLiNER2 model, set `MODEL_ID: fastino/gliner2-large-v1`, `MODEL_PLUGIN: deberta_gliner2`, and `IO_PLUGIN: deberta_gliner2_io` in the `environment` section of `compose.yaml`. Then run `docker compose up`.

`compose.yaml` requests device ID `0` by default without specifying a GPU driver. Edit `gpus.device_ids` (for example, `["1"]`) to select a different device using IDs recognized by your container GPU runtime; listing multiple IDs only makes them available to the container, it does not make vLLM use them automatically. Removing the driver setting does not make the CUDA-based image work with non-NVIDIA GPUs.

On GPUs without bfloat16 support, set `DTYPE: float16` in `environment` **if the installed vLLM version supports that GPU**. The prepared model directory is recreated at startup; downloaded checkpoints remain cached in the Compose volume.
