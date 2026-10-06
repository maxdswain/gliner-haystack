# gliner-haystack

## Serve GLiNER2.5 with vLLM Factory

Requires Docker Compose, the NVIDIA Container Toolkit, and an NVIDIA GPU supported by vLLM 0.20.0 (with bfloat16 support for the default settings). The Dockerfile installs the [vLLM Factory fork](https://github.com/fastino-ai/vllm-factory-fstn) from the included submodule and pins GLiNER2 to the upstream GLiNER2.5 plugin's tested PR commit; PyPI's current GLiNER2 release is too old for that plugin.

For a fresh checkout, clone with the submodule included:

```sh
git clone --recursive https://github.com/maxdswain/gliner-haystack.git
cd gliner-haystack
docker compose up --build
```

The first launch downloads `fastino/gliner2.5-multi-v1` into a persistent Hugging Face cache, prepares it in the container, and serves `POST /pooling` on port 8000. Example:

```sh
curl http://localhost:8000/pooling \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/current","data":{"text":"Alice works at Acme in Paris.","schema":{"entities":{"person":"","organization":"","location":""}},"runtime":"classic","include_spans":true}}'
```

To fall back to the upstream-documented GLiNER2 model if GLiNER2.5 fails on your hardware or environment, run:

```sh
MODEL_ID=fastino/gliner2-large-v1 MODEL_PLUGIN=deberta_gliner2 IO_PLUGIN=deberta_gliner2_io docker compose up --build
```

`DTYPE=float16` can be set for GPUs without bfloat16 support **if the installed vLLM version supports that GPU**. The prepared model directory is recreated at startup; downloaded checkpoints remain cached in the Compose volume.
